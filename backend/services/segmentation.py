"""
Tree segmentation service using R/lidR
"""
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import laspy
import logging

logger = logging.getLogger(__name__)

# Path to the R segmentation script
SCRIPT_DIR = Path(__file__).parent.parent.parent / "src"
R_SCRIPT = SCRIPT_DIR / "segment_trees.R"


def _compute_geoid_undulation() -> float:
    """
    Compute Brussels TAW -> ellipsoidal offset using the same logic as
    segmentation/filter_vegetation.py.
    """
    import pyproj

    try:
        pyproj.network.set_network_enabled(True)
        t = pyproj.Transformer.from_crs("EPSG:6190", "EPSG:4979", always_xy=True)
        _, _, h_ellipsoidal = t.transform(150000.0, 170000.0, 0.0)
        return float(h_ellipsoidal)
    except Exception:
        # Stable fallback when vertical grids/network are unavailable.
        return 42.876090428684414


def _build_transformers(crs: str):
    """
    Build horizontal and 3D transformers for the given input CRS.

    - horizontal: EPSG:<crs> -> EPSG:4326
    - 3d:         input+height -> EPSG:4979 (lon/lat/ellipsoidal height)
    """
    import pyproj

    horizontal = pyproj.Transformer.from_crs(f"EPSG:{crs}", "EPSG:4326", always_xy=True)

    # Use compound CRS with TAW heights when input is Belgian Lambert 72.
    if str(crs) == "31370":
        try:
            pyproj.network.set_network_enabled(True)
            three_d = pyproj.Transformer.from_crs("EPSG:6190", "EPSG:4979", always_xy=True)
            return horizontal, three_d, 0.0
        except Exception:
            # Fallback: horizontal transform + geoid correction.
            return horizontal, None, _compute_geoid_undulation()

    try:
        three_d = pyproj.Transformer.from_crs(f"EPSG:{crs}", "EPSG:4979", always_xy=True)
        return horizontal, three_d, 0.0
    except Exception:
        return horizontal, None, 0.0


def run_segmentation(
    input_path: Path,
    output_dir: Path,
    crs: str = "31370",
    subset: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run tree segmentation on a point cloud file.
    
    Args:
        input_path: Path to input LAS/LAZ file
        output_dir: Directory for output files
        crs: EPSG code for the input data
        subset: Optional subset bounds "xmin,ymin,xmax,ymax"
        
    Returns:
        Dictionary with segmentation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_las = output_dir / "segmented.las"
    
    # Build command
    cmd = ["Rscript", str(R_SCRIPT), str(input_path), str(output_las), crs]
    if subset:
        cmd.extend(["--subset", subset])
    
    logger.info(f"Running segmentation: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 60 minute timeout for large files
        )
        
        if result.returncode != 0:
            logger.error(f"Segmentation failed: {result.stderr}")
            raise RuntimeError(f"Segmentation failed: {result.stderr}")
        
        logger.info(f"Segmentation output: {result.stdout[-500:]}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Segmentation timed out after 60 minutes")
    
    if not output_las.exists():
        raise RuntimeError("Segmentation did not produce output file")
    
    # Extract tree metrics from segmented point cloud (with WGS84 conversion)
    trees = extract_tree_metrics(output_las, output_dir, crs)
    
    return {
        "segmented_las": str(output_las),
        "individual_trees_dir": str(output_dir / "individual_trees"),
        "trees": trees
    }


def extract_tree_metrics(segmented_las: Path, output_dir: Path, crs: str = "31370") -> List[Dict[str, Any]]:
    """
    Extract metrics for each segmented tree from the LAS file.
    Coordinates are converted to WGS84 (lon/lat).
    """
    import pyproj
    
    las = laspy.read(str(segmented_las))
    
    # Create transformers to WGS84
    transformer_xy, transformer_3d, geoid_offset = _build_transformers(crs)
    
    # Check for TreeID attribute
    if "TreeID" not in las.point_format.dimension_names:
        raise RuntimeError("TreeID not found in segmented point cloud")
    
    tree_ids = np.unique(las.TreeID)
    tree_ids = tree_ids[tree_ids > 0]  # Exclude ID 0 (unclassified)
    
    trees = []
    individual_dir = output_dir / "individual_trees"
    individual_dir.mkdir(exist_ok=True)
    
    for tid in tree_ids:
        mask = las.TreeID == tid
        points = las.points[mask]
        
        x = np.array(las.x[mask])
        y = np.array(las.y[mask])
        z = np.array(las.z[mask])
        # Use preserved orthometric heights when available (written by segment_trees.R).
        if "Z_TAW" in las.point_format.dimension_names:
            z_for_geo = np.array(las.Z_TAW[mask], dtype=float)
        else:
            z_for_geo = z
        
        # Convert center to WGS84
        center_x, center_y = float(x.mean()), float(y.mean())
        center_z = float(z_for_geo.mean())
        print(center_z)
        lon, lat = transformer_xy.transform(center_x, center_y)
        if transformer_3d is not None:
            _, _, center_h_ellipsoid = transformer_3d.transform(center_x, center_y, center_z)
        else:
            center_h_ellipsoid = center_z + geoid_offset
        
        # Convert bbox corners to WGS84 + ellipsoidal Z
        x_min, y_min, z_min = float(x.min()), float(y.min()), float(z_for_geo.min())
        x_max, y_max, z_max = float(x.max()), float(y.max()), float(z_for_geo.max())
        bbox_min_lon, bbox_min_lat = transformer_xy.transform(x_min, y_min)
        bbox_max_lon, bbox_max_lat = transformer_xy.transform(x_max, y_max)
        if transformer_3d is not None:
            _, _, z_min_ellipsoid = transformer_3d.transform(x_min, y_min, z_min)
            _, _, z_max_ellipsoid = transformer_3d.transform(x_max, y_max, z_max)
        else:
            z_min_ellipsoid = z_min + geoid_offset
            z_max_ellipsoid = z_max + geoid_offset
        
        # Estimate crown diameter as max horizontal extent (in meters)
        crown_diameter = max(x.max() - x.min(), y.max() - y.min())
        
        # Calculate metrics
        tree_info = {
            "tree_id": int(tid),
            "height": float(z.max() - z.min()),
            "n_points": int(len(x)),
            # WGS84 coordinates (longitude, latitude)
            "longitude": float(lon),
            "latitude": float(lat),
            # Original coordinates (for reference)
            "x": center_x,
            "y": center_y,
            # Ellipsoidal elevation (EPSG:4979 / WGS84)
            "z_center_ellipsoid": float(center_h_ellipsoid),
            "z_min": float(z_min_ellipsoid),
            "z_max": float(z_max_ellipsoid),
            # Bounding box in WGS84 [lon, lat, ellipsoidal z]
            "bbox_min": [bbox_min_lon, bbox_min_lat, float(z_min_ellipsoid)],
            "bbox_max": [bbox_max_lon, bbox_max_lat, float(z_max_ellipsoid)],
            "crown_diameter": float(crown_diameter),
        }
        trees.append(tree_info)
        
        # Save individual tree point cloud
        tree_las_path = individual_dir / f"tree_{tid}.las"
        if not tree_las_path.exists():
            # Create a new LAS file for this tree
            tree_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
            tree_las = laspy.LasData(tree_header)
            tree_las.points = points
            tree_las.write(str(tree_las_path))
    
    logger.info(f"Extracted metrics for {len(trees)} trees")
    return trees


def get_point_cloud_info(las_path: Path, crs: str = "31370") -> Dict[str, Any]:
    """
    Get basic information about a point cloud file.
    """
    las = laspy.read(str(las_path))
    
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    
    # Calculate center in WGS84
    import pyproj
    transformer_xy, transformer_3d, geoid_offset = _build_transformers(crs)
    center_x, center_y = x.mean(), y.mean()
    center_z = float(z.mean())
    lon, lat = transformer_xy.transform(center_x, center_y)
    if transformer_3d is not None:
        _, _, z_ellipsoid = transformer_3d.transform(float(center_x), float(center_y), center_z)
    else:
        z_ellipsoid = center_z + geoid_offset
    
    return {
        "n_points": len(las.points),
        "crs": f"EPSG:{crs}",
        "bbox_min": [float(x.min()), float(y.min()), float(z.min())],
        "bbox_max": [float(x.max()), float(y.max()), float(z.max())],
        "center": [float(lon), float(lat), float(z_ellipsoid)]
    }


def convert_to_web_format(las_path: Path, output_path: Path, crs: str = "31370") -> Path:
    """
    Convert LAS file to a web-friendly format (JSON with positions).
    Includes coordinate transformation to WGS84.
    """
    import pyproj
    
    las = laspy.read(str(las_path))
    
    # Transform coordinates to WGS84
    transformer = pyproj.Transformer.from_crs(f"EPSG:{crs}", "EPSG:4326", always_xy=True)
    
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    
    # Transform all points
    lons, lats = transformer.transform(x, y)
    
    # Get colors if available
    if hasattr(las, 'red'):
        colors = np.column_stack([
            las.red / 65535 * 255,
            las.green / 65535 * 255,
            las.blue / 65535 * 255
        ]).astype(np.uint8).tolist()
    else:
        # Default green color for vegetation
        colors = [[34, 139, 34]] * len(x)
    
    # Get TreeID if available
    tree_ids = None
    if "TreeID" in las.point_format.dimension_names:
        tree_ids = las.TreeID.tolist()
    
    # Subsample if too many points (for web performance)
    max_points = 500000
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        lons = lons[indices]
        lats = lats[indices]
        z = z[indices]
        colors = [colors[i] for i in indices]
        if tree_ids:
            tree_ids = [tree_ids[i] for i in indices]
    
    # Create output data
    data = {
        "positions": [[float(lon), float(lat), float(alt)] for lon, lat, alt in zip(lons, lats, z)],
        "colors": colors,
        "tree_ids": tree_ids,
        "count": len(lons)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return output_path

