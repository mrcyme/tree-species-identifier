"""
Tree Shape Prediction Service

Generates a parametric 3D mesh (cylinder trunk + ellipsoid crown) that
matches the tree's point cloud dimensions, inspired by the approach in
LiDAR-3D-Urban-Forest-Mapping/convert_trees_to_gltf.py.

All output vertices are in the same local coordinate space as the input
point cloud (centred at the tree's XY centroid, Z relative to ground).
"""
import numpy as np
import trimesh
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _estimate_dimensions(points: np.ndarray) -> Dict[str, float]:
    """
    Estimate tree dimensions from its point cloud.

    Uses a simple heuristic: the bottom 25 % of the height range is treated
    as trunk, the top 75 % as crown.  The trunk radius is derived from the
    lateral spread of the lower points; the crown radius from the upper points.

    Args:
        points: (N, 3) array in local/centred coordinates.

    Returns:
        Dictionary with h_tree, h_trunk, h_crown, crown_radius_xy, trunk_radius.
    """
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    h_tree = z_max - z_min

    # Split trunk / canopy at 25 % height
    trunk_threshold = z_min + h_tree * 0.25
    trunk_pts = points[z < trunk_threshold]
    crown_pts = points[z >= trunk_threshold]

    h_trunk = trunk_threshold - z_min
    h_crown = z_max - trunk_threshold

    # Crown horizontal radius — 95th-percentile of distances from centroid
    if len(crown_pts) >= 4:
        cx, cy = crown_pts[:, 0].mean(), crown_pts[:, 1].mean()
        dists = np.sqrt((crown_pts[:, 0] - cx) ** 2 + (crown_pts[:, 1] - cy) ** 2)
        crown_radius_xy = float(np.percentile(dists, 95))
    else:
        crown_radius_xy = h_crown * 0.5  # fallback

    # Trunk radius — 95th-percentile of distances in trunk zone
    if len(trunk_pts) >= 4:
        tx, ty = trunk_pts[:, 0].mean(), trunk_pts[:, 1].mean()
        dists_t = np.sqrt((trunk_pts[:, 0] - tx) ** 2 + (trunk_pts[:, 1] - ty) ** 2)
        trunk_radius = float(np.percentile(dists_t, 95))
        trunk_radius = max(trunk_radius, 0.05)   # at least 5 cm
        trunk_radius = min(trunk_radius, crown_radius_xy * 0.25)  # sanity cap
    else:
        trunk_radius = max(crown_radius_xy * 0.08, 0.05)

    return {
        "h_tree": float(h_tree),
        "h_trunk": float(h_trunk),
        "h_crown": float(h_crown),
        "crown_radius_xy": float(crown_radius_xy),
        "trunk_radius": float(trunk_radius),
        "z_min": float(z_min),
    }


def _build_mesh(dims: Dict[str, float], max_faces: int) -> trimesh.Trimesh:
    """
    Build the parametric cylinder trunk + ellipsoid crown mesh.

    The bottom of the trunk sits at z_min; the crown ellipsoid is centred
    at trunk_top + crown_radius_z.
    """
    z_min = dims["z_min"]
    h_trunk = max(dims["h_trunk"], 0.1)
    h_crown = max(dims["h_crown"], 0.5)
    crown_rx = max(dims["crown_radius_xy"], 0.2)
    crown_rz = h_crown / 2.0
    trunk_r = dims["trunk_radius"]

    # ---- Trunk cylinder ----
    # sections controls the polygon count; keep it low for a light mesh
    trunk_sections = min(12, max(6, max_faces // 10))
    trunk = trimesh.creation.cylinder(
        radius=trunk_r,
        height=h_trunk,
        sections=trunk_sections
    )
    # Bottom of trunk at z_min
    trunk.apply_translation([0.0, 0.0, z_min + h_trunk / 2.0])

    # ---- Crown ellipsoid ----
    # subdivisions=1 → 80 faces, subdivisions=2 → 320 faces
    crown_subdivisions = 2 if max_faces >= 200 else 1
    crown = trimesh.creation.icosphere(subdivisions=crown_subdivisions, radius=1.0)
    crown.apply_scale([crown_rx, crown_rx, crown_rz])
    crown_center_z = z_min + h_trunk + crown_rz
    crown.apply_translation([0.0, 0.0, crown_center_z])

    # ---- Colours ----
    trunk.visual.vertex_colors = np.full(
        (len(trunk.vertices), 4), [101, 67, 33, 255], dtype=np.uint8
    )
    crown.visual.vertex_colors = np.full(
        (len(crown.vertices), 4), [34, 139, 34, 255], dtype=np.uint8
    )

    tree = trimesh.util.concatenate([trunk, crown])

    # Decimate only if we're still above budget
    if len(tree.faces) > max_faces:
        try:
            tree = tree.simplify_quadratic_decimation(max_faces)
        except Exception:
            pass  # keep the original if decimation fails

    return tree


def generate_tree_shape(
    points: np.ndarray,
    max_faces: int = 200
) -> Dict[str, Any]:
    """
    Generate a parametric 3D mesh for a tree from its point cloud.

    Produces a cylinder trunk + ellipsoid (icosphere) crown, with dimensions
    derived from the point cloud, following the same approach used in
    LiDAR-3D-Urban-Forest-Mapping/convert_trees_to_gltf.py.

    Args:
        points:    (N, 3) float array in local/centred coordinates.
        max_faces: Soft upper bound on output triangle count.

    Returns:
        {"vertices": [[x,y,z], …], "faces": [[i,j,k], …]}
        or {"vertices": [], "faces": []} on failure.
    """
    if len(points) < 10:
        logger.warning("Not enough points to generate tree shape.")
        return {"vertices": [], "faces": []}

    try:
        dims = _estimate_dimensions(points)
        logger.debug(
            f"Tree dims — h_tree={dims['h_tree']:.1f}m  "
            f"h_trunk={dims['h_trunk']:.1f}m  h_crown={dims['h_crown']:.1f}m  "
            f"crown_r={dims['crown_radius_xy']:.2f}m  trunk_r={dims['trunk_radius']:.2f}m"
        )

        mesh = _build_mesh(dims, max_faces)

        return {
            "vertices": [[float(v[0]), float(v[1]), float(v[2])]
                         for v in mesh.vertices.tolist()],
            "faces": [[int(f[0]), int(f[1]), int(f[2])]
                      for f in mesh.faces.tolist()],
        }

    except Exception as e:
        logger.error(f"Error generating tree shape: {e}")
        return {"vertices": [], "faces": []}
