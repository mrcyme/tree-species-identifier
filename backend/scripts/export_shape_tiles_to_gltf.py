"""
Export tree shape meshes from DB as direct 3D Tiles (1.1) with glTF content.

What this exporter does:
- Reads trees with `shape_mesh` from the database.
- Batches trees into XY tiles (meters) using WebMercator for indexing.
- For each tile, creates one combined GLB mesh in local ENU coordinates.
- Writes a valid 3D Tiles `tileset.json` that references each GLB directly.
- Also writes debug metadata (`tiles_index.json` and per-tile JSON files).
"""
import asyncio
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import trimesh
from pyproj import Transformer
from sqlalchemy import select

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import async_session_maker  # noqa: E402
from database.models import Tree  # noqa: E402


WGS84_TO_WEBMERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
WGS84_TO_ECEF = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
ECEF_TO_WGS84 = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
TRUNK_RGBA = np.array([101, 67, 33, 255], dtype=np.uint8)
CANOPY_RGBA = np.array([34, 139, 34, 255], dtype=np.uint8)


@dataclass
class TileOrigin:
    lon: float
    lat: float
    h: float
    ecef: np.ndarray


def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    rem = len(data) % 4
    if rem == 0:
        return data
    return data + pad_byte * (4 - rem)


def _read_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"Not a GLB file: {path}")

    _, _, total_length = struct.unpack_from("<4sII", raw, 0)
    if total_length != len(raw):
        raise ValueError(f"Invalid GLB length for {path}")

    offset = 12
    json_obj: Dict[str, Any] | None = None
    bin_chunk = b""

    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_len]
        offset += chunk_len

        if chunk_type == 0x4E4F534A:  # JSON
            json_obj = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:  # BIN
            bin_chunk = bytes(chunk)

    if json_obj is None:
        raise ValueError(f"GLB JSON chunk missing in {path}")

    return json_obj, bin_chunk


def _write_glb(path: Path, gltf_json: Dict[str, Any], bin_chunk: bytes) -> None:
    json_bytes = json.dumps(gltf_json, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_bytes = _pad4(json_bytes, pad_byte=b" ")
    bin_chunk = _pad4(bin_chunk, pad_byte=b"\x00")

    chunks = []
    # JSON chunk
    chunks.append(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    chunks.append(json_bytes)
    # BIN chunk
    if len(bin_chunk) > 0:
        chunks.append(struct.pack("<II", len(bin_chunk), 0x004E4942))
        chunks.append(bin_chunk)

    body = b"".join(chunks)
    header = struct.pack("<4sII", b"glTF", 2, 12 + len(body))
    path.write_bytes(header + body)


def _append_aligned(bin_chunk: bytes, payload: bytes, alignment: int = 4) -> Tuple[bytes, int]:
    pad = (-len(bin_chunk)) % alignment
    if pad:
        bin_chunk += b"\x00" * pad
    offset = len(bin_chunk)
    bin_chunk += payload
    return bin_chunk, offset


def inject_feature_metadata_into_glb(glb_path: Path, tree_meta: Dict[str, Any]) -> None:
    """
    Inject EXT_mesh_features + EXT_structural_metadata into a single-tree GLB.
    This makes picked features expose per-tree metadata in viewers that rely on
    glTF feature IDs and structural metadata.
    """
    gltf, bin_chunk = _read_glb(glb_path)

    meshes = gltf.get("meshes") or []
    if not meshes:
        return

    accessors = gltf.setdefault("accessors", [])
    buffer_views = gltf.setdefault("bufferViews", [])
    buffers = gltf.setdefault("buffers", [{"byteLength": len(bin_chunk)}])
    if not buffers:
        buffers.append({"byteLength": len(bin_chunk)})

    # Class schema from metadata types
    class_properties: Dict[str, Dict[str, Any]] = {}
    property_table_properties: Dict[str, Dict[str, Any]] = {}

    def append_scalar_buffer(value: Any, key: str) -> None:
        nonlocal bin_chunk
        if isinstance(value, (int, np.integer)):
            payload = struct.pack("<i", int(value))
            component_type = "INT32"
            alignment = 4
        elif isinstance(value, (float, np.floating)):
            payload = struct.pack("<d", float(value))
            component_type = "FLOAT64"
            alignment = 8
        else:
            text = str(value)
            data = text.encode("utf-8")
            offsets = struct.pack("<II", 0, len(data))
            bin_chunk, values_off = _append_aligned(bin_chunk, data, alignment=1)
            values_view = len(buffer_views)
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": values_off,
                    "byteLength": len(data),
                }
            )
            bin_chunk, offs_off = _append_aligned(bin_chunk, offsets, alignment=4)
            offs_view = len(buffer_views)
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": offs_off,
                    "byteLength": len(offsets),
                }
            )
            class_properties[key] = {"type": "STRING"}
            property_table_properties[key] = {
                "values": values_view,
                "stringOffsets": offs_view,
                "stringOffsetType": "UINT32",
            }
            return

        bin_chunk, value_off = _append_aligned(bin_chunk, payload, alignment=alignment)
        value_view = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": value_off,
                "byteLength": len(payload),
            }
        )
        class_properties[key] = {"type": "SCALAR", "componentType": component_type}
        property_table_properties[key] = {"values": value_view}

    for k, v in tree_meta.items():
        append_scalar_buffer(v, k)

    # Add one feature-id attribute per primitive: all vertices -> feature 0
    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes") or {}
            position_accessor_idx = attrs.get("POSITION")
            if position_accessor_idx is None:
                continue

            vertex_count = int(accessors[position_accessor_idx]["count"])
            feature_ids = np.zeros(vertex_count, dtype=np.uint16).tobytes()
            bin_chunk, fid_off = _append_aligned(bin_chunk, feature_ids, alignment=2)

            fid_view_idx = len(buffer_views)
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": fid_off,
                    "byteLength": len(feature_ids),
                    "target": 34962,  # ARRAY_BUFFER
                }
            )

            fid_accessor_idx = len(accessors)
            accessors.append(
                {
                    "bufferView": fid_view_idx,
                    "byteOffset": 0,
                    "componentType": 5123,  # UNSIGNED_SHORT
                    "count": vertex_count,
                    "type": "SCALAR",
                    "min": [0],
                    "max": [0],
                }
            )

            attrs["_FEATURE_ID_0"] = fid_accessor_idx
            primitive["attributes"] = attrs

            prim_ext = primitive.setdefault("extensions", {})
            prim_ext["EXT_mesh_features"] = {
                "featureIds": [
                    {
                        "featureCount": 1,
                        "attribute": 0,
                        "propertyTable": 0,
                    }
                ]
            }

    top_ext = gltf.setdefault("extensions", {})
    top_ext["EXT_structural_metadata"] = {
        "schema": {
            "id": "tree_schema",
            "classes": {
                "tree": {
                    "properties": class_properties,
                }
            },
        },
        "propertyTables": [
            {
                "name": "tree_properties",
                "class": "tree",
                "count": 1,
                "properties": property_table_properties,
            }
        ],
    }

    ext_used = gltf.setdefault("extensionsUsed", [])
    for name in ("EXT_mesh_features", "EXT_structural_metadata"):
        if name not in ext_used:
            ext_used.append(name)

    buffers[0]["byteLength"] = len(bin_chunk)
    _write_glb(glb_path, gltf, bin_chunk)


def geodetic_to_ecef(lon: float, lat: float, h: float) -> np.ndarray:
    x, y, z = WGS84_TO_ECEF.transform(lon, lat, h)
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu(ecef: np.ndarray, origin: TileOrigin) -> np.ndarray:
    """
    Convert ECEF coordinates to ENU coordinates relative to `origin`.
    """
    dx, dy, dz = ecef - origin.ecef

    lat0 = math.radians(origin.lat)
    lon0 = math.radians(origin.lon)

    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)

    # ECEF -> ENU rotation matrix
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return np.array([east, north, up], dtype=np.float64)


def enu_to_ecef(enu: np.ndarray, origin: TileOrigin) -> np.ndarray:
    """
    Convert ENU coordinates to ECEF coordinates relative to `origin`.
    """
    e, n, u = float(enu[0]), float(enu[1]), float(enu[2])
    lat0 = math.radians(origin.lat)
    lon0 = math.radians(origin.lon)
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u
    return origin.ecef + np.array([dx, dy, dz], dtype=np.float64)


def enu_to_ecef_transform(origin: TileOrigin) -> List[float]:
    """
    4x4 column-major transform from local ENU coordinates to ECEF.
    """
    lat0 = math.radians(origin.lat)
    lon0 = math.radians(origin.lon)
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)

    # ENU basis vectors expressed in ECEF
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array(
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64
    )
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    tx, ty, tz = origin.ecef.tolist()

    return [
        float(east[0]),
        float(east[1]),
        float(east[2]),
        0.0,
        float(north[0]),
        float(north[1]),
        float(north[2]),
        0.0,
        float(up[0]),
        float(up[1]),
        float(up[2]),
        0.0,
        float(tx),
        float(ty),
        float(tz),
        1.0,
    ]


def region_from_enu_bounds(bounds_enu: np.ndarray, origin: TileOrigin) -> List[float]:
    """
    Build a 3D Tiles region [west, south, east, north, minH, maxH]
    from ENU bounds and origin.
    """
    mins = bounds_enu[0]
    maxs = bounds_enu[1]
    corners_enu = [
        np.array([x, y, z], dtype=np.float64)
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]

    lon_vals: List[float] = []
    lat_vals: List[float] = []
    h_vals: List[float] = []
    for c in corners_enu:
        ecef = enu_to_ecef(c, origin)
        lon, lat, h = ECEF_TO_WGS84.transform(float(ecef[0]), float(ecef[1]), float(ecef[2]))
        lon_vals.append(float(lon))
        lat_vals.append(float(lat))
        h_vals.append(float(h))

    west = math.radians(min(lon_vals))
    east = math.radians(max(lon_vals))
    south = math.radians(min(lat_vals))
    north = math.radians(max(lat_vals))
    min_h = min(h_vals)
    max_h = max(h_vals)
    return [west, south, east, north, min_h, max_h]


def tree_altitude_from_db(tree: Tree) -> float:
    """
    Use tree altitude stored in DB as-is (no exporter-side conversion).
    The DB pipeline is responsible for writing ellipsoidal heights.
    """
    if tree.z_min is not None and tree.z_max is not None:
        return float((tree.z_min + tree.z_max) / 2.0)
    if tree.z_min is not None:
        return float(tree.z_min)
    if tree.z_max is not None:
        return float(tree.z_max)
    return 0.0


def tile_key(lon: float, lat: float, tile_size_m: float) -> Tuple[int, int]:
    mx, my = WGS84_TO_WEBMERC.transform(lon, lat)
    return (math.floor(mx / tile_size_m), math.floor(my / tile_size_m))


def mesh_from_shape(shape_mesh: dict) -> trimesh.Trimesh | None:
    vertices = shape_mesh.get("vertices")
    faces = shape_mesh.get("faces")
    if not vertices or not faces:
        return None
    try:
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
        mesh = color_tree_mesh(mesh)
        return mesh
    except Exception:
        return None


def color_tree_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Assign explicit tree colors so GLBs remain visible in Cesium.
    Bottom 25% of mesh height is trunk (brown), top is canopy (green).
    """
    if len(mesh.vertices) == 0:
        return mesh

    z = mesh.vertices[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    z_split = z_min + (z_max - z_min) * 0.25

    colors = np.tile(CANOPY_RGBA, (len(mesh.vertices), 1))
    trunk_mask = z <= z_split
    colors[trunk_mask] = TRUNK_RGBA
    mesh.visual.vertex_colors = colors
    return mesh


def z_up_to_y_up_transform() -> np.ndarray:
    """
    Return 4x4 transform to store Z-up ENU geometry in glTF's Y-up frame.
    Cesium applies a Y-up -> Z-up transform at runtime, so this pre-rotation
    keeps world orientation correct.
    """
    # Rotation around X by -90 degrees:
    # x' = x
    # y' = z
    # z' = -y
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


async def load_trees_with_shapes() -> List[Tree]:
    async with async_session_maker() as session:
        query = (
            select(Tree)
            .where(Tree.shape_mesh.isnot(None))
            .where(Tree.longitude.isnot(None))
            .where(Tree.latitude.isnot(None))
        )
        result = await session.execute(query)
        return list(result.scalars().all())


def export_tile(
    trees: List[Tree],
    out_dir: Path,
    key: Tuple[int, int],
    max_trees_per_tile: int = 5000,
) -> Dict[str, Any] | None:
    """
    Export one logical tile as multiple per-tree GLBs + tree metadata entries.
    """
    if not trees:
        return None

    if len(trees) > max_trees_per_tile:
        trees = trees[:max_trees_per_tile]

    tree_entries: List[Dict[str, Any]] = []
    tree_records: List[Dict[str, Any]] = []
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for t in trees:
        if not t.shape_mesh:
            continue

        mesh = mesh_from_shape(t.shape_mesh)
        if mesh is None:
            continue

        lon = float(t.longitude)
        lat = float(t.latitude)
        alt = tree_altitude_from_db(t)
        origin = TileOrigin(lon=lon, lat=lat, h=alt, ecef=geodetic_to_ecef(lon, lat, alt))

        # Keep mesh in local ENU, but store it in glTF Y-up so Cesium's
        # runtime Y-up -> Z-up conversion restores correct world orientation.
        glb_mesh = mesh.copy()
        glb_mesh.apply_transform(z_up_to_y_up_transform())

        unique_tree_id = t.id if t.id is not None else t.tree_id
        glb_name = f"tiles/tree_{unique_tree_id}_{key[0]}_{key[1]}.glb"
        glb_path = out_dir / glb_name
        glb_mesh.export(str(glb_path), file_type="glb")

        bounds = mesh.bounds
        region = region_from_enu_bounds(bounds, origin)
        transform = enu_to_ecef_transform(origin)
        db_id = int(t.id) if t.id is not None else int(t.tree_id)
        tree_id = int(t.tree_id) if t.tree_id is not None else db_id
        species_name = t.species_name or "Unknown"
        species_confidence = (
            float(t.species_confidence) if t.species_confidence is not None else -1.0
        )
        height = float(t.height) if t.height is not None else 0.0
        crown_diameter = float(t.crown_diameter) if t.crown_diameter is not None else 0.0

        tree_entries.append(
            {
                "file": glb_name,
                "region": region,
                "transform": transform,
                "metadata": {
                    "db_id": db_id,
                    "tree_id": tree_id,
                    "species_name": species_name,
                    "species_confidence": species_confidence,
                    "height": height,
                    "crown_diameter": crown_diameter,
                    "longitude": lon,
                    "latitude": lat,
                    "altitude": alt,
                },
            }
        )

        # Make each single-tree GLB pickable with per-feature metadata.
        inject_feature_metadata_into_glb(glb_path, tree_entries[-1]["metadata"])

        tree_records.append(
            {
                "db_id": db_id,
                "tree_id": tree_id,
                "longitude": lon,
                "latitude": lat,
                "altitude": alt,
                "species_name": species_name,
                "species_confidence": species_confidence,
                "height": height,
                "crown_diameter": crown_diameter,
                "n_vertices": len(mesh.vertices),
                "n_faces": len(mesh.faces),
            }
        )

    if not tree_entries:
        return None

    west = min(e["region"][0] for e in tree_entries)
    south = min(e["region"][1] for e in tree_entries)
    east = max(e["region"][2] for e in tree_entries)
    north = max(e["region"][3] for e in tree_entries)
    min_h = min(e["region"][4] for e in tree_entries)
    max_h = max(e["region"][5] for e in tree_entries)

    tile_meta = {
        "tile_id": f"{key[0]}_{key[1]}",
        "region": [west, south, east, north, min_h, max_h],
        "tree_count": len(tree_records),
        "trees_as_tiles": tree_entries,
        "trees": tree_records,
    }

    with open(out_dir / f"tile_{key[0]}_{key[1]}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(tile_meta, f, indent=2)

    return tile_meta


async def export_tiles(
    output_dir: Path,
    tile_size_m: float = 200.0,
) -> dict:
    trees = await load_trees_with_shapes()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trees:
        return {"tiles": [], "summary": {"trees_with_shapes": 0, "tile_count": 0}}

    # Group by tile key
    grouped: Dict[Tuple[int, int], List[Tree]] = {}
    for t in trees:
        k = tile_key(float(t.longitude), float(t.latitude), tile_size_m)
        grouped.setdefault(k, []).append(t)

    tile_summaries: List[dict] = []
    for k, items in grouped.items():
        info = export_tile(
            items,
            output_dir,
            k,
        )
        if info:
            tile_summaries.append(info)

    tree_entries: List[Dict[str, Any]] = []
    for t in tile_summaries:
        tree_entries.extend(t["trees_as_tiles"])

    # Global export index
    export_index = {
        "format": "3d-tiles-1.1-gltf",
        "crs": {"position": "EPSG:4326", "tile_indexing": "EPSG:3857"},
        "tile_size_m": tile_size_m,
        "vertical_mode": "db_direct",
        "summary": {
            "trees_with_shapes": len(trees),
            "tile_count": len(tile_summaries),
            "tree_content_count": len(tree_entries),
        },
        "tiles": [
            {
                "tile_id": t["tile_id"],
                "tree_count": t["tree_count"],
                "region": t["region"],
            }
            for t in tile_summaries
        ],
    }

    with open(output_dir / "tiles_index.json", "w", encoding="utf-8") as f:
        json.dump(export_index, f, indent=2)

    # Generate direct 3D Tiles tileset (explicit tileset with glTF contents).
    if tree_entries:
        west = min(e["region"][0] for e in tree_entries)
        south = min(e["region"][1] for e in tree_entries)
        east = max(e["region"][2] for e in tree_entries)
        north = max(e["region"][3] for e in tree_entries)
        min_h = min(e["region"][4] for e in tree_entries)
        max_h = max(e["region"][5] for e in tree_entries)

        children = []
        for e in tree_entries:
            # Put per-tree metadata on the content object so pick/inspect flows
            # can resolve metadata from the clicked GLB content directly.
            content_with_metadata = {
                "uri": e["file"],
                "metadata": {
                    "class": "tree",
                    "properties": e["metadata"],
                },
            }
            children.append(
                {
                    "boundingVolume": {"region": e["region"]},
                    "geometricError": 0.0,
                    "transform": e["transform"],
                    "content": content_with_metadata,
                }
            )

        root_geometric_error = max(tile_size_m, 1.0)
        tileset = {
            "asset": {"version": "1.1"},
            "schema": {
                "classes": {
                    "tree": {
                        "properties": {
                            "db_id": {"type": "SCALAR", "componentType": "INT32"},
                            "tree_id": {"type": "SCALAR", "componentType": "INT32"},
                            "species_name": {"type": "STRING"},
                            "species_confidence": {
                                "type": "SCALAR",
                                "componentType": "FLOAT64",
                            },
                            "height": {"type": "SCALAR", "componentType": "FLOAT64"},
                            "crown_diameter": {"type": "SCALAR", "componentType": "FLOAT64"},
                            "longitude": {"type": "SCALAR", "componentType": "FLOAT64"},
                            "latitude": {"type": "SCALAR", "componentType": "FLOAT64"},
                            "altitude": {"type": "SCALAR", "componentType": "FLOAT64"},
                        }
                    }
                }
            },
            "geometricError": root_geometric_error,
            "root": {
                "boundingVolume": {"region": [west, south, east, north, min_h, max_h]},
                # Must be > 0 so runtime refines root and requests child content.
                "geometricError": root_geometric_error,
                "refine": "REPLACE",
                "children": children,
            },
        }

        with open(output_dir / "tileset.json", "w", encoding="utf-8") as f:
            json.dump(tileset, f, indent=2)

    return export_index


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export DB shape meshes into direct 3D Tiles (1.1) with glTF content."
    )
    parser.add_argument(
        "--output",
        default="../output/shape_gltf_tiles",
        help="Output directory for GLB tiles and metadata index.",
    )
    parser.add_argument(
        "--tile-size-m",
        type=float,
        default=200.0,
        help="Tile size in meters (WebMercator indexing).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir / args.output).resolve()

    result = await export_tiles(
        output_dir=out_dir,
        tile_size_m=args.tile_size_m,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Export written to: {out_dir}")
    print("3D Tiles entrypoint: tileset.json")
    print("Vertical mode: db_direct (no exporter-side altitude conversion)")


if __name__ == "__main__":
    asyncio.run(main())
