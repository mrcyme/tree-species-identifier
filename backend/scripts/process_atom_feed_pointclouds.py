"""
Ingest Brussels Atom feed point-cloud tiles and run tree segmentation pipeline:

1) Parse ZIP URLs from Atom feed (`link rel="section"`).
2) Download each ZIP and extract LAS/LAZ file(s).
3) Run segmentation for each LAS/LAZ.
4) Save segmented trees into DB.
5) Delete extracted full LAS/LAZ file after successful processing.

Example:
    cd backend
    python scripts/process_atom_feed_pointclouds.py \
      --feed-url "https://urbisdownload.datastore.brussels/atomfeed/ff1124e1-424e-11ee-b156-00090ffe0001-en.xml" \
      --tile-size-m 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import laspy
from sqlalchemy import select, func

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import async_session_maker, init_db, close_db
from database.models import PointCloud, Tree
from database.repository import TreeRepository
from services.segmentation import run_segmentation, get_point_cloud_info


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_FEED_URL = (
    "https://urbisdownload.datastore.brussels/atomfeed/"
    "ff1124e1-424e-11ee-b156-00090ffe0001-en.xml"
)

logger = logging.getLogger("atom-feed-pipeline")


@dataclass
class FeedItem:
    url: str
    bbox: str | None
    time: str | None

    @property
    def zip_name(self) -> str:
        return Path(urlparse(self.url).path).name


@dataclass
class SegmentationChunk:
    segmented_las: Path
    trees: List[dict]
    individual_trees_dir: Path
    label: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_progress(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.exists():
        return {"version": 1, "updated_at": _utc_now_iso(), "entries": {}}
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("progress file root is not an object")
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        data["entries"] = entries
        data.setdefault("version", 1)
        data.setdefault("updated_at", _utc_now_iso())
        return data
    except Exception as exc:
        backup = progress_path.with_suffix(progress_path.suffix + ".corrupt")
        progress_path.replace(backup)
        logger.warning(
            "Progress file was invalid and moved to %s (%s). Starting fresh.",
            backup,
            exc,
        )
        return {"version": 1, "updated_at": _utc_now_iso(), "entries": {}}


def _save_progress(progress_path: Path, progress: Dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = _utc_now_iso()
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(progress_path)


def _progress_key(source_filename: str, las_path: Path) -> str:
    return f"{source_filename}::{las_path.name}"


def parse_atom_feed(feed_url: str) -> List[FeedItem]:
    logger.info("Fetching Atom feed: %s", feed_url)
    with urlopen(feed_url, timeout=120) as resp:
        data = resp.read()

    root = ET.fromstring(data)
    links = root.findall(".//atom:entry/atom:link[@rel='section']", ATOM_NS)

    items: List[FeedItem] = []
    seen = set()
    for link in links:
        href = link.attrib.get("href")
        if not href or not href.lower().endswith(".zip"):
            continue
        if href in seen:
            continue
        seen.add(href)
        items.append(
            FeedItem(
                url=href,
                bbox=link.attrib.get("bbox"),
                time=link.attrib.get("time"),
            )
        )

    logger.info("Found %d ZIP links in feed", len(items))
    return items


def download_zip(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        logger.info("Using existing ZIP: %s", target_path.name)
        return target_path

    logger.info("Downloading: %s", url)
    with urlopen(url, timeout=300) as resp:
        content = resp.read()
    target_path.write_bytes(content)
    logger.info("Saved ZIP: %s (%.1f MB)", target_path.name, target_path.stat().st_size / (1024 * 1024))
    return target_path


def extract_las_files(zip_path: Path, extract_dir: Path) -> List[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    las_files: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            lower = member.lower()
            if lower.endswith(".las") or lower.endswith(".laz"):
                out_path = extract_dir / Path(member).name
                if not out_path.exists():
                    with zf.open(member) as src, out_path.open("wb") as dst:
                        dst.write(src.read())
                las_files.append(out_path)
    if not las_files:
        raise RuntimeError(f"No LAS/LAZ file found in ZIP: {zip_path}")
    return las_files


async def point_cloud_exists(filename: str) -> bool:
    async with async_session_maker() as session:
        result = await session.execute(
            PointCloud.__table__.select().where(PointCloud.filename == filename).limit(1)
        )
        return result.first() is not None


async def get_point_cloud_processing_status(filename: str) -> Tuple[bool, int | None, int]:
    """
    Check if a feed tile is already processed.

    Returns:
        (already_processed, point_cloud_id, tree_count)
    """
    async with async_session_maker() as session:
        result = await session.execute(
            select(PointCloud).where(PointCloud.filename == filename).order_by(PointCloud.created_at.desc())
        )
        point_cloud = result.scalars().first()
        if point_cloud is None:
            return False, None, 0

        tree_count_result = await session.execute(
            select(func.count(Tree.id)).where(Tree.point_cloud_id == point_cloud.id)
        )
        linked_tree_count = int(tree_count_result.scalar() or 0)
        total_tree_count = max(int(point_cloud.n_trees or 0), linked_tree_count)
        return total_tree_count > 0, int(point_cloud.id), total_tree_count


def _las_xy_bounds(las_path: Path) -> Tuple[float, float, float, float]:
    with laspy.open(str(las_path)) as fh:
        mins = fh.header.mins
        maxs = fh.header.maxs
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def _build_subset_grid(
    bounds: Tuple[float, float, float, float],
    cell_size_m: float,
    overlap_m: float,
) -> List[Tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = bounds
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    step = max(cell_size_m - overlap_m, cell_size_m * 0.5)
    nx = max(1, math.ceil((max_x - min_x) / step))
    ny = max(1, math.ceil((max_y - min_y) / step))

    subsets: List[Tuple[float, float, float, float]] = []
    for ix in range(nx):
        for iy in range(ny):
            sx0 = min_x + ix * step
            sy0 = min_y + iy * step
            sx1 = min(max_x, sx0 + cell_size_m)
            sy1 = min(max_y, sy0 + cell_size_m)
            subsets.append((sx0, sy0, sx1, sy1))
    return subsets


def _subset_to_arg(subset: Tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.3f}" for v in subset)


def run_segmentation_with_fallback(
    las_path: Path,
    job_dir: Path,
    crs: str,
    enable_subtiling_fallback: bool,
    fallback_cell_size_m: float,
    fallback_overlap_m: float,
) -> List[SegmentationChunk]:
    try:
        seg = run_segmentation(las_path, job_dir, crs=crs)
        return [
            SegmentationChunk(
                segmented_las=Path(seg["segmented_las"]),
                trees=seg["trees"],
                individual_trees_dir=Path(seg["individual_trees_dir"]),
                label="full",
            )
        ]
    except Exception as exc:
        if not enable_subtiling_fallback:
            raise
        logger.warning(
            "Full-tile segmentation failed for %s. Retrying with spatial chunks: %s",
            las_path.name,
            exc,
        )

    bounds = _las_xy_bounds(las_path)
    subsets = _build_subset_grid(bounds, fallback_cell_size_m, fallback_overlap_m)
    logger.info(
        "Sub-tiling fallback for %s: %d chunks (size=%sm overlap=%sm)",
        las_path.name,
        len(subsets),
        fallback_cell_size_m,
        fallback_overlap_m,
    )

    chunks: List[SegmentationChunk] = []
    for idx, subset in enumerate(subsets, start=1):
        subset_label = f"chunk_{idx:03d}"
        subset_dir = job_dir / subset_label
        subset_arg = _subset_to_arg(subset)
        try:
            seg = run_segmentation(las_path, subset_dir, crs=crs, subset=subset_arg)
            if not seg["trees"]:
                continue
            chunks.append(
                SegmentationChunk(
                    segmented_las=Path(seg["segmented_las"]),
                    trees=seg["trees"],
                    individual_trees_dir=Path(seg["individual_trees_dir"]),
                    label=subset_label,
                )
            )
            logger.info("%s succeeded with %d trees", subset_label, len(seg["trees"]))
        except Exception as exc:
            logger.warning("%s failed (%s): %s", subset_label, subset_arg, exc)

    if not chunks:
        raise RuntimeError(
            f"Segmentation failed for {las_path.name} both full-tile and all sub-tiles."
        )
    return chunks


async def process_las_file(
    las_path: Path,
    source_filename: str,
    work_root: Path,
    crs: str,
    enable_subtiling_fallback: bool,
    fallback_cell_size_m: float,
    fallback_overlap_m: float,
) -> dict:
    logger.info("Processing LAS: %s", las_path.name)
    job_dir = work_root / las_path.stem
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1) segmentation (+ fallback chunking if needed)
    seg_chunks = run_segmentation_with_fallback(
        las_path=las_path,
        job_dir=job_dir,
        crs=crs,
        enable_subtiling_fallback=enable_subtiling_fallback,
        fallback_cell_size_m=fallback_cell_size_m,
        fallback_overlap_m=fallback_overlap_m,
    )
    pc_info = get_point_cloud_info(las_path, crs=crs)

    # 2) ensure point cloud exists in DB
    async with async_session_maker() as session:
        repo = TreeRepository(session)
        pc = await repo.get_or_create_point_cloud(
            filename=source_filename,
            crs=f"EPSG:{crs}",
            bbox_min=(
                pc_info.get("bbox_min", [None, None])[0],
                pc_info.get("bbox_min", [None, None])[1],
            ),
            bbox_max=(
                pc_info.get("bbox_max", [None, None])[0],
                pc_info.get("bbox_max", [None, None])[1],
            ),
            n_points=pc_info.get("n_points"),
        )
        # Required when we later open a new session for tree inserts:
        # without commit, the point_cloud row may not be visible and FK checks fail.
        await repo.commit()
        point_cloud_id = pc.id

    total_trees = 0
    for chunk in seg_chunks:
        # 3) save chunk trees in DB
        trees_for_db = [dict(t) for t in chunk.trees]
        for tree_data in trees_for_db:
            tree_data["point_cloud_path"] = str(
                chunk.individual_trees_dir / f"tree_{tree_data['tree_id']}.las"
            )

        async with async_session_maker() as session:
            repo = TreeRepository(session)
            await repo.bulk_upsert_trees(point_cloud_id, trees_for_db)
            await repo.commit()

        total_trees += len(trees_for_db)

    return {
        "source": source_filename,
        "point_cloud_id": point_cloud_id,
        "n_trees": total_trees,
        "n_chunks": len(seg_chunks),
    }


async def run_pipeline(args: argparse.Namespace) -> None:
    await init_db()
    try:
        items = parse_atom_feed(args.feed_url)
        if args.limit is not None:
            items = items[: args.limit]

        downloads_dir = Path(args.downloads_dir).resolve()
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        progress_path = Path(args.progress_file).resolve()
        progress = _load_progress(progress_path)
        entries: Dict[str, Dict[str, Any]] = progress["entries"]
        logger.info("Using progress file: %s", progress_path)

        summaries = []
        for idx, item in enumerate(items, start=1):
            logger.info("[%d/%d] %s", idx, len(items), item.zip_name)
            try:
                already_processed, existing_pc_id, existing_tree_count = await get_point_cloud_processing_status(
                    item.zip_name
                )

                if already_processed and not args.reprocess_existing:
                    logger.info(
                        "DB has existing trees for tile %s (point_cloud_id=%s, trees=%d). "
                        "Will rely on per-LAS progress checkpoint for resume safety.",
                        item.zip_name,
                        existing_pc_id,
                        existing_tree_count,
                    )

                # Legacy behavior (stricter): skip as soon as a point_cloud row exists.
                if args.skip_existing_db and existing_pc_id is not None:
                    logger.info("Skipping already-ingested point cloud: %s", item.zip_name)
                    continue

                zip_path = download_zip(item.url, downloads_dir / item.zip_name)
                extracted = extract_las_files(zip_path, work_dir / "extracted" / zip_path.stem)
                for las_path in extracted:
                    key = _progress_key(item.zip_name, las_path)
                    existing_entry = entries.get(key, {})
                    if (
                        not args.reprocess_existing
                        and isinstance(existing_entry, dict)
                        and existing_entry.get("status") == "completed"
                    ):
                        logger.info("Skipping completed LAS from progress file: %s", key)
                        continue

                    attempts = int(existing_entry.get("attempts", 0)) + 1
                    entries[key] = {
                        **existing_entry,
                        "source": item.zip_name,
                        "las": las_path.name,
                        "status": "in_progress",
                        "attempts": attempts,
                        "last_error": None,
                        "updated_at": _utc_now_iso(),
                    }
                    _save_progress(progress_path, progress)

                    try:
                        summary = await process_las_file(
                            las_path=las_path,
                            source_filename=item.zip_name,
                            work_root=work_dir / "processing",
                            crs=args.crs,
                            enable_subtiling_fallback=args.enable_subtiling_fallback,
                            fallback_cell_size_m=args.fallback_cell_size_m,
                            fallback_overlap_m=args.fallback_overlap_m,
                        )
                        summaries.append(summary)
                        logger.info(
                            "Processed %s: chunks=%d trees=%d",
                            summary["source"],
                            summary.get("n_chunks", 1),
                            summary["n_trees"],
                        )
                        entries[key] = {
                            **entries[key],
                            "status": "completed",
                            "completed_at": _utc_now_iso(),
                            "last_error": None,
                            "last_summary": {
                                "n_chunks": summary.get("n_chunks", 1),
                                "n_trees": summary.get("n_trees", 0),
                            },
                            "updated_at": _utc_now_iso(),
                        }
                        _save_progress(progress_path, progress)
                        try:
                            las_path.unlink()
                            logger.info("Deleted extracted LAS/LAZ after successful processing: %s", las_path)
                        except Exception as cleanup_exc:
                            logger.warning("Could not delete extracted LAS/LAZ %s: %s", las_path, cleanup_exc)
                    except Exception as exc:
                        entries[key] = {
                            **entries[key],
                            "status": "failed",
                            "last_error": str(exc),
                            "updated_at": _utc_now_iso(),
                        }
                        _save_progress(progress_path, progress)
                        logger.exception("Failed processing %s (%s): %s", item.zip_name, las_path.name, exc)
                        if not args.continue_on_error:
                            raise
            except Exception as exc:
                logger.exception("Failed processing %s: %s", item.zip_name, exc)
                if not args.continue_on_error:
                    raise

        logger.info("Processed %d point-cloud files in total", len(summaries))

    finally:
        await close_db()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process Atom feed point-clouds: download, segment, and save trees to DB."
    )
    parser.add_argument("--feed-url", default=DEFAULT_FEED_URL, help="Atom feed URL.")
    parser.add_argument(
        "--downloads-dir",
        default="../output/atom_feed_downloads",
        help="Directory to store downloaded ZIP files.",
    )
    parser.add_argument(
        "--work-dir",
        default="../output/atom_feed_work",
        help="Directory to store extracted/processed intermediates.",
    )
    parser.add_argument("--crs", default="31370", help="EPSG code of LAS input CRS.")
    parser.add_argument(
        "--enable-subtiling-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If full-tile segmentation fails, retry using spatial sub-tiles.",
    )
    parser.add_argument(
        "--fallback-cell-size-m",
        type=float,
        default=700.0,
        help="Sub-tile size (meters) used for segmentation fallback.",
    )
    parser.add_argument(
        "--fallback-overlap-m",
        type=float,
        default=15.0,
        help="Overlap (meters) between fallback sub-tiles.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N feed ZIP entries.")
    parser.add_argument(
        "--skip-existing-db",
        action="store_true",
        help="Legacy: skip feed items when a point_cloud row already exists (even if no trees yet).",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Force reprocessing tiles even when trees already exist in DB.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing next files if one file fails.",
    )
    parser.add_argument(
        "--progress-file",
        default="../output/atom_feed_work/atom_feed_progress.json",
        help="JSON file used to persist per-LAS progress checkpoints for robust resume.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = build_arg_parser().parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
