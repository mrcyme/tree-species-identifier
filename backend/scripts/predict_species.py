"""
Predict species for trees in the database that don't have a species yet.

Groups individual tree LAS files, merges them into a temporary combined LAS
with a TreeID column, runs DetailView species prediction, and updates the DB.

Usage:
    cd backend
    python scripts/predict_species.py
    python scripts/predict_species.py --db-batch-size 500 --gpu-batch-size 8
    python scripts/predict_species.py --n-aug 5 --recompute
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import async_session_maker, init_db, close_db
from database.models import Tree
from database.repository import TreeRepository

logger = logging.getLogger("predict-species")

PROJECT_DIR = Path(__file__).parent.parent.parent
DETAILVIEW_DIR = PROJECT_DIR / "DetailView"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "model_202305171452_60"


def _resolve_path(path_str: str, backend_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (backend_root / p).resolve()


def _merge_tree_las_files(
    tree_paths: list[tuple[int, Path]],
) -> Path:
    """
    Merge individual tree LAS files into a single temporary LAS with TreeID.

    Args:
        tree_paths: list of (tree_id_within_file, las_path) tuples

    Returns:
        Path to the temporary merged LAS file.
    """
    all_x, all_y, all_z = [], [], []
    all_tree_ids = []

    for tid, las_path in tree_paths:
        las = laspy.read(str(las_path))
        all_x.append(np.array(las.x))
        all_y.append(np.array(las.y))
        all_z.append(np.array(las.z))
        all_tree_ids.append(np.full(len(las.x), tid, dtype=np.int32))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    tree_ids = np.concatenate(all_tree_ids)

    header = laspy.LasHeader(point_format=0)
    header.offsets = [x.min(), y.min(), z.min()]
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(name="TreeID", type=np.int32))

    out = laspy.LasData(header)
    out.x = x
    out.y = y
    out.z = z
    out.TreeID = tree_ids

    tmp = tempfile.NamedTemporaryFile(suffix=".las", delete=False)
    out.write(tmp.name)
    tmp.close()
    return Path(tmp.name)


def _run_predict_batch(
    merged_las: Path,
    model_path: Path,
    n_aug: int,
    gpu_batch_size: int,
) -> dict[int, dict]:
    """
    Run DetailView prediction on a merged LAS and return results keyed by tree_id.
    """
    sys.path.insert(0, str(DETAILVIEW_DIR))
    import inspect
    import torch
    from predict import run_predict

    _orig_dataloader = torch.utils.data.DataLoader

    class _PatchedDataLoader(_orig_dataloader):
        def __init__(self, *args, **kwargs):
            kwargs["batch_size"] = gpu_batch_size
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = _PatchedDataLoader
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            predict_kwargs = dict(
                prediction_data=str(merged_las),
                path_las="",
                model_path=str(model_path),
                tree_id_col="TreeID",
                n_aug=n_aug,
                output_dir=tmpdir,
                path_csv_lookup=str(DETAILVIEW_DIR / "lookup.csv"),
                projection_backend="numpy",
                output_type="csv",
            )
            if "force_device" in inspect.signature(run_predict).parameters:
                predict_kwargs["force_device"] = "cuda" if torch.cuda.is_available() else "cpu"
            _, _, joined_df, probs_df = run_predict(**predict_kwargs)
    finally:
        torch.utils.data.DataLoader = _orig_dataloader

    results: dict[int, dict] = {}
    for _, row in joined_df.iterrows():
        try:
            tid = int(row["filename"].split("_")[-1])
        except Exception:
            tid = int(row["filename"])

        pred: dict = {
            "species_id": int(row["species_id"]),
            "species_name": row["species"],
            "confidence": float(row["species_prob"]),
        }

        if probs_df is not None:
            tree_row = probs_df[probs_df["File"] == row["filename"]]
            if len(tree_row) > 0:
                pred["probabilities"] = {
                    col: float(tree_row[col].values[0])
                    for col in probs_df.columns[1:]
                }

        results[tid] = pred

    return results


async def predict_species(
    db_batch_size: int,
    gpu_batch_size: int,
    n_aug: int,
    model_path: Path,
    recompute: bool,
    limit: int | None,
) -> dict:
    backend_root = Path(__file__).parent.parent.resolve()
    last_id = 0
    seen = 0
    updated = 0
    skipped = 0
    failed = 0

    while True:
        if limit is not None and seen >= limit:
            break

        async with async_session_maker() as session:
            query = (
                select(Tree)
                .where(Tree.id > last_id)
                .where(Tree.point_cloud_path.is_not(None))
                .order_by(Tree.id.asc())
                .limit(db_batch_size)
            )
            if not recompute:
                query = query.where(Tree.species_name.is_(None))

            trees = list((await session.execute(query)).scalars().all())
            if not trees:
                break

            # Collect valid trees with existing LAS files
            valid_trees: list[tuple[Tree, Path]] = []
            for tree in trees:
                last_id = int(tree.id)
                seen += 1
                if limit is not None and seen > limit:
                    break

                las_path = _resolve_path(tree.point_cloud_path, backend_root)
                if not las_path.exists():
                    logger.warning("Tree %s skipped: LAS missing (%s)", tree.id, las_path)
                    skipped += 1
                    continue
                valid_trees.append((tree, las_path))

            if not valid_trees:
                logger.info("Batch: no valid LAS files, skipping")
                continue

            # Build (tree_id_for_las, las_path) list and keep DB id mapping
            # Use the tree's tree_id (within point cloud) as the TreeID in merged LAS
            # but we need unique IDs, so use db id instead
            tree_paths = [(int(t.id), p) for t, p in valid_trees]
            logger.info("Merging %d tree LAS files for prediction...", len(tree_paths))
            merged_las = None
            try:
                merged_las = _merge_tree_las_files(tree_paths)
                logger.info("Running species prediction on %d trees...", len(tree_paths))
                results = _run_predict_batch(merged_las, model_path, n_aug, gpu_batch_size)
            except Exception as exc:
                logger.exception("Batch prediction failed: %s", exc)
                failed += len(tree_paths)
                continue
            finally:
                if merged_las and merged_las.exists():
                    merged_las.unlink()

            # Update DB
            repo = TreeRepository(session)
            for db_id, pred in results.items():
                try:
                    await repo.update_species_by_db_id(
                        db_id=db_id,
                        species_id=pred["species_id"],
                        species_name=pred["species_name"],
                        confidence=pred["confidence"],
                        probabilities=pred.get("probabilities"),
                    )
                    updated += 1
                except Exception as exc:
                    logger.exception("Failed to update tree %s: %s", db_id, exc)
                    failed += 1

            await repo.commit()
            logger.info(
                "Batch done: predicted=%d updated=%d (totals: seen=%d updated=%d skipped=%d failed=%d)",
                len(results),
                updated,
                seen,
                updated,
                skipped,
                failed,
            )

    return {"seen": seen, "updated": updated, "skipped": skipped, "failed": failed}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict species for trees in the database without a species prediction."
    )
    parser.add_argument(
        "--db-batch-size",
        type=int,
        default=200,
        help="Number of trees fetched from DB per batch (default: 200).",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=8,
        help="Batch size for GPU inference. 8 is safe for 8GB VRAM (default: 8).",
    )
    parser.add_argument(
        "--n-aug",
        type=int,
        default=3,
        help="Number of augmentation passes (default: 3).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model weights.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute species for trees that already have a prediction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N trees (useful for testing).",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> None:
    await init_db()
    try:
        stats = await predict_species(
            db_batch_size=args.db_batch_size,
            gpu_batch_size=args.gpu_batch_size,
            n_aug=args.n_aug,
            model_path=Path(args.model_path),
            recompute=args.recompute,
            limit=args.limit,
        )
        logger.info(
            "Species prediction complete: seen=%d updated=%d skipped=%d failed=%d",
            stats["seen"],
            stats["updated"],
            stats["skipped"],
            stats["failed"],
        )
    finally:
        await close_db()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = build_arg_parser().parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
