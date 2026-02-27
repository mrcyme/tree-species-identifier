"""
Predict shape meshes for trees stored in the database.

For each tree with a valid `point_cloud_path`, this script reads the
individual tree LAS, runs shape prediction, and stores `shape_mesh` in DB.

Usage:
    cd backend
    python scripts/predict_shape.py
    python scripts/predict_shape.py --batch-size 200 --max-faces 150
    python scripts/predict_shape.py --recompute
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import laspy
import numpy as np
from sqlalchemy import select

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import async_session_maker, init_db, close_db
from database.models import Tree
from database.repository import TreeRepository
from services.shape_prediction import generate_tree_shape


logger = logging.getLogger("predict-shape")


def _resolve_tree_las_path(path_str: str, backend_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (backend_root / p).resolve()


async def predict_shapes(
    batch_size: int,
    max_faces: int,
    min_points: int,
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
                .limit(batch_size)
            )
            if not recompute:
                query = query.where(Tree.shape_mesh.is_(None))

            trees = list((await session.execute(query)).scalars().all())
            if not trees:
                break

            repo = TreeRepository(session)
            batch_seen = 0
            batch_updated = 0
            batch_skipped = 0
            batch_failed = 0

            for tree in trees:
                if limit is not None and seen >= limit:
                    break

                last_id = int(tree.id)
                seen += 1
                batch_seen += 1

                tree_las_path = _resolve_tree_las_path(tree.point_cloud_path, backend_root)
                if not tree_las_path.exists():
                    logger.warning("Tree %s skipped: LAS file missing (%s)", tree.id, tree_las_path)
                    skipped += 1
                    batch_skipped += 1
                    continue

                try:
                    las = laspy.read(str(tree_las_path))
                    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
                    if len(x) < min_points:
                        logger.warning(
                            "Tree %s skipped: not enough points (%d < %d)",
                            tree.id,
                            len(x),
                            min_points,
                        )
                        skipped += 1
                        batch_skipped += 1
                        continue

                    local_points = np.column_stack([x - x.mean(), y - y.mean(), z - z.mean()])
                    shape_mesh = generate_tree_shape(local_points, max_faces=max_faces)

                    vertices = shape_mesh.get("vertices", [])
                    faces = shape_mesh.get("faces", [])
                    if not vertices or not faces:
                        logger.warning("Tree %s skipped: empty shape mesh generated", tree.id)
                        skipped += 1
                        batch_skipped += 1
                        continue

                    await repo.update_shape_prediction(db_id=int(tree.id), shape_mesh=shape_mesh)
                    updated += 1
                    batch_updated += 1
                except Exception as exc:
                    logger.exception("Tree %s failed during shape prediction: %s", tree.id, exc)
                    failed += 1
                    batch_failed += 1

            await repo.commit()
            logger.info(
                "Batch done: seen=%d updated=%d skipped=%d failed=%d (totals: seen=%d updated=%d skipped=%d failed=%d)",
                batch_seen,
                batch_updated,
                batch_skipped,
                batch_failed,
                seen,
                updated,
                skipped,
                failed,
            )

    return {"seen": seen, "updated": updated, "skipped": skipped, "failed": failed}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict and store tree shape meshes for trees in the database."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of trees processed per DB batch.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=100,
        help="Maximum face budget for each generated tree mesh.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="Minimum number of points required to run shape prediction.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute shape for trees that already have shape_mesh.",
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
        stats = await predict_shapes(
            batch_size=args.batch_size,
            max_faces=args.max_faces,
            min_points=args.min_points,
            recompute=args.recompute,
            limit=args.limit,
        )
        logger.info(
            "Shape prediction complete: seen=%d updated=%d skipped=%d failed=%d",
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
