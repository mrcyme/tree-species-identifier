"""
Script to predict roots for existing trees in the database.

Run this script to add root predictions to trees that were processed before
the root prediction feature was added.

Usage:
    cd backend
    python scripts/predict_roots_for_existing_trees.py

This script can be deleted after running once.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from services.root_prediction import predict_roots
from database import async_session_maker
from database.repository import TreeRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def predict_roots_for_existing_trees(batch_size: int = 100):
    """
    Predict roots for all trees that don't have root predictions yet.
    """
    total_updated = 0
    
    while True:
        async with async_session_maker() as session:
            repo = TreeRepository(session)
            
            # Get trees without roots
            trees = await repo.get_trees_without_roots(limit=batch_size)
            
            if not trees:
                logger.info("No more trees without root predictions.")
                break
            
            logger.info(f"Processing batch of {len(trees)} trees...")
            
            for tree in trees:
                try:
                    # Use tree height and crown diameter
                    height = tree.height or 10.0  # Default if not available
                    crown_diameter = tree.crown_diameter or (height * 0.5)  # Estimate
                    
                    # Predict roots
                    root_result = predict_roots(
                        tree_height=height,
                        crown_diameter=crown_diameter,
                        root_type=None,  # Random selection
                        seed=tree.id  # Use DB ID for reproducibility
                    )
                    
                    # Update database
                    await repo.update_root_prediction(
                        db_id=tree.id,
                        root_type=root_result["root_type"],
                        root_lod0=root_result["lod0"],
                        root_lod1=root_result["lod1"],
                        root_lod2=root_result["lod2"],
                        root_lod3=root_result["lod3"],
                        root_seed=root_result["metadata"].get("seed")
                    )
                    
                    logger.debug(f"  Tree {tree.id}: {root_result['root_type']} roots generated")
                    total_updated += 1
                    
                except Exception as e:
                    logger.error(f"  Failed to predict roots for tree {tree.id}: {e}")
            
            # Commit batch
            await repo.commit()
            logger.info(f"  Batch committed. Total updated: {total_updated}")
    
    logger.info(f"✓ Root prediction complete! Updated {total_updated} trees.")
    return total_updated


async def main():
    """Main entry point"""
    print("=" * 60)
    print("Root Prediction for Existing Trees")
    print("=" * 60)
    print()
    print("This script will add root predictions to all trees in the")
    print("database that don't have roots yet.")
    print()
    
    try:
        total = await predict_roots_for_existing_trees()
        print(f"\n✓ Successfully added root predictions to {total} trees!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())






