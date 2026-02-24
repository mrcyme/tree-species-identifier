"""
Migration script to add root prediction columns to the trees table.

Run this script to add the new columns for storing root prediction data.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database.config import engine as async_engine


async def run_migration():
    """Add root prediction columns to trees table"""
    print("Running migration: Adding root prediction columns to trees table...")
    
    async with async_engine.begin() as conn:
        # Check if columns already exist
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trees' AND column_name = 'root_type'
        """)
        result = await conn.execute(check_query)
        if result.fetchone():
            print("  Migration already applied - root columns exist.")
            return
        
        # Add new columns
        print("  Adding root_type column...")
        await conn.execute(text("""
            ALTER TABLE trees ADD COLUMN root_type VARCHAR(50) NULL
        """))
        
        print("  Adding root_lod0 column (full mesh)...")
        await conn.execute(text("""
            ALTER TABLE trees ADD COLUMN root_lod0 JSONB NULL
        """))
        
        print("  Adding root_lod1 column (convex hull)...")
        await conn.execute(text("""
            ALTER TABLE trees ADD COLUMN root_lod1 JSONB NULL
        """))
        
        print("  Adding root_lod2 column (cylinder)...")
        await conn.execute(text("""
            ALTER TABLE trees ADD COLUMN root_lod2 JSONB NULL
        """))
        
        print("  Adding root_lod3 column (circle)...")
        await conn.execute(text("""
            ALTER TABLE trees ADD COLUMN root_lod3 JSONB NULL
        """))
        
        print("  Adding root_seed column...")
        await conn.execute(text("""
            ALTER TABLE trees ADD COLUMN root_seed INTEGER NULL
        """))
        
        print("✓ Migration complete!")


async def rollback_migration():
    """Remove root prediction columns from trees table"""
    print("Rolling back migration: Removing root prediction columns...")
    
    async with async_engine.begin() as conn:
        columns = ['root_type', 'root_lod0', 'root_lod1', 'root_lod2', 'root_lod3', 'root_seed']
        
        for col in columns:
            try:
                print(f"  Dropping {col} column...")
                await conn.execute(text(f"ALTER TABLE trees DROP COLUMN IF EXISTS {col}"))
            except Exception as e:
                print(f"  Warning: Could not drop {col}: {e}")
        
        print("✓ Rollback complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run database migration for root prediction columns")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    args = parser.parse_args()
    
    if args.rollback:
        asyncio.run(rollback_migration())
    else:
        asyncio.run(run_migration())

