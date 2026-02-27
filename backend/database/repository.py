"""
Database repository for tree operations
"""
import logging
from typing import List, Optional, Tuple
from sqlalchemy import select, func, and_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from .models import Tree, PointCloud

logger = logging.getLogger(__name__)

# Tolerance for location matching (in degrees, ~5m at equator)
LOCATION_TOLERANCE = 0.00005


class TreeRepository:
    """Repository for tree database operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_or_create_point_cloud(
        self,
        filename: str,
        crs: str = "EPSG:31370",
        bbox_min: Optional[Tuple[float, float]] = None,
        bbox_max: Optional[Tuple[float, float]] = None,
        n_points: Optional[int] = None
    ) -> PointCloud:
        """Get existing or create new point cloud record"""
        # Check if point cloud with same filename exists
        # Use first() instead of scalar_one_or_none() to handle potential duplicates
        result = await self.session.execute(
            select(PointCloud).where(PointCloud.filename == filename).order_by(PointCloud.created_at.desc())
        )
        pc = result.scalars().first()
        
        if pc:
            logger.info(f"Found existing point cloud: {filename} (id={pc.id})")
            return pc
        
        # Create new point cloud
        pc = PointCloud(
            filename=filename,
            original_crs=crs,
            bbox_min_lon=bbox_min[0] if bbox_min else None,
            bbox_min_lat=bbox_min[1] if bbox_min else None,
            bbox_max_lon=bbox_max[0] if bbox_max else None,
            bbox_max_lat=bbox_max[1] if bbox_max else None,
            n_points=n_points
        )
        self.session.add(pc)
        await self.session.flush()
        logger.info(f"Created new point cloud record: {filename} (id={pc.id})")
        return pc
    
    async def find_tree_by_location(
        self,
        longitude: float,
        latitude: float,
        tolerance: float = LOCATION_TOLERANCE,
        preferred_point_cloud_id: Optional[int] = None,
    ) -> Optional[Tree]:
        """
        Find one existing tree near the given location.

        This intentionally returns the best match (first row) instead of
        expecting uniqueness, because historical data can contain duplicate
        rows at near-identical coordinates.
        """
        query = select(Tree).where(
            and_(
                func.abs(Tree.longitude - longitude) < tolerance,
                func.abs(Tree.latitude - latitude) < tolerance,
            )
        )
        if preferred_point_cloud_id is not None:
            query = query.order_by(
                case((Tree.point_cloud_id == preferred_point_cloud_id, 0), else_=1),
                Tree.updated_at.desc(),
                Tree.id.desc(),
            )
        else:
            query = query.order_by(Tree.updated_at.desc(), Tree.id.desc())
        result = await self.session.execute(query.limit(1))
        return result.scalars().first()

    async def find_tree_by_point_cloud_tree_id(
        self,
        point_cloud_id: int,
        tree_id: int,
    ) -> Optional[Tree]:
        """Find exact tree identity within a point cloud."""
        result = await self.session.execute(
            select(Tree)
            .where(
                and_(
                    Tree.point_cloud_id == point_cloud_id,
                    Tree.tree_id == tree_id,
                )
            )
            .order_by(Tree.updated_at.desc(), Tree.id.desc())
            .limit(1)
        )
        return result.scalars().first()
    
    async def upsert_tree(
        self,
        point_cloud_id: int,
        tree_id: int,
        longitude: float,
        latitude: float,
        height: float,
        crown_diameter: Optional[float] = None,
        n_points: Optional[int] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        x_original: Optional[float] = None,
        y_original: Optional[float] = None,
        bbox_min: Optional[list] = None,
        bbox_max: Optional[list] = None,
        point_cloud_path: Optional[str] = None
    ) -> Tree:
        """
        Insert or update tree record.
        Uses location-based deduplication to avoid creating duplicate trees.
        """
        # First, use deterministic identity for idempotent retries.
        existing = await self.find_tree_by_point_cloud_tree_id(point_cloud_id, tree_id)
        if existing is None:
            # Fallback to location-based dedup for cross-tile overlaps.
            existing = await self.find_tree_by_location(
                longitude,
                latitude,
                preferred_point_cloud_id=point_cloud_id,
            )
        
        if existing:
            # Update existing tree with new data
            existing.height = height
            existing.crown_diameter = crown_diameter
            existing.n_points = n_points
            existing.z_min = z_min
            existing.z_max = z_max
            existing.bbox_min = bbox_min
            existing.bbox_max = bbox_max
            if point_cloud_path:
                existing.point_cloud_path = point_cloud_path
            logger.debug(f"Updated existing tree at ({longitude:.6f}, {latitude:.6f})")
            return existing
        
        # Create new tree
        tree = Tree(
            point_cloud_id=point_cloud_id,
            tree_id=tree_id,
            longitude=longitude,
            latitude=latitude,
            height=height,
            crown_diameter=crown_diameter,
            n_points=n_points,
            z_min=z_min,
            z_max=z_max,
            x_original=x_original,
            y_original=y_original,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            point_cloud_path=point_cloud_path
        )
        self.session.add(tree)
        logger.debug(f"Created new tree at ({longitude:.6f}, {latitude:.6f})")
        return tree
    
    async def bulk_upsert_trees(
        self,
        point_cloud_id: int,
        trees_data: List[dict]
    ) -> List[Tree]:
        """
        Bulk insert/update trees.
        Returns list of tree objects.
        """
        result_trees = []
        
        for tree_data in trees_data:
            tree = await self.upsert_tree(
                point_cloud_id=point_cloud_id,
                tree_id=tree_data["tree_id"],
                longitude=tree_data["longitude"],
                latitude=tree_data["latitude"],
                height=tree_data["height"],
                crown_diameter=tree_data.get("crown_diameter"),
                n_points=tree_data.get("n_points"),
                z_min=tree_data.get("z_min"),
                z_max=tree_data.get("z_max"),
                x_original=tree_data.get("x"),
                y_original=tree_data.get("y"),
                bbox_min=tree_data.get("bbox_min"),
                bbox_max=tree_data.get("bbox_max"),
                point_cloud_path=tree_data.get("point_cloud_path")
            )
            result_trees.append(tree)
        
        await self.session.flush()
        return result_trees
    
    async def update_species_prediction(
        self,
        tree_id: int,
        point_cloud_id: int,
        species_id: int,
        species_name: str,
        confidence: float,
        probabilities: Optional[dict] = None
    ) -> Optional[Tree]:
        """Update species prediction for a tree"""
        result = await self.session.execute(
            select(Tree).where(
                and_(
                    Tree.point_cloud_id == point_cloud_id,
                    Tree.tree_id == tree_id
                )
            )
        )
        tree = result.scalar_one_or_none()
        
        if tree:
            tree.species_id = species_id
            tree.species_name = species_name
            tree.species_confidence = confidence
            tree.species_probabilities = probabilities
            logger.debug(f"Updated species for tree {tree_id}: {species_name}")
            return tree
        
        logger.warning(f"Tree {tree_id} not found for species update")
        return None
    
    async def update_species_by_db_id(
        self,
        db_id: int,
        species_id: int,
        species_name: str,
        confidence: float,
        probabilities: Optional[dict] = None
    ) -> Optional[Tree]:
        """Update species prediction for a tree by database ID"""
        result = await self.session.execute(
            select(Tree).where(Tree.id == db_id)
        )
        tree = result.scalar_one_or_none()
        
        if tree:
            tree.species_id = species_id
            tree.species_name = species_name
            tree.species_confidence = confidence
            tree.species_probabilities = probabilities
            return tree
        
        return None
    
    async def update_root_prediction(
        self,
        db_id: int,
        root_type: str,
        root_lod0: dict,
        root_lod1: dict,
        root_lod2: dict,
        root_lod3: dict,
        root_seed: Optional[int] = None
    ) -> Optional[Tree]:
        """Update root prediction for a tree by database ID"""
        result = await self.session.execute(
            select(Tree).where(Tree.id == db_id)
        )
        tree = result.scalar_one_or_none()
        
        if tree:
            tree.root_type = root_type
            tree.root_lod0 = root_lod0
            tree.root_lod1 = root_lod1
            tree.root_lod2 = root_lod2
            tree.root_lod3 = root_lod3
            tree.root_seed = root_seed
            logger.debug(f"Updated root prediction for tree {db_id}: {root_type}")
            return tree
        
        logger.warning(f"Tree {db_id} not found for root prediction update")
        return None
    
    async def update_shape_prediction(
        self,
        db_id: int,
        shape_mesh: dict
    ) -> Optional[Tree]:
        """Update shape prediction for a tree by database ID"""
        result = await self.session.execute(
            select(Tree).where(Tree.id == db_id)
        )
        tree = result.scalar_one_or_none()
        
        if tree:
            tree.shape_mesh = shape_mesh
            logger.debug(f"Updated shape prediction for tree {db_id}")
            return tree
        
        logger.warning(f"Tree {db_id} not found for shape prediction update")
        return None
    
    async def get_trees_without_roots(self, limit: int = 100) -> List[Tree]:
        """Get trees that don't have root predictions yet"""
        result = await self.session.execute(
            select(Tree).where(Tree.root_type.is_(None)).limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_all_trees(
        self,
        limit: int = 1000,
        offset: int = 0,
        with_species_only: bool = False
    ) -> List[Tree]:
        """Get all trees from database"""
        query = select(Tree).order_by(Tree.created_at.desc())
        
        if with_species_only:
            query = query.where(Tree.species_name.isnot(None))
        
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_tree_by_id(self, tree_id: int) -> Optional[Tree]:
        """Get single tree by database ID"""
        result = await self.session.execute(
            select(Tree).where(Tree.id == tree_id)
        )
        return result.scalar_one_or_none()
    
    async def get_trees_by_point_cloud(self, point_cloud_id: int) -> List[Tree]:
        """Get all trees for a specific point cloud"""
        result = await self.session.execute(
            select(Tree).where(Tree.point_cloud_id == point_cloud_id)
        )
        return list(result.scalars().all())
    
    async def get_trees_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float
    ) -> List[Tree]:
        """Get trees within a bounding box"""
        result = await self.session.execute(
            select(Tree).where(
                and_(
                    Tree.longitude >= min_lon,
                    Tree.longitude <= max_lon,
                    Tree.latitude >= min_lat,
                    Tree.latitude <= max_lat
                )
            )
        )
        return list(result.scalars().all())
    
    async def get_tree_count(self) -> int:
        """Get total number of trees"""
        result = await self.session.execute(
            select(func.count(Tree.id))
        )
        return result.scalar() or 0
    
    async def get_trees_stats(self) -> dict:
        """Get statistics about trees in database"""
        count = await self.get_tree_count()
        
        # Count by species
        species_result = await self.session.execute(
            select(
                Tree.species_name,
                func.count(Tree.id).label('count')
            ).where(
                Tree.species_name.isnot(None)
            ).group_by(Tree.species_name)
        )
        species_counts = {row[0]: row[1] for row in species_result.all()}
        
        # Get bounding box of all trees
        bbox_result = await self.session.execute(
            select(
                func.min(Tree.longitude),
                func.min(Tree.latitude),
                func.max(Tree.longitude),
                func.max(Tree.latitude)
            )
        )
        bbox = bbox_result.one_or_none()
        
        return {
            "total_trees": count,
            "trees_with_species": sum(species_counts.values()),
            "species_distribution": species_counts,
            "bbox": {
                "min_lon": bbox[0] if bbox else None,
                "min_lat": bbox[1] if bbox else None,
                "max_lon": bbox[2] if bbox else None,
                "max_lat": bbox[3] if bbox else None,
            } if bbox and bbox[0] is not None else None
        }
    
    async def commit(self):
        """Commit current transaction"""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback current transaction"""
        await self.session.rollback()


