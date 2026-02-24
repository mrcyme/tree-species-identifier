"""
SQLAlchemy models for tree data
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Float, Integer, DateTime, Text, JSON, 
    UniqueConstraint, Index, func, ForeignKey
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY

from .config import Base


class PointCloud(Base):
    """Represents a point cloud file that has been processed"""
    __tablename__ = "point_clouds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # File identification
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_crs: Mapped[str] = mapped_column(String(50), default="EPSG:31370")
    
    # Bounding box in WGS84
    bbox_min_lon: Mapped[float] = mapped_column(Float, nullable=True)
    bbox_min_lat: Mapped[float] = mapped_column(Float, nullable=True)
    bbox_max_lon: Mapped[float] = mapped_column(Float, nullable=True)
    bbox_max_lat: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Statistics
    n_points: Mapped[int] = mapped_column(Integer, nullable=True)
    n_trees: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Relationship to trees
    trees = relationship("Tree", back_populates="point_cloud", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PointCloud(id={self.id}, filename='{self.filename}', n_trees={self.n_trees})>"


class Tree(Base):
    """Represents a segmented tree with metrics and species prediction"""
    __tablename__ = "trees"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Reference to source point cloud
    point_cloud_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("point_clouds.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Tree identification - unique by location
    tree_id: Mapped[int] = mapped_column(Integer, nullable=False)  # ID within point cloud
    
    # Location in WGS84 (primary identifier for deduplication)
    longitude: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    
    # Original coordinates (source CRS)
    x_original: Mapped[float] = mapped_column(Float, nullable=True)
    y_original: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Elevation
    z_min: Mapped[float] = mapped_column(Float, nullable=True)
    z_max: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Tree metrics
    height: Mapped[float] = mapped_column(Float, nullable=True)
    crown_diameter: Mapped[float] = mapped_column(Float, nullable=True)
    n_points: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Bounding box in WGS84
    bbox_min: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [lon, lat, z]
    bbox_max: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [lon, lat, z]
    
    # Species prediction
    species_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    species_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    species_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    species_probabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Path to individual tree point cloud file
    point_cloud_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Root prediction - 4 levels of detail stored as JSON
    root_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'taproot', 'heart', 'plate'
    root_lod0: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Full mesh: {vertices, faces}
    root_lod1: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Convex hull: {vertices, faces}
    root_lod2: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Cylinder: {center, radius, height}
    root_lod3: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Circle: {center, radius}
    root_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Seed for reproducibility
    
    # Shape prediction stored as JSON
    shape_mesh: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # {vertices, faces}
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Relationship back to point cloud
    point_cloud = relationship("PointCloud", back_populates="trees")
    
    # Unique constraint on location (rounded to ~0.5m precision)
    # This prevents duplicate trees at the same location
    __table_args__ = (
        # Index for spatial queries
        Index('idx_tree_location', 'longitude', 'latitude'),
        # Unique constraint based on rounded coordinates (approx 0.5m tolerance)
        # We use a unique constraint on (point_cloud_id, tree_id) for exact matches
        UniqueConstraint('point_cloud_id', 'tree_id', name='uq_tree_in_pointcloud'),
    )
    
    def __repr__(self):
        return f"<Tree(id={self.id}, species='{self.species_name}', height={self.height:.1f}m)>"
    
    def to_dict(self, include_root_mesh: bool = True) -> dict:
        """
        Convert to dictionary for API response.
        
        Args:
            include_root_mesh: If True, include full root mesh data (LOD0-LOD3).
                               If False, only include root_type and root_lod3 (circle).
                               Set to False for list views to reduce payload size.
        """
        result = {
            "id": self.id,
            "tree_id": self.tree_id,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "height": self.height,
            "crown_diameter": self.crown_diameter,
            "n_points": self.n_points,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "species_id": self.species_id,
            "species_name": self.species_name,
            "species_confidence": self.species_confidence,
            "point_cloud_path": self.point_cloud_path,
            "root_type": self.root_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if include_root_mesh:
            # Include full root mesh data
            result["root_lod0"] = self.root_lod0
            result["root_lod1"] = self.root_lod1
            result["root_lod2"] = self.root_lod2
            result["root_lod3"] = self.root_lod3
            result["root_seed"] = self.root_seed
            result["shape"] = self.shape_mesh
        else:
            # Only include lightweight root data (circle footprint)
            result["root_lod3"] = self.root_lod3
        
        return result

