"""
Pydantic schemas for the Tree Species Identifier API
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class TreeMetrics(BaseModel):
    """Metrics for an individual tree"""
    tree_id: int
    height: float  # meters
    crown_diameter: Optional[float] = None
    crown_volume: Optional[float] = None
    crown_surface: Optional[float] = None
    n_points: int
    
    # Position (in original CRS)
    x: float
    y: float
    z_min: float
    z_max: float
    
    # Position in WGS84
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    
    # Bounding box
    bbox_min: List[float]  # [x, y, z]
    bbox_max: List[float]  # [x, y, z]


class SpeciesPrediction(BaseModel):
    """Species prediction for a tree"""
    species_id: int
    species_name: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class TreeShape(BaseModel):
    """3D shape mesh for the tree"""
    vertices: List[List[float]]
    faces: List[List[int]]


class TreeInfo(BaseModel):
    """Complete information about a tree"""
    tree_id: int
    metrics: TreeMetrics
    species: Optional[SpeciesPrediction] = None
    root: Optional[Dict[str, Any]] = None
    shape: Optional[TreeShape] = None
    point_cloud_url: Optional[str] = None  # URL to download individual tree point cloud
    db_id: Optional[int] = None  # Database ID for persistent trees


class PointCloudInfo(BaseModel):
    """Information about the uploaded point cloud"""
    filename: str
    n_points: int
    crs: str
    bbox_min: List[float]
    bbox_max: List[float]
    center: List[float]  # Centroid in WGS84 for map positioning


class ProcessingStatus(BaseModel):
    """Status of processing job"""
    job_id: str
    status: str  # "pending", "segmenting", "predicting", "completed", "failed"
    progress: float  # 0-100
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProcessingResult(BaseModel):
    """Result of point cloud processing"""
    job_id: str
    status: str
    point_cloud: Optional[PointCloudInfo] = None
    trees: Optional[List[TreeInfo]] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response after uploading a point cloud"""
    job_id: str
    message: str


