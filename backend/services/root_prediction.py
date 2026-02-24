"""
Root Prediction Service

Generates procedural tree roots based on tree height and crown diameter.
Uses the TreeRootGenerator from root_generation module.

Root Types:
- Taproot: Deep central root with smaller lateral branches
- Heart: Radiating roots spreading downward and outward  
- Plate (Shallow): Wide, shallow root system
"""
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

# Root types matching the original implementation
class RootType(Enum):
    TAPROOT = "taproot"
    HEART = "heart"
    PLATE = "plate"


@dataclass
class BranchParameters:
    """Parameters controlling branch generation"""
    branching_angle_mean: float = 35.0
    branching_angle_std: float = 10.0
    branching_probability: float = 0.7
    max_branches_per_node: int = 3
    gravitropism: float = 0.3
    random_tilt: float = 15.0
    segment_length: float = 0.15
    segment_length_decay: float = 0.92
    initial_radius: float = 0.08
    radius_decay: float = 0.85
    min_radius: float = 0.005
    max_depth: int = 6
    depth_probability_decay: float = 0.85


@dataclass
class RootSegment:
    """Represents a single root segment"""
    start: np.ndarray
    end: np.ndarray
    radius: float
    depth: int


class TreeRootGenerator:
    """Generates tree roots at multiple levels of detail"""
    
    def __init__(self, root_type: RootType = RootType.HEART, seed: Optional[int] = None):
        self.root_type = root_type
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.params = self._get_default_params(root_type)
        self.segments: List[RootSegment] = []
        self.all_points: List[np.ndarray] = []
        
    def _get_default_params(self, root_type: RootType) -> BranchParameters:
        """Get default parameters for each root type"""
        if root_type == RootType.TAPROOT:
            return BranchParameters(
                branching_angle_mean=25.0,
                branching_angle_std=8.0,
                branching_probability=0.5,
                max_branches_per_node=2,
                gravitropism=0.7,
                random_tilt=10.0,
                segment_length=0.2,
                segment_length_decay=0.9,
                initial_radius=0.1,
                radius_decay=0.8,
                min_radius=0.005,
                max_depth=8,
                depth_probability_decay=0.9
            )
        elif root_type == RootType.HEART:
            return BranchParameters(
                branching_angle_mean=40.0,
                branching_angle_std=15.0,
                branching_probability=0.75,
                max_branches_per_node=3,
                gravitropism=0.4,
                random_tilt=20.0,
                segment_length=0.15,
                segment_length_decay=0.92,
                initial_radius=0.08,
                radius_decay=0.85,
                min_radius=0.004,
                max_depth=6,
                depth_probability_decay=0.85
            )
        else:  # PLATE
            return BranchParameters(
                branching_angle_mean=60.0,
                branching_angle_std=20.0,
                branching_probability=0.8,
                max_branches_per_node=4,
                gravitropism=0.15,
                random_tilt=25.0,
                segment_length=0.18,
                segment_length_decay=0.95,
                initial_radius=0.06,
                radius_decay=0.88,
                min_radius=0.003,
                max_depth=5,
                depth_probability_decay=0.8
            )
    
    def scale_to_tree(self, tree_height: float, crown_diameter: float):
        """Scale root parameters based on tree dimensions"""
        # Estimate tree age from height (rough approximation)
        # Trees typically grow 0.3-1m per year, use average
        estimated_age = max(5, tree_height / 0.5)
        
        # Scale factors based on tree size
        size_factor = (tree_height + crown_diameter) / 15.0  # Normalized to medium tree
        size_factor = max(0.3, min(3.0, size_factor))
        
        # Scale segment length based on tree size
        self.params.segment_length *= size_factor
        
        # Scale radii
        self.params.initial_radius *= size_factor
        self.params.min_radius *= size_factor
        
        # Adjust complexity based on tree size
        if estimated_age > 30:
            self.params.max_depth = min(10, self.params.max_depth + 2)
        elif estimated_age < 10:
            self.params.max_depth = max(3, self.params.max_depth - 2)
    
    def _rotate_vector(self, vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotate vector around axis by angle (radians) using Rodrigues' formula"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return (vec * cos_a + 
                np.cross(axis, vec) * sin_a + 
                axis * np.dot(axis, vec) * (1 - cos_a))
    
    def _get_perpendicular(self, vec: np.ndarray) -> np.ndarray:
        """Get a perpendicular vector"""
        if abs(vec[0]) < 0.9:
            perp = np.cross(vec, np.array([1, 0, 0]))
        else:
            perp = np.cross(vec, np.array([0, 1, 0]))
        return perp / np.linalg.norm(perp)
    
    def _generate_branch_direction(self, parent_direction: np.ndarray, depth: int) -> np.ndarray:
        """Generate a new branch direction based on parent and parameters"""
        angle = np.radians(self.rng.normal(
            self.params.branching_angle_mean,
            self.params.branching_angle_std
        ))
        
        rotation_angle = self.rng.uniform(0, 2 * np.pi)
        perp = self._get_perpendicular(parent_direction)
        perp = self._rotate_vector(perp, parent_direction, rotation_angle)
        new_direction = self._rotate_vector(parent_direction, perp, angle)
        
        random_axis = self._get_perpendicular(new_direction)
        random_tilt = np.radians(self.rng.normal(0, self.params.random_tilt))
        new_direction = self._rotate_vector(new_direction, random_axis, random_tilt)
        
        down = np.array([0, 0, -1])
        new_direction = new_direction * (1 - self.params.gravitropism) + down * self.params.gravitropism
        
        return new_direction / np.linalg.norm(new_direction)
    
    def _generate_branch(self, start: np.ndarray, direction: np.ndarray, 
                         radius: float, depth: int) -> None:
        """Recursively generate a branch and its children"""
        if depth >= self.params.max_depth or radius < self.params.min_radius:
            return
        
        length = self.params.segment_length * (self.params.segment_length_decay ** depth)
        end = start + direction * length
        
        segment = RootSegment(
            start=start.copy(),
            end=end.copy(),
            radius=radius,
            depth=depth
        )
        
        self.segments.append(segment)
        self.all_points.append(start.copy())
        self.all_points.append(end.copy())
        
        depth_factor = self.params.depth_probability_decay ** depth
        
        # Continue main branch
        if self.rng.random() < 0.9 * depth_factor:
            continued_direction = self._generate_branch_direction(direction, depth)
            continued_direction = direction * 0.7 + continued_direction * 0.3
            continued_direction /= np.linalg.norm(continued_direction)
            
            self._generate_branch(
                end,
                continued_direction,
                radius * self.params.radius_decay,
                depth + 1
            )
        
        # Generate side branches
        if self.rng.random() < self.params.branching_probability * depth_factor:
            num_branches = self.rng.integers(1, self.params.max_branches_per_node + 1)
            
            for _ in range(num_branches):
                branch_direction = self._generate_branch_direction(direction, depth)
                branch_radius = radius * self.params.radius_decay * self.rng.uniform(0.6, 0.9)
                
                self._generate_branch(
                    end,
                    branch_direction,
                    branch_radius,
                    depth + 1
                )
    
    def generate(self, num_primary_roots: int = None) -> None:
        """Generate the complete root system"""
        self.segments = []
        self.all_points = []
        
        origin = np.array([0.0, 0.0, 0.0])
        
        if num_primary_roots is None:
            if self.root_type == RootType.TAPROOT:
                num_primary_roots = 1
            elif self.root_type == RootType.HEART:
                num_primary_roots = 5
            else:
                num_primary_roots = 8
        
        if self.root_type == RootType.TAPROOT:
            # Main taproot going straight down
            self._generate_branch(
                origin,
                np.array([0, 0, -1]),
                self.params.initial_radius,
                0
            )
            
            # Lateral roots
            for i in range(4):
                angle = i * np.pi / 2 + self.rng.uniform(-0.3, 0.3)
                direction = np.array([
                    np.cos(angle) * 0.6,
                    np.sin(angle) * 0.6,
                    -0.8
                ])
                direction /= np.linalg.norm(direction)
                
                self._generate_branch(
                    origin,
                    direction,
                    self.params.initial_radius * 0.5,
                    1
                )
        else:
            # Radial distribution for heart and plate roots
            for i in range(num_primary_roots):
                angle = i * 2 * np.pi / num_primary_roots + self.rng.uniform(-0.2, 0.2)
                
                if self.root_type == RootType.HEART:
                    horizontal = self.rng.uniform(0.4, 0.7)
                    vertical = -np.sqrt(1 - horizontal**2)
                else:  # PLATE
                    horizontal = self.rng.uniform(0.7, 0.95)
                    vertical = -np.sqrt(1 - horizontal**2)
                
                direction = np.array([
                    np.cos(angle) * horizontal,
                    np.sin(angle) * horizontal,
                    vertical
                ])
                
                self._generate_branch(
                    origin,
                    direction,
                    self.params.initial_radius * self.rng.uniform(0.8, 1.0),
                    0
                )
    
    def get_lod0_mesh(self) -> Dict[str, Any]:
        """LOD 0: Full detail mesh with vertices and faces"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        n_sides = 8
        
        for segment in self.segments:
            direction = segment.end - segment.start
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction /= length
            
            perp1 = self._get_perpendicular(direction)
            perp2 = np.cross(direction, perp1)
            
            start_circle = []
            end_circle = []
            
            for i in range(n_sides):
                angle = i * 2 * np.pi / n_sides
                offset = (np.cos(angle) * perp1 + np.sin(angle) * perp2) * segment.radius
                start_circle.append(segment.start + offset)
                end_circle.append(segment.end + offset * self.params.radius_decay)
            
            all_vertices.extend(start_circle)
            all_vertices.extend(end_circle)
            
            for i in range(n_sides):
                next_i = (i + 1) % n_sides
                all_faces.append([
                    vertex_offset + i,
                    vertex_offset + next_i,
                    vertex_offset + n_sides + next_i
                ])
                all_faces.append([
                    vertex_offset + i,
                    vertex_offset + n_sides + next_i,
                    vertex_offset + n_sides + i
                ])
            
            vertex_offset = len(all_vertices)
        
        if not all_vertices:
            return {"vertices": [], "faces": []}
        
        vertices = [[float(v[0]), float(v[1]), float(v[2])] for v in all_vertices]
        faces = [[int(f[0]), int(f[1]), int(f[2])] for f in all_faces]
        
        return {"vertices": vertices, "faces": faces}
    
    def get_lod1_mesh(self, max_faces: int = 50) -> Dict[str, Any]:
        """LOD 1: Convex hull bounding volume"""
        if len(self.all_points) < 4:
            return {"vertices": [], "faces": []}
        
        points = np.array(self.all_points)
        
        # Add padding points
        padded_points = []
        for segment in self.segments:
            for point in [segment.start, segment.end]:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            offset = np.array([dx, dy, dz]) * segment.radius
                            padded_points.append(point + offset)
        
        if padded_points:
            points = np.vstack([points, np.array(padded_points)])
        
        try:
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            
            if len(hull.simplices) > max_faces:
                step = max(1, len(hull.vertices) // int(np.sqrt(max_faces)))
                simplified_indices = hull.vertices[::step]
                simplified_points = points[simplified_indices]
                if len(simplified_points) >= 4:
                    simplified_hull = ConvexHull(simplified_points)
                    verts = [[float(v[0]), float(v[1]), float(v[2])] for v in simplified_points]
                    faces = [[int(f[0]), int(f[1]), int(f[2])] for f in simplified_hull.simplices]
                    return {"vertices": verts, "faces": faces}
            
            verts = [[float(v[0]), float(v[1]), float(v[2])] for v in points[hull.vertices]]
            faces = [[int(f[0]), int(f[1]), int(f[2])] for f in hull.simplices]
            return {"vertices": verts, "faces": faces}
        except Exception:
            return {"vertices": [], "faces": []}
    
    def get_lod2_cylinder(self) -> Dict[str, Any]:
        """LOD 2: Bounding cylinder"""
        if not self.all_points:
            return {"center": [0, 0, 0], "radius": 0, "height": 0}
        
        points = np.array(self.all_points)
        xy_points = points[:, :2]
        center_xy = np.mean(xy_points, axis=0)
        distances = np.linalg.norm(xy_points - center_xy, axis=1)
        radius = float(np.max(distances) * 1.1)
        
        z_min = float(np.min(points[:, 2]))
        z_max = float(np.max(points[:, 2]))
        height = z_max - z_min
        
        center = [float(center_xy[0]), float(center_xy[1]), float((z_min + z_max) / 2)]
        
        return {"center": center, "radius": radius, "height": height}
    
    def get_lod3_circle(self) -> Dict[str, Any]:
        """LOD 3: Bounding circle (2D footprint)"""
        if not self.all_points:
            return {"center": [0, 0], "radius": 0}
        
        points = np.array(self.all_points)
        xy_points = points[:, :2]
        center = np.mean(xy_points, axis=0)
        distances = np.linalg.norm(xy_points - center, axis=1)
        radius = float(np.max(distances) * 1.1)
        
        return {"center": [float(center[0]), float(center[1])], "radius": radius}


def predict_roots(
    tree_height: float,
    crown_diameter: float,
    root_type: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Predict roots for a tree based on its dimensions.
    
    Args:
        tree_height: Height of the tree in meters
        crown_diameter: Crown diameter in meters
        root_type: Optional root type ('taproot', 'heart', 'plate'). If None, randomly selected.
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with root data for all 4 LOD levels
    """
    # Select random root type if not specified
    if root_type is None:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        root_type = rng.choice(['taproot', 'heart', 'plate'])
    
    root_type_enum = RootType(root_type)
    
    # Create generator
    generator = TreeRootGenerator(root_type=root_type_enum, seed=seed)
    
    # Scale to tree dimensions
    generator.scale_to_tree(tree_height, crown_diameter)
    
    # Generate roots
    generator.generate()
    
    # Get all LOD levels
    lod0 = generator.get_lod0_mesh()
    lod1 = generator.get_lod1_mesh()
    lod2 = generator.get_lod2_cylinder()
    lod3 = generator.get_lod3_circle()
    
    return {
        "root_type": root_type,
        "lod0": lod0,  # Full mesh
        "lod1": lod1,  # Convex hull
        "lod2": lod2,  # Cylinder
        "lod3": lod3,  # Circle
        "metadata": {
            "tree_height": tree_height,
            "crown_diameter": crown_diameter,
            "num_segments": len(generator.segments),
            "seed": seed
        }
    }


def run_root_prediction_for_tree(
    tree_id: int,
    height: float,
    crown_diameter: float,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run root prediction for a single tree.
    
    Args:
        tree_id: ID of the tree
        height: Tree height in meters
        crown_diameter: Crown diameter in meters
        seed: Optional seed (if None, uses tree_id as seed for reproducibility)
        
    Returns:
        Root prediction result
    """
    if seed is None:
        seed = tree_id  # Use tree_id as seed for reproducibility
    
    logger.info(f"Predicting roots for tree {tree_id}: height={height:.1f}m, crown={crown_diameter:.1f}m")
    
    result = predict_roots(
        tree_height=height,
        crown_diameter=crown_diameter,
        root_type=None,  # Random selection
        seed=seed
    )
    
    result["tree_id"] = tree_id
    
    logger.info(f"Generated {result['root_type']} roots with {result['metadata']['num_segments']} segments")
    
    return result






