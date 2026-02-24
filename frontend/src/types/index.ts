// Tree-related types

export interface TreeMetrics {
  tree_id: number;
  height: number;
  crown_diameter?: number;
  crown_volume?: number;
  crown_surface?: number;
  n_points: number;
  // WGS84 coordinates (longitude, latitude)
  longitude: number;
  latitude: number;
  // Original coordinates
  x: number;
  y: number;
  z_min: number;
  z_max: number;
  // Bounding box in WGS84 [lon, lat, z]
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
}

export interface SpeciesPrediction {
  species_id: number;
  species_name: string;
  confidence: number;
  probabilities?: Record<string, number>;
}

// Root mesh data (vertices and faces for 3D rendering)
export interface RootMesh {
  vertices: [number, number, number][];
  faces: [number, number, number][];
}

// Root cylinder data (LOD2)
export interface RootCylinder {
  center: [number, number, number];
  radius: number;
  height: number;
}

// Root circle data (LOD3)
export interface RootCircle {
  center: [number, number];
  radius: number;
}

// Root prediction data with all LOD levels
export interface RootPrediction {
  root_type: 'taproot' | 'heart' | 'plate';
  lod0: RootMesh;  // Full detail mesh
  lod1: RootMesh;  // Convex hull
  lod2: RootCylinder;  // Bounding cylinder
  lod3: RootCircle;  // Bounding circle
  metadata?: {
    tree_height?: number;
    crown_diameter?: number;
    num_segments?: number;
    seed?: number;
  };
}

// Tree shape mesh (trunk + canopy convex hull)
export interface TreeShape {
  vertices: [number, number, number][];
  faces: [number, number, number][];
}

export interface TreeInfo {
  tree_id: number;
  metrics: TreeMetrics;
  species?: SpeciesPrediction;
  root?: RootPrediction;  // Root prediction data
  shape?: TreeShape;      // 3D shape mesh
  point_cloud_url?: string;
  db_id?: number;  // Database ID for persistent trees
}

export interface PointCloudInfo {
  filename: string;
  n_points: number;
  crs: string;
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
  center: [number, number, number];
  data_url: string;
}

export interface ProcessingStatus {
  job_id: string;
  status: 'pending' | 'analyzing' | 'segmenting' | 'converting' | 'predicting' | 'finalizing' | 'completed' | 'failed';
  progress: number;
  message?: string;
}

export interface ProcessingResult {
  job_id: string;
  status: string;
  point_cloud?: PointCloudInfo;
  trees?: TreeInfo[];
  tree_count?: number;
  error?: string;
}

export interface PointCloudData {
  positions: [number, number, number][];
  colors: [number, number, number][];
  tree_ids?: number[];
  count: number;
}

export interface TreePointCloudData extends PointCloudData {
  tree_id: number;
  center: [number, number, number];
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
}

// Species info
export interface SpeciesInfo {
  id: number;
  name: string;
  common_name: string;
}

