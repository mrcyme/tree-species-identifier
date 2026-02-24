import axios from 'axios';
import type { 
  ProcessingStatus, 
  ProcessingResult, 
  TreeInfo, 
  PointCloudData,
  TreePointCloudData,
  SpeciesInfo,
  RootMesh,
  RootCylinder,
  RootCircle,
  TreeShape
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function uploadPointCloud(
  file: File, 
  crs: string = '31370',
  nAug: number = 5
): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/api/upload', formData, {
    params: { crs, n_aug: nAug },
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<ProcessingStatus> {
  const response = await api.get(`/api/jobs/${jobId}`);
  return response.data;
}

export async function getJobResult(jobId: string): Promise<ProcessingResult> {
  const response = await api.get(`/api/jobs/${jobId}/result`);
  return response.data;
}

export async function getTrees(jobId: string): Promise<TreeInfo[]> {
  const response = await api.get(`/api/jobs/${jobId}/trees`);
  return response.data;
}

export async function getTree(jobId: string, treeId: number): Promise<TreeInfo> {
  const response = await api.get(`/api/jobs/${jobId}/trees/${treeId}`);
  return response.data;
}

export async function getTreePointCloud(jobId: string, treeId: number): Promise<TreePointCloudData> {
  const response = await api.get(`/api/jobs/${jobId}/trees/${treeId}/pointcloud`);
  return response.data;
}

export async function getFullPointCloud(jobId: string): Promise<PointCloudData> {
  const response = await api.get(`/api/jobs/${jobId}/pointcloud`);
  return response.data;
}

export async function getSpeciesList(): Promise<SpeciesInfo[]> {
  const response = await api.get('/api/species');
  return response.data;
}

// Database-backed tree endpoints
export interface DbTree {
  id: number;
  tree_id: number;
  longitude: number;
  latitude: number;
  height: number | null;
  crown_diameter: number | null;
  n_points: number | null;
  z_min: number | null;
  z_max: number | null;
  bbox_min: number[] | null;
  bbox_max: number[] | null;
  species_id: number | null;
  species_name: string | null;
  species_confidence: number | null;
  point_cloud_path: string | null;
  // Root prediction fields
  root_type: 'taproot' | 'heart' | 'plate' | null;
  root_lod0?: RootMesh | null;  // Full mesh (only in detail view)
  root_lod1?: RootMesh | null;  // Convex hull (only in detail view)
  root_lod2?: RootCylinder | null;  // Cylinder (only in detail view)
  root_lod3?: RootCircle | null;  // Circle (included in list view)
  root_seed?: number | null;
  // Shape prediction
  shape_mesh?: TreeShape | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface DbTreesResponse {
  trees: DbTree[];
  count: number;
  limit: number;
  offset: number;
}

export interface TreeStats {
  total_trees: number;
  trees_with_species: number;
  species_distribution: Record<string, number>;
  bbox: {
    min_lon: number | null;
    min_lat: number | null;
    max_lon: number | null;
    max_lat: number | null;
  } | null;
}

export async function getAllTreesFromDb(
  limit: number = 1000,
  offset: number = 0,
  withSpecies: boolean = false
): Promise<DbTreesResponse> {
  const response = await api.get('/api/trees', {
    params: { limit, offset, with_species: withSpecies }
  });
  return response.data;
}

export async function getTreeStats(): Promise<TreeStats> {
  const response = await api.get('/api/trees/stats');
  return response.data;
}

export async function getTreeFromDb(treeId: number): Promise<DbTree> {
  const response = await api.get(`/api/trees/${treeId}`);
  return response.data;
}

export async function getDbTreePointCloud(treeId: number): Promise<TreePointCloudData> {
  const response = await api.get(`/api/trees/${treeId}/pointcloud`);
  return response.data;
}

export async function getTreesInBbox(
  minLon: number,
  minLat: number,
  maxLon: number,
  maxLat: number
): Promise<{ trees: DbTree[]; count: number }> {
  const response = await api.get('/api/trees/bbox', {
    params: { min_lon: minLon, min_lat: minLat, max_lon: maxLon, max_lat: maxLat }
  });
  return response.data;
}

// Poll job status until complete
export async function pollJobStatus(
  jobId: string, 
  onProgress: (status: ProcessingStatus) => void,
  intervalMs: number = 2000
): Promise<ProcessingResult> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        onProgress(status);
        
        if (status.status === 'completed') {
          const result = await getJobResult(jobId);
          resolve(result);
        } else if (status.status === 'failed') {
          const result = await getJobResult(jobId);
          reject(new Error(result.error || 'Processing failed'));
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (error) {
        reject(error);
      }
    };
    
    poll();
  });
}



