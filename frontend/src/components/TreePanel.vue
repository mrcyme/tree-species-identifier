<script setup lang="ts">
import { ref, watch, computed } from 'vue';
import type { TreeInfo, TreePointCloudData, RootMesh, RootCylinder, RootCircle, TreeShape } from '../types';
import { getTreePointCloud, getDbTreePointCloud, getTreeFromDb, type DbTree } from '../services/api';
import TreeViewer from './TreeViewer.vue';

const props = defineProps<{
  tree: TreeInfo | null;
  jobId: string | null;
  useDbEndpoint?: boolean;
}>();

const emit = defineEmits<{
  close: [];
}>();

const pointCloud = ref<TreePointCloudData | null>(null);
const fullTreeData = ref<DbTree | null>(null);  // Full tree data including root mesh
const loading = ref(false);

// Get root LOD0 (full mesh) from full tree data or props
const rootLod0 = computed<RootMesh | null>(() => {
  if (fullTreeData.value?.root_lod0) return fullTreeData.value.root_lod0 as RootMesh;
  if (props.tree?.root?.lod0) return props.tree.root.lod0;
  const dbTree = props.tree as any;
  if (dbTree?.root_lod0) return dbTree.root_lod0;
  return null;
});

// Get root LOD1 (convex hull)
const rootLod1 = computed<RootMesh | null>(() => {
  if (fullTreeData.value?.root_lod1) return fullTreeData.value.root_lod1 as RootMesh;
  if (props.tree?.root?.lod1) return props.tree.root.lod1;
  const dbTree = props.tree as any;
  if (dbTree?.root_lod1) return dbTree.root_lod1;
  return null;
});

// Get root LOD2 (cylinder)
const rootLod2 = computed<RootCylinder | null>(() => {
  if (fullTreeData.value?.root_lod2) return fullTreeData.value.root_lod2 as RootCylinder;
  if (props.tree?.root?.lod2) return props.tree.root.lod2;
  const dbTree = props.tree as any;
  if (dbTree?.root_lod2) return dbTree.root_lod2;
  return null;
});

// Get root LOD3 (circle)
const rootLod3 = computed<RootCircle | null>(() => {
  if (fullTreeData.value?.root_lod3) return fullTreeData.value.root_lod3 as RootCircle;
  if (props.tree?.root?.lod3) return props.tree.root.lod3;
  const dbTree = props.tree as any;
  if (dbTree?.root_lod3) return dbTree.root_lod3;
  return null;
});

// Get shape mesh
const shapeMesh = computed<TreeShape | null>(() => {
  if (fullTreeData.value?.shape_mesh) return fullTreeData.value.shape_mesh as TreeShape;
  if (props.tree?.shape) return props.tree.shape;
  const dbTree = props.tree as any;
  if (dbTree?.shape_mesh) return dbTree.shape_mesh;
  return null;
});

// Root type formatted for display
const rootTypeDisplay = computed(() => {
  const rootType = fullTreeData.value?.root_type || props.tree?.root?.root_type || (props.tree as any)?.root_type;
  if (!rootType) return null;
  
  const typeMap: Record<string, string> = {
    'taproot': 'Taproot (Deep Central)',
    'heart': 'Heart (Radiating)',
    'plate': 'Plate (Shallow Spreading)'
  };
  return typeMap[rootType] || rootType;
});

watch([() => props.tree, () => props.jobId, () => props.useDbEndpoint], async ([newTree, newJobId, useDb]) => {
  if (!newTree) {
    pointCloud.value = null;
    fullTreeData.value = null;
    return;
  }
  
  loading.value = true;
  fullTreeData.value = null;
  
  try {
    if (useDb && newTree.db_id) {
      // Load full tree data (including root mesh) from database endpoint
      const [treeData, pcData] = await Promise.all([
        getTreeFromDb(newTree.db_id),
        getDbTreePointCloud(newTree.db_id)
      ]);
      fullTreeData.value = treeData;
      pointCloud.value = pcData;
    } else if (newJobId) {
      // Load from job endpoint
      pointCloud.value = await getTreePointCloud(newJobId, newTree.tree_id);
    } else {
      pointCloud.value = null;
    }
  } catch (e) {
    console.error('Failed to load tree data:', e);
    pointCloud.value = null;
    fullTreeData.value = null;
  } finally {
    loading.value = false;
  }
}, { immediate: true });

const confidenceColor = computed(() => {
  if (!props.tree?.species?.confidence) return '#666';
  const conf = props.tree.species.confidence;
  if (conf >= 0.7) return '#22c55e';
  if (conf >= 0.5) return '#f59e0b';
  return '#ef4444';
});

function formatSpeciesName(name: string): string {
  return name.replace(/_/g, ' ');
}

// Get top 5 species probabilities
const topProbabilities = computed(() => {
  if (!props.tree?.species?.probabilities) return [];
  
  const probs = Object.entries(props.tree.species.probabilities)
    .map(([name, prob]) => ({ name, prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);
  
  return probs;
});

// Display DB ID if present
const treeLabel = computed(() => {
  if (props.tree?.db_id) {
    return `Tree #${props.tree.tree_id} (DB: ${props.tree.db_id})`;
  }
  return `Tree #${props.tree?.tree_id}`;
});
</script>

<template>
  <div class="tree-panel" v-if="tree">
    <div class="panel-header">
      <h3>🌲 {{ treeLabel }}</h3>
      <button class="close-btn" @click="emit('close')">✕</button>
    </div>
    
    <div class="viewer-container">
      <TreeViewer 
        v-if="pointCloud" 
        :pointCloud="pointCloud"
        :rootLod0="rootLod0"
        :rootLod1="rootLod1"
        :rootLod2="rootLod2"
        :rootLod3="rootLod3"
        :shape="shapeMesh"
      />
      <div v-else-if="loading" class="loading">
        <span>Loading point cloud...</span>
      </div>
      <div v-else class="loading">
        <span>No point cloud available</span>
      </div>
    </div>
    
    <div class="info-section">
      <h4>Species Prediction</h4>
      <div v-if="tree.species" class="species-info">
        <div class="species-name">
          {{ formatSpeciesName(tree.species.species_name) }}
        </div>
        <div class="confidence-bar">
          <div class="confidence-label">
            Confidence: {{ (tree.species.confidence * 100).toFixed(1) }}%
          </div>
          <div class="bar-bg">
            <div 
              class="bar-fill"
              :style="{ 
                width: (tree.species.confidence * 100) + '%',
                backgroundColor: confidenceColor 
              }"
            ></div>
          </div>
        </div>
        
        <div v-if="topProbabilities.length" class="probabilities">
          <div class="prob-title">Top predictions:</div>
          <div 
            v-for="p in topProbabilities" 
            :key="p.name" 
            class="prob-item"
          >
            <span class="prob-name">{{ formatSpeciesName(p.name) }}</span>
            <span class="prob-value">{{ (p.prob * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
      <div v-else class="no-species">
        Species prediction unavailable
      </div>
    </div>
    
    <div class="info-section">
      <h4>Measurements</h4>
      <div class="metrics-grid">
        <div class="metric">
          <span class="metric-label">Height</span>
          <span class="metric-value">{{ tree.metrics.height.toFixed(1) }} m</span>
        </div>
        <div class="metric">
          <span class="metric-label">Crown Ø</span>
          <span class="metric-value">{{ (tree.metrics.crown_diameter || 0).toFixed(1) }} m</span>
        </div>
        <div class="metric">
          <span class="metric-label">Points</span>
          <span class="metric-value">{{ tree.metrics.n_points.toLocaleString() }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Z Range</span>
          <span class="metric-value">
            {{ tree.metrics.z_min.toFixed(1) }} - {{ tree.metrics.z_max.toFixed(1) }} m
          </span>
        </div>
      </div>
    </div>
    
    <div class="info-section">
      <h4>Position</h4>
      <div class="position-info">
        <div v-if="tree.metrics.longitude && tree.metrics.latitude">
          <div>Longitude: {{ tree.metrics.longitude.toFixed(6) }}°</div>
          <div>Latitude: {{ tree.metrics.latitude.toFixed(6) }}°</div>
        </div>
        <div v-else>
          <div>X: {{ tree.metrics.x.toFixed(2) }}</div>
          <div>Y: {{ tree.metrics.y.toFixed(2) }}</div>
        </div>
      </div>
    </div>
    
    <div class="info-section" v-if="rootTypeDisplay">
      <h4>🌱 Root System</h4>
      <div class="root-info">
        <div class="root-type">
          <span class="root-type-label">Type:</span>
          <span class="root-type-value">{{ rootTypeDisplay }}</span>
        </div>
        <div class="root-description">
          <p v-if="rootTypeDisplay?.includes('Taproot')">
            Deep central root with smaller lateral branches. Common in oaks and carrots.
          </p>
          <p v-else-if="rootTypeDisplay?.includes('Heart')">
            Radiating roots spreading downward and outward. Common in most deciduous trees.
          </p>
          <p v-else-if="rootTypeDisplay?.includes('Plate')">
            Wide, shallow root system staying near the surface. Common in spruce and birch.
          </p>
        </div>
        <div class="root-note" v-if="rootLod0">
          <span>ℹ️ Roots are displayed in the 3D view above (toggle LOD with 🌱 button)</span>
        </div>
        <div class="root-note" v-else-if="loading">
          <span>⏳ Loading root data...</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tree-panel {
  background: rgba(15, 23, 42, 0.95);
  border-radius: 12px;
  color: #e2e8f0;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  max-height: 100%;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #334155;
}

.panel-header h3 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  color: #94a3b8;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0.25rem;
  line-height: 1;
}

.close-btn:hover {
  color: #f1f5f9;
}

.viewer-container {
  height: 450px;
  min-height: 400px;
  background: linear-gradient(135deg, #e8f4ea 0%, #d4e8d8 50%, #c5dcc9 100%);
  position: relative;
  border-radius: 8px;
  margin: 0 0.5rem;
}

.loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #64748b;
}

.info-section {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #334155;
}

.info-section:last-child {
  border-bottom: none;
}

.info-section h4 {
  margin: 0 0 0.75rem;
  font-size: 0.85rem;
  font-weight: 600;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.species-info .species-name {
  font-size: 1.1rem;
  font-weight: 600;
  font-style: italic;
  color: #22c55e;
  margin-bottom: 0.75rem;
}

.confidence-bar {
  margin-bottom: 1rem;
}

.confidence-label {
  font-size: 0.85rem;
  color: #94a3b8;
  margin-bottom: 0.25rem;
}

.bar-bg {
  height: 8px;
  background: #334155;
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  transition: width 0.3s;
  border-radius: 4px;
}

.probabilities {
  background: rgba(30, 41, 59, 0.5);
  border-radius: 6px;
  padding: 0.75rem;
}

.prob-title {
  font-size: 0.8rem;
  color: #64748b;
  margin-bottom: 0.5rem;
}

.prob-item {
  display: flex;
  justify-content: space-between;
  padding: 0.25rem 0;
  font-size: 0.85rem;
}

.prob-name {
  color: #cbd5e1;
  font-style: italic;
}

.prob-value {
  color: #94a3b8;
}

.no-species {
  color: #64748b;
  font-style: italic;
}

.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}

.metric {
  background: rgba(30, 41, 59, 0.5);
  padding: 0.75rem;
  border-radius: 6px;
}

.metric-label {
  display: block;
  font-size: 0.75rem;
  color: #64748b;
  margin-bottom: 0.25rem;
}

.metric-value {
  font-size: 1rem;
  font-weight: 500;
  color: #f1f5f9;
}

.position-info {
  font-family: monospace;
  font-size: 0.85rem;
  color: #94a3b8;
}

.position-info div {
  margin-bottom: 0.25rem;
}

.root-info {
  background: rgba(139, 69, 19, 0.15);
  border-radius: 8px;
  padding: 0.75rem;
  border: 1px solid rgba(139, 69, 19, 0.3);
}

.root-type {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.root-type-label {
  color: #94a3b8;
  font-size: 0.9rem;
}

.root-type-value {
  color: #cd853f;
  font-weight: 500;
}

.root-description {
  font-size: 0.85rem;
  color: #94a3b8;
  line-height: 1.4;
  margin-bottom: 0.5rem;
}

.root-description p {
  margin: 0;
}

.root-note {
  font-size: 0.75rem;
  color: #64748b;
  font-style: italic;
}
</style>
