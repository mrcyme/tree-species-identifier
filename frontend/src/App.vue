<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import type { TreeInfo, ProcessingStatus, ProcessingResult } from './types';
import { uploadPointCloud, pollJobStatus, getAllTreesFromDb, getTreeStats, type DbTree, type TreeStats } from './services/api';
import UploadPanel from './components/UploadPanel.vue';
import MapView from './components/MapView.vue';
import TreePanel from './components/TreePanel.vue';

// Processing state
const isProcessing = ref(false);
const status = ref<ProcessingStatus>();
const result = ref<ProcessingResult | null>(null);
const selectedTree = ref<TreeInfo | null>(null);

// Database trees state
const dbTrees = ref<DbTree[]>([]);
const treeStats = ref<TreeStats | null>(null);
const isLoadingDbTrees = ref(false);
const dbError = ref<string | null>(null);
const selectedDbTree = ref<DbTree | null>(null);

const mapView = ref<InstanceType<typeof MapView>>();

const hasResult = computed(() => result.value?.status === 'completed');
const hasDbTrees = computed(() => dbTrees.value.length > 0);

// Load trees from database on mount
onMounted(async () => {
  await loadDbTrees();
});

async function loadDbTrees() {
  isLoadingDbTrees.value = true;
  dbError.value = null;
  
  try {
    const [treesResponse, statsResponse] = await Promise.all([
      getAllTreesFromDb(1000, 0),
      getTreeStats()
    ]);
    
    dbTrees.value = treesResponse.trees;
    treeStats.value = statsResponse;
    
    console.log(`Loaded ${treesResponse.count} trees from database`);
    
  } catch (error: any) {
    console.warn('Could not load trees from database:', error.message);
    dbError.value = error.message;
  } finally {
    isLoadingDbTrees.value = false;
  }
}

async function handleUpload(file: File, crs: string) {
  isProcessing.value = true;
  selectedTree.value = null;
  selectedDbTree.value = null;
  result.value = null;
  
  try {
    const { job_id } = await uploadPointCloud(file, crs);
    
    const finalResult = await pollJobStatus(job_id, (s) => {
      status.value = s;
    });
    
    result.value = finalResult;
    
    // Refresh database trees after processing
    await loadDbTrees();
    
  } catch (error: any) {
    console.error('Processing failed:', error);
    status.value = {
      job_id: status.value?.job_id || '',
      status: 'failed',
      progress: 0,
      message: error.message || 'Processing failed'
    };
  } finally {
    isProcessing.value = false;
  }
}

function handleSelectTree(tree: TreeInfo) {
  selectedTree.value = tree;
  selectedDbTree.value = null;
  mapView.value?.selectTree(tree.tree_id);
}

function handleSelectDbTree(tree: DbTree) {
  selectedDbTree.value = tree;
  selectedTree.value = null;
  mapView.value?.selectDbTree(tree.id);
}

function handleCloseTreePanel() {
  selectedTree.value = null;
  selectedDbTree.value = null;
  mapView.value?.selectTree(null);
}

// Convert DbTree to TreeInfo for TreePanel compatibility
function dbTreeToTreeInfo(tree: DbTree): TreeInfo {
  return {
    tree_id: tree.tree_id,
    db_id: tree.id,
    metrics: {
      tree_id: tree.tree_id,
      height: tree.height || 0,
      crown_diameter: tree.crown_diameter,
      n_points: tree.n_points || 0,
      x: tree.longitude, // Using lon/lat as x/y
      y: tree.latitude,
      z_min: tree.z_min || 0,
      z_max: tree.z_max || 0,
      bbox_min: tree.bbox_min || [tree.longitude, tree.latitude, tree.z_min || 0],
      bbox_max: tree.bbox_max || [tree.longitude, tree.latitude, tree.z_max || 0],
      longitude: tree.longitude,
      latitude: tree.latitude
    },
    species: tree.species_name ? {
      species_id: tree.species_id || 0,
      species_name: tree.species_name,
      confidence: tree.species_confidence || 0
    } : undefined,
    point_cloud_url: `/api/trees/${tree.id}/pointcloud`
  };
}

// Selected tree for panel (can be from result or DB)
const selectedTreeForPanel = computed(() => {
  if (selectedTree.value) return selectedTree.value;
  if (selectedDbTree.value) return dbTreeToTreeInfo(selectedDbTree.value);
  return null;
});

// Job ID for tree panel (null for DB trees)
const jobIdForPanel = computed(() => {
  if (selectedTree.value && result.value) return result.value.job_id;
  return null;
});
</script>

<template>
  <div class="app">
    <div class="map-area">
      <MapView 
        ref="mapView"
        :result="result"
        :dbTrees="dbTrees"
        @selectTree="handleSelectTree"
        @selectDbTree="handleSelectDbTree"
      />
    </div>
    
    <div class="sidebar left">
      <UploadPanel 
        :isProcessing="isProcessing"
        :status="status"
        @upload="handleUpload"
      />
      
      <!-- Stats panel when we have DB trees but no current processing result -->
      <div v-if="!hasResult && treeStats && hasDbTrees" class="stats-panel">
        <h3>🗃️ Database Trees</h3>
        <div class="stats-grid">
          <div class="stat">
            <span class="stat-value">{{ treeStats.total_trees }}</span>
            <span class="stat-label">Total Trees</span>
          </div>
          <div class="stat">
            <span class="stat-value">{{ treeStats.trees_with_species }}</span>
            <span class="stat-label">With Species</span>
          </div>
        </div>
        <div v-if="Object.keys(treeStats.species_distribution || {}).length > 0" class="species-list">
          <h4>Species Distribution</h4>
          <div class="species-item" v-for="(count, species) in treeStats.species_distribution" :key="species">
            <span class="species-name">{{ species }}</span>
            <span class="species-count">{{ count }}</span>
          </div>
        </div>
      </div>
      
      <!-- Loading indicator -->
      <div v-if="isLoadingDbTrees && !hasResult" class="loading-panel">
        <span>Loading trees from database...</span>
      </div>
    </div>
    
    <div class="sidebar right" v-if="selectedTreeForPanel">
      <TreePanel 
        :tree="selectedTreeForPanel"
        :jobId="jobIdForPanel"
        :useDbEndpoint="!jobIdForPanel"
        @close="handleCloseTreePanel"
      />
    </div>
    
    <div class="logo">
      <span class="logo-icon">🌳</span>
      <span class="logo-text">Tree Species ID</span>
    </div>
  </div>
</template>

<style>
/* MapLibre GL styles - must be first */
@import 'maplibre-gl/dist/maplibre-gl.css';

/* Global styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  width: 100%;
  height: 100%;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
}

#app {
  width: 100%;
  height: 100%;
}
</style>

<style scoped>
.app {
  width: 100%;
  height: 100vh;
  display: flex;
  position: relative;
  overflow: hidden;
}

.map-area {
  flex: 1;
  height: 100%;
}

.sidebar {
  position: absolute;
  top: 1rem;
  max-height: calc(100vh - 2rem);
  display: flex;
  flex-direction: column;
  gap: 1rem;
  z-index: 10;
}

.sidebar.left {
  left: 1rem;
  width: 360px;
}

.sidebar.right {
  right: 1rem;
  width: 500px;
}

.stats-panel {
  background: rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 12px;
  padding: 1rem;
  backdrop-filter: blur(10px);
}

.stats-panel h3 {
  margin-bottom: 0.75rem;
  font-size: 1rem;
  color: #a5b4fc;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.stat {
  background: rgba(99, 102, 241, 0.1);
  padding: 0.75rem;
  border-radius: 8px;
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: #818cf8;
}

.stat-label {
  font-size: 0.75rem;
  color: #94a3b8;
}

.species-list h4 {
  font-size: 0.85rem;
  color: #94a3b8;
  margin-bottom: 0.5rem;
}

.species-item {
  display: flex;
  justify-content: space-between;
  padding: 0.25rem 0;
  font-size: 0.85rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.species-name {
  color: #e2e8f0;
}

.species-count {
  color: #818cf8;
  font-weight: 600;
}

.loading-panel {
  background: rgba(15, 23, 42, 0.95);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
  color: #94a3b8;
}

.logo {
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  background: rgba(15, 23, 42, 0.9);
  padding: 0.5rem 1rem;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  z-index: 10;
}

.logo-icon {
  font-size: 1.25rem;
}

.logo-text {
  font-weight: 600;
  color: #f1f5f9;
}
</style>
