<script setup lang="ts">
import { ref, onMounted, watch, shallowRef, computed } from 'vue';
import maplibregl from 'maplibre-gl';
import { Deck } from '@deck.gl/core';
import { PolygonLayer, ScatterplotLayer } from '@deck.gl/layers';
import type { TreeInfo, ProcessingResult } from '../types';
import { type DbTree } from '../services/api';

const props = defineProps<{
  result: ProcessingResult | null;
  dbTrees?: DbTree[];
}>();

const emit = defineEmits<{
  selectTree: [tree: TreeInfo];
  selectDbTree: [tree: DbTree];
}>();

const mapContainer = ref<HTMLDivElement>();
const deckCanvas = ref<HTMLCanvasElement>();
const map = shallowRef<maplibregl.Map>();
const deck = shallowRef<Deck>();

const selectedTreeId = ref<number | null>(null);
const selectedDbTreeId = ref<number | null>(null);
const hoveredTreeId = ref<number | null>(null);

// Compute display trees: prefer result trees, fall back to DB trees
const displayTrees = computed(() => {
  if (props.result?.trees && props.result.trees.length > 0) {
    return props.result.trees;
  }
  return null;
});

const dbTreesForDisplay = computed(() => {
  if (displayTrees.value) return null;
  return props.dbTrees || [];
});

onMounted(() => {
  if (!mapContainer.value || !deckCanvas.value) return;
  
  // Initialize MapLibre with satellite imagery (Esri World Imagery - free, no API key)
  map.value = new maplibregl.Map({
    container: mapContainer.value,
    style: {
      version: 8,
      sources: {
        satellite: {
          type: 'raster',
          tiles: [
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
          ],
          tileSize: 256,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }
      },
      layers: [
        {
          id: 'satellite',
          type: 'raster',
          source: 'satellite',
          minzoom: 0,
          maxzoom: 19
        }
      ]
    },
    center: [4.35, 50.85], // Default to Brussels
    zoom: 15,
    pitch: 0, // 2D view, no tilt
    bearing: 0
  });
  
  // Initialize Deck.gl for tree overlays (2D only)
  deck.value = new Deck({
    canvas: deckCanvas.value,
    initialViewState: {
      longitude: 4.35,
      latitude: 50.85,
      zoom: 15,
      pitch: 0, // 2D view
      bearing: 0
    },
    controller: true,
    layers: [],
    getCursor: ({ isHovering }) => isHovering ? 'pointer' : 'grab',
    onHover: (info) => {
      if (info.object && (info.object.tree_id || info.object.db_id)) {
        hoveredTreeId.value = info.object.tree_id || info.object.db_id;
      } else {
        hoveredTreeId.value = null;
      }
    },
    onClick: (info) => {
      if (info.object) {
        // Handle result tree click
        if (info.object.tree_id && displayTrees.value) {
          const tree = displayTrees.value.find(t => t.tree_id === info.object.tree_id);
          if (tree) {
            selectedTreeId.value = tree.tree_id;
            selectedDbTreeId.value = null;
            emit('selectTree', tree);
          }
        }
        // Handle DB tree click
        else if (info.object.db_id && dbTreesForDisplay.value) {
          const tree = dbTreesForDisplay.value.find(t => t.id === info.object.db_id);
          if (tree) {
            selectedDbTreeId.value = tree.id;
            selectedTreeId.value = null;
            emit('selectDbTree', tree);
          }
        }
      }
    },
    onViewStateChange: ({ viewState }) => {
      // Sync deck.gl view changes back to the map
      if (map.value) {
        map.value.jumpTo({
          center: [viewState.longitude, viewState.latitude],
          zoom: viewState.zoom,
          pitch: 0, // Keep 2D
          bearing: viewState.bearing
        });
      }
    }
  });
  
  // Sync map and deck viewports
  map.value.on('move', () => {
    if (!map.value || !deck.value) return;
    
    const center = map.value.getCenter();
    deck.value.setProps({
      viewState: {
        longitude: center.lng,
        latitude: center.lat,
        zoom: map.value.getZoom(),
        pitch: 0, // Keep 2D
        bearing: map.value.getBearing()
      }
    });
  });
  
  map.value.on('load', () => {
    // If we have DB trees, fly to them
    if (props.dbTrees && props.dbTrees.length > 0) {
      flyToDbTrees();
    }
    updateLayers();
  });
});

// Watch for DB trees changes
watch(() => props.dbTrees, (newDbTrees) => {
  if (newDbTrees && newDbTrees.length > 0 && !props.result?.trees) {
    flyToDbTrees();
  }
  updateLayers();
}, { immediate: true });

function flyToDbTrees() {
  if (!map.value || !props.dbTrees || props.dbTrees.length === 0) return;
  
  // Calculate center of all DB trees
  const lons = props.dbTrees.map(t => t.longitude);
  const lats = props.dbTrees.map(t => t.latitude);
  const centerLon = lons.reduce((a, b) => a + b, 0) / lons.length;
  const centerLat = lats.reduce((a, b) => a + b, 0) / lats.length;
  
  map.value.flyTo({
    center: [centerLon, centerLat],
    zoom: 17,
    pitch: 0, // 2D view
    duration: 2000
  });
}

watch(() => props.result, async (newResult) => {
  if (newResult?.status === 'completed' && newResult.point_cloud?.center) {
    // Fly to the point cloud location
    if (map.value) {
      map.value.flyTo({
        center: [newResult.point_cloud.center[0], newResult.point_cloud.center[1]],
        zoom: 17,
        pitch: 0, // 2D view
        duration: 2000
      });
    }
    
    updateLayers();
  }
}, { immediate: true });

watch([selectedTreeId, selectedDbTreeId, hoveredTreeId], () => {
  updateLayers();
});

function updateLayers() {
  if (!deck.value) return;
  
  const layers = [];
  
  // Tree bounding circles for result trees
  if (displayTrees.value && displayTrees.value.length > 0) {
    // Create circular polygons for each tree using WGS84 coordinates
    const treePolygons = displayTrees.value.map(tree => {
      const metrics = tree.metrics;
      
      // Use direct WGS84 coordinates from backend
      const treeLon = metrics.longitude;
      const treeLat = metrics.latitude;
      
      // Approximate meters per degree for crown polygon
      const metersPerDegreeLat = 111320;
      const metersPerDegreeLon = 111320 * Math.cos(treeLat * Math.PI / 180);
      
      // Create a circular polygon around the tree
      const radiusMeters = (metrics.crown_diameter || 5) / 2;
      const radiusLon = radiusMeters / metersPerDegreeLon;
      const radiusLat = radiusMeters / metersPerDegreeLat;
      
      const points = [];
      const segments = 32; // More segments for smoother circles
      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        points.push([
          treeLon + radiusLon * Math.cos(angle),
          treeLat + radiusLat * Math.sin(angle)
        ]);
      }
      
      return {
        tree_id: tree.tree_id,
        polygon: points,
        center: [treeLon, treeLat],
        height: metrics.height,
        species: tree.species?.species_name || 'Unknown'
      };
    });
    
    // Bounding circle layer
    layers.push(
      new PolygonLayer({
        id: 'tree-bounds',
        data: treePolygons,
        getPolygon: (d: any) => d.polygon,
        getFillColor: (d: any) => {
          if (selectedTreeId.value === d.tree_id) {
            return [255, 215, 0, 120]; // Gold for selected (more opaque)
          }
          if (hoveredTreeId.value === d.tree_id) {
            return [100, 200, 255, 100]; // Light blue for hovered
          }
          return [34, 197, 94, 80]; // Green for normal (more opaque for 2D)
        },
        getLineColor: (d: any) => {
          if (selectedTreeId.value === d.tree_id) {
            return [255, 215, 0, 255]; // Gold border for selected
          }
          return [34, 197, 94, 200]; // Green border
        },
        lineWidthMinPixels: 2,
        pickable: true,
        extruded: false,
        stroked: true,
        filled: true,
        updateTriggers: {
          getFillColor: [selectedTreeId.value, hoveredTreeId.value],
          getLineColor: [selectedTreeId.value]
        }
      })
    );
    
    // Tree center markers
    layers.push(
      new ScatterplotLayer({
        id: 'tree-centers',
        data: treePolygons,
        getPosition: (d: any) => d.center,
        getRadius: 4,
        getFillColor: (d: any) => {
          if (selectedTreeId.value === d.tree_id) {
            return [255, 215, 0, 255]; // Gold for selected
          }
          return [255, 255, 255, 220]; // White center dot
        },
        radiusMinPixels: 5,
        radiusMaxPixels: 8,
        pickable: true,
        updateTriggers: {
          getFillColor: [selectedTreeId.value]
        }
      })
    );
  }
  
  // Database trees (when no result trees)
  if (dbTreesForDisplay.value && dbTreesForDisplay.value.length > 0) {
    const dbTreePolygons = dbTreesForDisplay.value.map(tree => {
      const treeLon = tree.longitude;
      const treeLat = tree.latitude;
      
      const metersPerDegreeLat = 111320;
      const metersPerDegreeLon = 111320 * Math.cos(treeLat * Math.PI / 180);
      
      const radiusMeters = (tree.crown_diameter || 5) / 2;
      const radiusLon = radiusMeters / metersPerDegreeLon;
      const radiusLat = radiusMeters / metersPerDegreeLat;
      
      const points = [];
      const segments = 32; // More segments for smoother circles
      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        points.push([
          treeLon + radiusLon * Math.cos(angle),
          treeLat + radiusLat * Math.sin(angle)
        ]);
      }
      
      return {
        db_id: tree.id,
        tree_id: tree.tree_id,
        polygon: points,
        center: [treeLon, treeLat],
        height: tree.height,
        species: tree.species_name || 'Unknown'
      };
    });
    
    // DB Tree bounding circles
    layers.push(
      new PolygonLayer({
        id: 'db-tree-bounds',
        data: dbTreePolygons,
        getPolygon: (d: any) => d.polygon,
        getFillColor: (d: any) => {
          if (selectedDbTreeId.value === d.db_id) {
            return [255, 215, 0, 120]; // Gold for selected
          }
          if (hoveredTreeId.value === d.db_id) {
            return [100, 200, 255, 100]; // Light blue for hovered
          }
          // Color by species
          if (d.species && d.species !== 'Unknown') {
            return [99, 102, 241, 100]; // Purple for identified (more opaque)
          }
          return [34, 197, 94, 80]; // Green for unidentified (more opaque)
        },
        getLineColor: (d: any) => {
          if (selectedDbTreeId.value === d.db_id) {
            return [255, 215, 0, 255]; // Gold border for selected
          }
          if (d.species && d.species !== 'Unknown') {
            return [99, 102, 241, 220]; // Purple border for identified
          }
          return [34, 197, 94, 200]; // Green border
        },
        lineWidthMinPixels: 2,
        pickable: true,
        extruded: false,
        stroked: true,
        filled: true,
        updateTriggers: {
          getFillColor: [selectedDbTreeId.value, hoveredTreeId.value],
          getLineColor: [selectedDbTreeId.value]
        }
      })
    );
    
    // DB Tree center markers
    layers.push(
      new ScatterplotLayer({
        id: 'db-tree-centers',
        data: dbTreePolygons,
        getPosition: (d: any) => d.center,
        getRadius: 4,
        getFillColor: (d: any) => {
          if (selectedDbTreeId.value === d.db_id) {
            return [255, 215, 0, 255]; // Gold for selected
          }
          if (d.species && d.species !== 'Unknown') {
            return [139, 92, 246, 255]; // Purple for identified
          }
          return [255, 255, 255, 220]; // White for unidentified
        },
        radiusMinPixels: 5,
        radiusMaxPixels: 8,
        pickable: true,
        updateTriggers: {
          getFillColor: [selectedDbTreeId.value]
        }
      })
    );
  }
  
  deck.value.setProps({ layers });
}

// Total tree count
const treeCount = computed(() => {
  if (displayTrees.value) return displayTrees.value.length;
  if (dbTreesForDisplay.value) return dbTreesForDisplay.value.length;
  return 0;
});

const hasAnyTrees = computed(() => treeCount.value > 0);

defineExpose({
  selectTree: (treeId: number | null) => {
    selectedTreeId.value = treeId;
    selectedDbTreeId.value = null;
  },
  selectDbTree: (dbId: number | null) => {
    selectedDbTreeId.value = dbId;
    selectedTreeId.value = null;
  }
});
</script>

<template>
  <div class="map-container">
    <div ref="mapContainer" class="mapbox-map"></div>
    <canvas ref="deckCanvas" class="deck-canvas"></canvas>
    
    <div v-if="!hasAnyTrees" class="overlay-message">
      <div class="message-box">
        <span class="icon">🗺️</span>
        <p>Upload a point cloud to visualize trees</p>
        <p class="sub">Or wait for database trees to load...</p>
      </div>
    </div>
    
    <div v-if="hasAnyTrees" class="tree-count">
      🌳 {{ treeCount }} trees {{ result?.trees ? 'detected' : 'from database' }}
    </div>
  </div>
</template>

<style scoped>
.map-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.mapbox-map {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.deck-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: auto;
}

.overlay-message {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.7);
  pointer-events: none;
}

.message-box {
  text-align: center;
  color: #e2e8f0;
}

.message-box .icon {
  font-size: 3rem;
  display: block;
  margin-bottom: 1rem;
}

.message-box p {
  font-size: 1.1rem;
  color: #94a3b8;
}

.message-box .sub {
  font-size: 0.9rem;
  margin-top: 0.5rem;
  color: #64748b;
}

.tree-count {
  position: absolute;
  bottom: 1rem;
  left: 1rem;
  background: rgba(15, 23, 42, 0.9);
  color: #22c55e;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-weight: 500;
}
</style>
