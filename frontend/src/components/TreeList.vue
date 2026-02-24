<script setup lang="ts">
import { computed } from 'vue';
import type { TreeInfo } from '../types';

const props = defineProps<{
  trees: TreeInfo[];
  selectedTreeId: number | null;
}>();

const emit = defineEmits<{
  select: [tree: TreeInfo];
}>();

const sortedTrees = computed(() => {
  return [...props.trees].sort((a, b) => a.tree_id - b.tree_id);
});

function formatSpeciesName(name: string): string {
  return name.replace(/_/g, ' ');
}

function getConfidenceClass(confidence: number): string {
  if (confidence >= 0.7) return 'high';
  if (confidence >= 0.5) return 'medium';
  return 'low';
}
</script>

<template>
  <div class="tree-list">
    <div class="list-header">
      <h3>🌲 Detected Trees ({{ trees.length }})</h3>
    </div>
    
    <div class="list-content">
      <div 
        v-for="tree in sortedTrees" 
        :key="tree.tree_id"
        class="tree-item"
        :class="{ selected: tree.tree_id === selectedTreeId }"
        @click="emit('select', tree)"
      >
        <div class="tree-id">#{{ tree.tree_id }}</div>
        
        <div class="tree-info">
          <div class="species-name" v-if="tree.species">
            {{ formatSpeciesName(tree.species.species_name) }}
          </div>
          <div class="species-name unknown" v-else>Unknown species</div>
          
          <div class="tree-stats">
            <span class="stat">
              <span class="stat-icon">📏</span>
              {{ tree.metrics.height.toFixed(1) }}m
            </span>
            <span class="stat" v-if="tree.species">
              <span 
                class="confidence-dot"
                :class="getConfidenceClass(tree.species.confidence)"
              ></span>
              {{ (tree.species.confidence * 100).toFixed(0) }}%
            </span>
          </div>
        </div>
        
        <div class="tree-action">→</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tree-list {
  background: rgba(15, 23, 42, 0.95);
  border-radius: 12px;
  color: #e2e8f0;
  display: flex;
  flex-direction: column;
  max-height: 100%;
  overflow: hidden;
}

.list-header {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #334155;
}

.list-header h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
}

.list-content {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
}

.tree-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.tree-item:hover {
  background: rgba(59, 130, 246, 0.15);
}

.tree-item.selected {
  background: rgba(34, 197, 94, 0.2);
  border: 1px solid rgba(34, 197, 94, 0.4);
}

.tree-id {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #22c55e, #16a34a);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.8rem;
  flex-shrink: 0;
}

.tree-info {
  flex: 1;
  min-width: 0;
}

.species-name {
  font-weight: 500;
  font-style: italic;
  color: #f1f5f9;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.species-name.unknown {
  color: #64748b;
  font-style: normal;
}

.tree-stats {
  display: flex;
  gap: 0.75rem;
  margin-top: 0.25rem;
}

.stat {
  font-size: 0.8rem;
  color: #94a3b8;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.stat-icon {
  font-size: 0.7rem;
}

.confidence-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.confidence-dot.high {
  background: #22c55e;
}

.confidence-dot.medium {
  background: #f59e0b;
}

.confidence-dot.low {
  background: #ef4444;
}

.tree-action {
  color: #64748b;
  font-size: 1.25rem;
}

.tree-item:hover .tree-action,
.tree-item.selected .tree-action {
  color: #f1f5f9;
}
</style>

