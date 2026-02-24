<script setup lang="ts">
import { ref, computed } from 'vue';
import type { ProcessingStatus } from '../types';

const props = defineProps<{
  isProcessing: boolean;
  status?: ProcessingStatus;
}>();

const emit = defineEmits<{
  upload: [file: File, crs: string];
}>();

const file = ref<File | null>(null);
const crs = ref('31370');
const isDragging = ref(false);

const fileInput = ref<HTMLInputElement>();

const statusColor = computed(() => {
  if (!props.status) return '#666';
  switch (props.status.status) {
    case 'completed': return '#22c55e';
    case 'failed': return '#ef4444';
    case 'pending': return '#f59e0b';
    default: return '#3b82f6';
  }
});

function handleDrop(e: DragEvent) {
  isDragging.value = false;
  const files = e.dataTransfer?.files;
  if (files && files.length > 0 && files[0]) {
    handleFile(files[0]);
  }
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files && input.files.length > 0 && input.files[0]) {
    handleFile(input.files[0]);
  }
}

function handleFile(f: File) {
  if (!f.name.toLowerCase().endsWith('.las') && !f.name.toLowerCase().endsWith('.laz')) {
    alert('Please upload a LAS or LAZ file');
    return;
  }
  file.value = f;
}

function submit() {
  if (file.value) {
    emit('upload', file.value, crs.value);
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
</script>

<template>
  <div class="upload-panel">
    <h2>🌳 Tree Species Identifier</h2>
    
    <div 
      class="dropzone"
      :class="{ dragging: isDragging, 'has-file': file }"
      @dragover.prevent="isDragging = true"
      @dragleave="isDragging = false"
      @drop.prevent="handleDrop"
      @click="fileInput?.click()"
    >
      <input 
        ref="fileInput"
        type="file" 
        accept=".las,.laz"
        @change="handleFileSelect"
        hidden
      />
      
      <div v-if="!file" class="dropzone-content">
        <span class="icon">📁</span>
        <p>Drop LAS/LAZ file here</p>
        <p class="hint">or click to browse</p>
      </div>
      
      <div v-else class="file-info">
        <span class="icon">📄</span>
        <p class="filename">{{ file.name }}</p>
        <p class="filesize">{{ formatBytes(file.size) }}</p>
      </div>
    </div>
    
    <div class="form-group">
      <label for="crs">Coordinate System (EPSG)</label>
      <select id="crs" v-model="crs" :disabled="isProcessing">
        <option value="31370">EPSG:31370 (Belgian Lambert 72)</option>
        <option value="4326">EPSG:4326 (WGS84)</option>
        <option value="25833">EPSG:25833 (ETRS89 / UTM 33N)</option>
        <option value="32631">EPSG:32631 (WGS84 / UTM 31N)</option>
      </select>
    </div>
    
    <button 
      class="submit-btn"
      :disabled="!file || isProcessing"
      @click="submit"
    >
      <span v-if="!isProcessing">🔍 Analyze Trees</span>
      <span v-else>⏳ Processing...</span>
    </button>
    
    <div v-if="status" class="status-panel">
      <div class="status-header">
        <span 
          class="status-dot" 
          :style="{ backgroundColor: statusColor }"
        ></span>
        <span class="status-text">{{ status.status }}</span>
      </div>
      
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          :style="{ width: status.progress + '%' }"
        ></div>
      </div>
      
      <p class="status-message">{{ status.message }}</p>
    </div>
  </div>
</template>

<style scoped>
.upload-panel {
  padding: 1.5rem;
  background: rgba(15, 23, 42, 0.95);
  border-radius: 12px;
  color: #e2e8f0;
}

h2 {
  margin: 0 0 1.5rem;
  font-size: 1.25rem;
  font-weight: 600;
  color: #f1f5f9;
}

.dropzone {
  border: 2px dashed #475569;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 1rem;
}

.dropzone:hover,
.dropzone.dragging {
  border-color: #22c55e;
  background: rgba(34, 197, 94, 0.1);
}

.dropzone.has-file {
  border-color: #3b82f6;
  background: rgba(59, 130, 246, 0.1);
}

.dropzone-content .icon,
.file-info .icon {
  font-size: 2.5rem;
  display: block;
  margin-bottom: 0.5rem;
}

.dropzone-content p {
  margin: 0.25rem 0;
  color: #94a3b8;
}

.dropzone-content .hint {
  font-size: 0.85rem;
  color: #64748b;
}

.file-info .filename {
  font-weight: 500;
  color: #f1f5f9;
  word-break: break-all;
}

.file-info .filesize {
  color: #64748b;
  font-size: 0.85rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
  color: #94a3b8;
}

.form-group select {
  width: 100%;
  padding: 0.5rem;
  border-radius: 6px;
  border: 1px solid #475569;
  background: #1e293b;
  color: #e2e8f0;
  font-size: 0.9rem;
}

.submit-btn {
  width: 100%;
  padding: 0.75rem;
  border-radius: 8px;
  border: none;
  background: linear-gradient(135deg, #22c55e, #16a34a);
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.status-panel {
  margin-top: 1.5rem;
  padding: 1rem;
  background: rgba(30, 41, 59, 0.8);
  border-radius: 8px;
}

.status-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-text {
  font-weight: 500;
  text-transform: capitalize;
}

.progress-bar {
  height: 6px;
  background: #334155;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #3b82f6);
  transition: width 0.3s;
}

.status-message {
  margin: 0;
  font-size: 0.85rem;
  color: #94a3b8;
}
</style>

