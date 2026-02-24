<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted } from 'vue';
import type { TreePointCloudData, RootMesh, RootCylinder, RootCircle, TreeShape } from '../types';

const props = defineProps<{
  pointCloud: TreePointCloudData;
  rootLod0?: RootMesh | null;  // Full mesh
  rootLod1?: RootMesh | null;  // Convex hull
  rootLod2?: RootCylinder | null;  // Cylinder
  rootLod3?: RootCircle | null;  // Circle
  shape?: TreeShape | null;       // 3D canopy+trunk shape mesh
}>();

const canvas = ref<HTMLCanvasElement>();
let animationId: number;

// Camera state
let rotationX = 0;
let rotationY = 0;
let zoom = 1.0;

// LOD selection
const selectedLod = ref<0 | 1 | 2 | 3>(0);

// Shape visibility toggle
const showShape = ref(false);

// Mouse interaction state
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

const lodLabels = ['LOD 0: Full Mesh', 'LOD 1: Convex Hull', 'LOD 2: Cylinder', 'LOD 3: Circle'];

function cycleLod() {
  selectedLod.value = ((selectedLod.value + 1) % 4) as 0 | 1 | 2 | 3;
}

// Helper to transform a 3D point with full rotation
function transform3D(
  x: number, y: number, z: number,
  cosX: number, sinX: number,
  cosY: number, sinY: number,
  maxRange: number,
  centerX: number, centerY: number, centerZ: number
): { rx: number; ry: number; rz: number } {
  let nx = (x - centerX) / maxRange;
  let ny = (y - centerY) / maxRange;
  let nz = (z - centerZ) / maxRange;
  
  const rx1 = nx * cosY - ny * sinY;
  const ry1 = nx * sinY + ny * cosY;
  const rz1 = nz;
  
  const rx2 = rx1;
  const ry2 = ry1 * cosX - rz1 * sinX;
  const rz2 = ry1 * sinX + rz1 * cosX;
  
  return { rx: rx2, ry: ry2, rz: rz2 };
}

// Get root bounds for any LOD
function getRootBounds(rootScale: number, rootOffsetZ: number): { min: [number, number, number]; max: [number, number, number] } | null {
  const lod = selectedLod.value;
  
  if (lod === 0 && props.rootLod0?.vertices?.length) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const v of props.rootLod0.vertices) {
      const sx = v[0] * rootScale;
      const sy = v[1] * rootScale;
      const sz = v[2] * rootScale + rootOffsetZ;
      minX = Math.min(minX, sx); minY = Math.min(minY, sy); minZ = Math.min(minZ, sz);
      maxX = Math.max(maxX, sx); maxY = Math.max(maxY, sy); maxZ = Math.max(maxZ, sz);
    }
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
  }
  
  if (lod === 1 && props.rootLod1?.vertices?.length) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const v of props.rootLod1.vertices) {
      const sx = v[0] * rootScale;
      const sy = v[1] * rootScale;
      const sz = v[2] * rootScale + rootOffsetZ;
      minX = Math.min(minX, sx); minY = Math.min(minY, sy); minZ = Math.min(minZ, sz);
      maxX = Math.max(maxX, sx); maxY = Math.max(maxY, sy); maxZ = Math.max(maxZ, sz);
    }
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
  }
  
  if (lod === 2 && props.rootLod2) {
    const cyl = props.rootLod2;
    const r = cyl.radius * rootScale;
    const h = cyl.height * rootScale;
    const cx = cyl.center[0] * rootScale;
    const cy = cyl.center[1] * rootScale;
    const cz = cyl.center[2] * rootScale + rootOffsetZ;
    return {
      min: [cx - r, cy - r, cz - h/2],
      max: [cx + r, cy + r, cz + h/2]
    };
  }
  
  if (lod === 3 && props.rootLod3) {
    const circle = props.rootLod3;
    const r = circle.radius * rootScale;
    const cx = circle.center[0] * rootScale;
    const cy = circle.center[1] * rootScale;
    return {
      min: [cx - r, cy - r, rootOffsetZ - 0.1],
      max: [cx + r, cy + r, rootOffsetZ]
    };
  }
  
  return null;
}

function render() {
  if (!canvas.value || !props.pointCloud) return;
  
  const ctx = canvas.value.getContext('2d');
  if (!ctx) return;
  
  const width = canvas.value.width;
  const height = canvas.value.height;
  
  // Clear canvas
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#e8f4ea');
  gradient.addColorStop(0.5, '#d4e8d8');
  gradient.addColorStop(1, '#c5dcc9');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  
  const positions = props.pointCloud.positions;
  const colors = props.pointCloud.colors;
  const center = props.pointCloud.center;
  
  if (positions.length === 0) return;
  
  // Get tree bounding box
  const treeBbox = {
    min: [...(props.pointCloud.bbox_min || [0, 0, 0])] as [number, number, number],
    max: [...(props.pointCloud.bbox_max || [1, 1, 1])] as [number, number, number]
  };
  
  const treeHeight = treeBbox.max[2] - treeBbox.min[2];
  const treeWidth = Math.max(
    treeBbox.max[0] - treeBbox.min[0],
    treeBbox.max[1] - treeBbox.min[1]
  );
  
  // Calculate root scaling
  let rootScale = 1;
  let rootOffsetZ = treeBbox.min[2];
  
  // Determine scale from LOD0 if available, otherwise estimate
  const refMesh = props.rootLod0 || props.rootLod1;
  if (refMesh && refMesh.vertices?.length > 0) {
    let rootMinZ = Infinity, rootMaxZ = -Infinity, rootMaxXY = 0;
    for (const v of refMesh.vertices) {
      rootMinZ = Math.min(rootMinZ, v[2]);
      rootMaxZ = Math.max(rootMaxZ, v[2]);
      rootMaxXY = Math.max(rootMaxXY, Math.abs(v[0]), Math.abs(v[1]));
    }
    const rootDepth = rootMaxZ - rootMinZ;
    const rootSpread = rootMaxXY * 2;
    const targetRootDepth = treeHeight * 0.4;
    const targetRootSpread = Math.max(treeWidth * 0.6, treeHeight * 0.3);
    const depthScale = rootDepth > 0 ? targetRootDepth / rootDepth : 1;
    const spreadScale = rootSpread > 0 ? targetRootSpread / rootSpread : 1;
    rootScale = Math.max(depthScale, spreadScale);
  } else if (props.rootLod2) {
    // Scale from cylinder
    const targetRootDepth = treeHeight * 0.4;
    rootScale = props.rootLod2.height > 0 ? targetRootDepth / props.rootLod2.height : 1;
  } else if (props.rootLod3) {
    // Scale from circle
    const targetRootSpread = Math.max(treeWidth * 0.6, treeHeight * 0.3);
    rootScale = props.rootLod3.radius > 0 ? targetRootSpread / (props.rootLod3.radius * 2) : 1;
  }
  
  // Calculate combined bounding box
  const combinedBbox = {
    min: [...treeBbox.min] as [number, number, number],
    max: [...treeBbox.max] as [number, number, number]
  };
  
  const rootBounds = getRootBounds(rootScale, rootOffsetZ);
  if (rootBounds) {
    combinedBbox.min[0] = Math.min(combinedBbox.min[0], rootBounds.min[0]);
    combinedBbox.min[1] = Math.min(combinedBbox.min[1], rootBounds.min[1]);
    combinedBbox.min[2] = Math.min(combinedBbox.min[2], rootBounds.min[2]);
    combinedBbox.max[0] = Math.max(combinedBbox.max[0], rootBounds.max[0]);
    combinedBbox.max[1] = Math.max(combinedBbox.max[1], rootBounds.max[1]);
    combinedBbox.max[2] = Math.max(combinedBbox.max[2], rootBounds.max[2]);
  }
  
  const range = [
    combinedBbox.max[0] - combinedBbox.min[0],
    combinedBbox.max[1] - combinedBbox.min[1],
    combinedBbox.max[2] - combinedBbox.min[2]
  ];
  const maxRange = Math.max(...range) || 1;
  
  const combinedCenterX = (combinedBbox.min[0] + combinedBbox.max[0]) / 2;
  const combinedCenterY = (combinedBbox.min[1] + combinedBbox.max[1]) / 2;
  const combinedCenterZ = (combinedBbox.min[2] + combinedBbox.max[2]) / 2;
  
  const cosX = Math.cos(rotationX);
  const sinX = Math.sin(rotationX);
  const cosY = Math.cos(rotationY);
  const sinY = Math.sin(rotationY);
  
  const baseScale = Math.min(width, height) * 0.35;
  const scale = baseScale * zoom;
  
  // Draw roots based on selected LOD
  const lod = selectedLod.value;
  
  if (lod === 0 && props.rootLod0?.vertices?.length && props.rootLod0?.faces?.length) {
    drawMesh(ctx, props.rootLod0, width, height, scale, rootScale, rootOffsetZ, 
             cosX, sinX, cosY, sinY, maxRange, combinedCenterX, combinedCenterY, combinedCenterZ,
             'rgba(139, 90, 43, ', 'rgba(101, 67, 33, ');
  }
  
  if (lod === 1 && props.rootLod1?.vertices?.length && props.rootLod1?.faces?.length) {
    drawMesh(ctx, props.rootLod1, width, height, scale, rootScale, rootOffsetZ,
             cosX, sinX, cosY, sinY, maxRange, combinedCenterX, combinedCenterY, combinedCenterZ,
             'rgba(46, 139, 87, ', 'rgba(34, 100, 60, ');
  }
  
  if (lod === 2 && props.rootLod2) {
    drawCylinder(ctx, props.rootLod2, width, height, scale, rootScale, rootOffsetZ,
                 cosX, sinX, cosY, sinY, maxRange, combinedCenterX, combinedCenterY, combinedCenterZ);
  }
  
  if (lod === 3 && props.rootLod3) {
    drawCircle(ctx, props.rootLod3, width, height, scale, rootScale, rootOffsetZ,
               cosX, sinX, cosY, sinY, maxRange, combinedCenterX, combinedCenterY, combinedCenterZ);
  }
  
  // Draw tree shape mesh (trunk + canopy) if toggle is on
  if (showShape.value && props.shape?.vertices?.length && props.shape?.faces?.length) {
    drawShapeMesh(ctx, props.shape, width, height, scale,
                  cosX, sinX, cosY, sinY, maxRange,
                  combinedCenterX, combinedCenterY, combinedCenterZ);
  }
  
  // Draw tree point cloud
  const projected: Array<{ x: number; y: number; z: number; color: [number, number, number] }> = [];
  const maxPoints = 8000;
  const step = Math.max(1, Math.floor(positions.length / maxPoints));
  
  for (let i = 0; i < positions.length; i += step) {
    const pos = positions[i];
    if (!pos) continue;
    
    const { rx, ry, rz } = transform3D(
      pos[0] ?? 0, pos[1] ?? 0, pos[2] ?? 0,
      cosX, sinX, cosY, sinY, maxRange,
      combinedCenterX, combinedCenterY, combinedCenterZ
    );
    
    projected.push({
      x: width / 2 + rx * scale,
      y: height / 2 - rz * scale,
      z: ry,
      color: colors[i] as [number, number, number]
    });
  }
  
  projected.sort((a, b) => a.z - b.z);
  
  const pointSize = Math.max(1.5, 2 * zoom);
  for (const p of projected) {
    const brightness = 0.5 + (p.z + 0.5) * 0.5;
    ctx.fillStyle = `rgb(${Math.floor(p.color[0] * brightness)}, ${Math.floor(p.color[1] * brightness)}, ${Math.floor(p.color[2] * brightness)})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, pointSize, 0, Math.PI * 2);
    ctx.fill();
  }
  
  // Draw controls hint
  ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText('🖱️ Drag to rotate • Scroll to zoom', 10, height - 10);
  
  animationId = requestAnimationFrame(render);
}

function drawMesh(
  ctx: CanvasRenderingContext2D,
  mesh: RootMesh,
  width: number, height: number, scale: number,
  rootScale: number, rootOffsetZ: number,
  cosX: number, sinX: number, cosY: number, sinY: number,
  maxRange: number, centerX: number, centerY: number, centerZ: number,
  fillColor: string, strokeColor: string
) {
  ctx.globalAlpha = 0.85;
  
  const projectedVertices: Array<{ x: number; y: number; z: number }> = [];
  
  for (const v of mesh.vertices) {
    const sx = v[0] * rootScale;
    const sy = v[1] * rootScale;
    const sz = v[2] * rootScale + rootOffsetZ;
    
    const { rx, ry, rz } = transform3D(sx, sy, sz, cosX, sinX, cosY, sinY, maxRange, centerX, centerY, centerZ);
    projectedVertices.push({ x: width / 2 + rx * scale, y: height / 2 - rz * scale, z: ry });
  }
  
  const facesWithDepth: Array<{ indices: number[]; avgZ: number }> = [];
  for (const face of mesh.faces) {
    const avgZ = ((projectedVertices[face[0]]?.z ?? 0) + 
                  (projectedVertices[face[1]]?.z ?? 0) + 
                  (projectedVertices[face[2]]?.z ?? 0)) / 3;
    facesWithDepth.push({ indices: face as number[], avgZ });
  }
  facesWithDepth.sort((a, b) => a.avgZ - b.avgZ);
  
  for (const { indices } of facesWithDepth) {
    const v0 = projectedVertices[indices[0]];
    const v1 = projectedVertices[indices[1]];
    const v2 = projectedVertices[indices[2]];
    if (!v0 || !v1 || !v2) continue;
    
    const avgZ = (v0.z + v1.z + v2.z) / 3;
    const brightness = 0.5 + (avgZ + 0.5) * 0.5;
    
    ctx.fillStyle = fillColor + (0.6 * brightness) + ')';
    ctx.beginPath();
    ctx.moveTo(v0.x, v0.y);
    ctx.lineTo(v1.x, v1.y);
    ctx.lineTo(v2.x, v2.y);
    ctx.closePath();
    ctx.fill();
    
    ctx.strokeStyle = strokeColor + (0.7 * brightness) + ')';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  
  ctx.globalAlpha = 1.0;
}

function drawShapeMesh(
  ctx: CanvasRenderingContext2D,
  mesh: TreeShape,
  width: number, height: number, scale: number,
  cosX: number, sinX: number, cosY: number, sinY: number,
  maxRange: number, centerX: number, centerY: number, centerZ: number
) {
  ctx.globalAlpha = 0.8;

  const projectedVertices: Array<{ x: number; y: number; z: number }> = [];

  for (const v of mesh.vertices) {
    const { rx, ry, rz } = transform3D(
      v[0], v[1], v[2],
      cosX, sinX, cosY, sinY,
      maxRange, centerX, centerY, centerZ
    );
    projectedVertices.push({ x: width / 2 + rx * scale, y: height / 2 - rz * scale, z: ry });
  }

  // Sort faces back-to-front for correct painter's algorithm rendering
  const facesWithDepth = mesh.faces.map(face => ({
    indices: face as number[],
    avgZ: ((projectedVertices[face[0]]?.z ?? 0) +
           (projectedVertices[face[1]]?.z ?? 0) +
           (projectedVertices[face[2]]?.z ?? 0)) / 3
  }));
  facesWithDepth.sort((a, b) => a.avgZ - b.avgZ);

  for (const { indices, avgZ } of facesWithDepth) {
    const v0 = projectedVertices[indices[0]];
    const v1 = projectedVertices[indices[1]];
    const v2 = projectedVertices[indices[2]];
    if (!v0 || !v1 || !v2) continue;

    const brightness = 0.55 + (avgZ + 0.5) * 0.45;
    const r = Math.floor(34 * brightness);
    const g = Math.floor(139 * brightness);
    const b = Math.floor(34 * brightness);

    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.55)`;
    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
    ctx.lineWidth = 0.8;

    ctx.beginPath();
    ctx.moveTo(v0.x, v0.y);
    ctx.lineTo(v1.x, v1.y);
    ctx.lineTo(v2.x, v2.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  ctx.globalAlpha = 1.0;
}

function drawCylinder(
  ctx: CanvasRenderingContext2D,
  cyl: RootCylinder,
  width: number, height: number, scale: number,
  rootScale: number, rootOffsetZ: number,
  cosX: number, sinX: number, cosY: number, sinY: number,
  maxRange: number, centerX: number, centerY: number, centerZ: number
) {
  const cx = cyl.center[0] * rootScale;
  const cy = cyl.center[1] * rootScale;
  const cz = cyl.center[2] * rootScale + rootOffsetZ;
  const r = cyl.radius * rootScale;
  const h = cyl.height * rootScale;
  
  // Generate cylinder vertices
  const segments = 16;
  const topVertices: Array<{ x: number; y: number; z: number }> = [];
  const bottomVertices: Array<{ x: number; y: number; z: number }> = [];
  
  for (let i = 0; i < segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    const vx = cx + Math.cos(angle) * r;
    const vy = cy + Math.sin(angle) * r;
    
    const { rx: rx1, ry: ry1, rz: rz1 } = transform3D(vx, vy, cz + h/2, cosX, sinX, cosY, sinY, maxRange, centerX, centerY, centerZ);
    const { rx: rx2, ry: ry2, rz: rz2 } = transform3D(vx, vy, cz - h/2, cosX, sinX, cosY, sinY, maxRange, centerX, centerY, centerZ);
    
    topVertices.push({ x: width / 2 + rx1 * scale, y: height / 2 - rz1 * scale, z: ry1 });
    bottomVertices.push({ x: width / 2 + rx2 * scale, y: height / 2 - rz2 * scale, z: ry2 });
  }
  
  ctx.globalAlpha = 0.6;
  
  // Draw sides
  ctx.fillStyle = 'rgba(70, 130, 180, 0.5)';
  ctx.strokeStyle = 'rgba(50, 90, 130, 0.8)';
  ctx.lineWidth = 1.5;
  
  for (let i = 0; i < segments; i++) {
    const next = (i + 1) % segments;
    ctx.beginPath();
    ctx.moveTo(topVertices[i].x, topVertices[i].y);
    ctx.lineTo(topVertices[next].x, topVertices[next].y);
    ctx.lineTo(bottomVertices[next].x, bottomVertices[next].y);
    ctx.lineTo(bottomVertices[i].x, bottomVertices[i].y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  
  // Draw top cap
  ctx.beginPath();
  ctx.moveTo(topVertices[0].x, topVertices[0].y);
  for (let i = 1; i < segments; i++) {
    ctx.lineTo(topVertices[i].x, topVertices[i].y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(100, 160, 210, 0.4)';
  ctx.fill();
  ctx.stroke();
  
  // Draw bottom cap
  ctx.beginPath();
  ctx.moveTo(bottomVertices[0].x, bottomVertices[0].y);
  for (let i = 1; i < segments; i++) {
    ctx.lineTo(bottomVertices[i].x, bottomVertices[i].y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(50, 100, 150, 0.4)';
  ctx.fill();
  ctx.stroke();
  
  ctx.globalAlpha = 1.0;
}

function drawCircle(
  ctx: CanvasRenderingContext2D,
  circle: RootCircle,
  width: number, height: number, scale: number,
  rootScale: number, rootOffsetZ: number,
  cosX: number, sinX: number, cosY: number, sinY: number,
  maxRange: number, centerX: number, centerY: number, centerZ: number
) {
  const cx = circle.center[0] * rootScale;
  const cy = circle.center[1] * rootScale;
  const r = circle.radius * rootScale;
  
  // Generate circle vertices
  const segments = 32;
  const vertices: Array<{ x: number; y: number; z: number }> = [];
  
  for (let i = 0; i < segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    const vx = cx + Math.cos(angle) * r;
    const vy = cy + Math.sin(angle) * r;
    
    const { rx, ry, rz } = transform3D(vx, vy, rootOffsetZ, cosX, sinX, cosY, sinY, maxRange, centerX, centerY, centerZ);
    vertices.push({ x: width / 2 + rx * scale, y: height / 2 - rz * scale, z: ry });
  }
  
  ctx.globalAlpha = 0.7;
  
  // Fill circle
  ctx.beginPath();
  ctx.moveTo(vertices[0].x, vertices[0].y);
  for (let i = 1; i < segments; i++) {
    ctx.lineTo(vertices[i].x, vertices[i].y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 140, 0, 0.5)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(200, 100, 0, 0.9)';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  ctx.globalAlpha = 1.0;
}

// Mouse event handlers
function onMouseDown(e: MouseEvent) {
  isDragging = true;
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
}

function onMouseMove(e: MouseEvent) {
  if (!isDragging) return;
  const deltaX = e.clientX - lastMouseX;
  const deltaY = e.clientY - lastMouseY;
  rotationY += deltaX * 0.01;
  rotationX += deltaY * 0.01;
  rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
}

function onMouseUp() { isDragging = false; }
function onMouseLeave() { isDragging = false; }

function onWheel(e: WheelEvent) {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  zoom *= delta;
  zoom = Math.max(0.3, Math.min(5, zoom));
}

function onTouchStart(e: TouchEvent) {
  if (e.touches.length === 1) {
    isDragging = true;
    lastMouseX = e.touches[0].clientX;
    lastMouseY = e.touches[0].clientY;
  }
}

function onTouchMove(e: TouchEvent) {
  if (!isDragging || e.touches.length !== 1) return;
  e.preventDefault();
  const deltaX = e.touches[0].clientX - lastMouseX;
  const deltaY = e.touches[0].clientY - lastMouseY;
  rotationY += deltaX * 0.01;
  rotationX += deltaY * 0.01;
  rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));
  lastMouseX = e.touches[0].clientX;
  lastMouseY = e.touches[0].clientY;
}

function onTouchEnd() { isDragging = false; }

function resetView() {
  rotationX = 0;
  rotationY = 0;
  zoom = 1.0;
}

onMounted(() => {
  if (canvas.value) {
    canvas.value.width = canvas.value.offsetWidth * window.devicePixelRatio;
    canvas.value.height = canvas.value.offsetHeight * window.devicePixelRatio;
    
    canvas.value.addEventListener('mousedown', onMouseDown);
    canvas.value.addEventListener('mousemove', onMouseMove);
    canvas.value.addEventListener('mouseup', onMouseUp);
    canvas.value.addEventListener('mouseleave', onMouseLeave);
    canvas.value.addEventListener('wheel', onWheel, { passive: false });
    canvas.value.addEventListener('touchstart', onTouchStart);
    canvas.value.addEventListener('touchmove', onTouchMove, { passive: false });
    canvas.value.addEventListener('touchend', onTouchEnd);
    
    render();
  }
});

watch([() => props.pointCloud, () => props.rootLod0], () => {
  resetView();
});

onUnmounted(() => {
  if (animationId) cancelAnimationFrame(animationId);
  
  if (canvas.value) {
    canvas.value.removeEventListener('mousedown', onMouseDown);
    canvas.value.removeEventListener('mousemove', onMouseMove);
    canvas.value.removeEventListener('mouseup', onMouseUp);
    canvas.value.removeEventListener('mouseleave', onMouseLeave);
    canvas.value.removeEventListener('wheel', onWheel);
    canvas.value.removeEventListener('touchstart', onTouchStart);
    canvas.value.removeEventListener('touchmove', onTouchMove);
    canvas.value.removeEventListener('touchend', onTouchEnd);
  }
});
</script>

<template>
  <div class="viewer-wrapper">
    <canvas ref="canvas" class="tree-viewer"></canvas>
    <div class="controls">
      <button 
        class="shape-btn" 
        :class="{ active: showShape }"
        @click="showShape = !showShape" 
        :title="showShape ? 'Hide 3D shape' : 'Show 3D shape'"
        :disabled="!shape?.vertices?.length"
      >
        <span>🌳</span>
      </button>
      <button class="lod-btn" @click="cycleLod" :title="lodLabels[selectedLod]">
        <span class="lod-icon">🌱</span>
        <span class="lod-text">{{ selectedLod }}</span>
      </button>
      <button class="reset-btn" @click="resetView" title="Reset view">↺</button>
    </div>
    <div class="lod-label">
      {{ lodLabels[selectedLod] }}
      <span v-if="showShape" class="shape-badge">+ 3D Shape</span>
    </div>
  </div>
</template>

<style scoped>
.viewer-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}

.tree-viewer {
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
}

.tree-viewer:active {
  cursor: grabbing;
}

.controls {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  gap: 6px;
}

.reset-btn, .lod-btn {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: none;
  background: rgba(255, 255, 255, 0.9);
  color: #334155;
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}

.shape-btn {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: 2px solid transparent;
  background: rgba(255, 255, 255, 0.9);
  color: #334155;
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}

.shape-btn.active {
  background: rgba(34, 197, 94, 0.25);
  border-color: #22c55e;
  box-shadow: 0 2px 8px rgba(34, 197, 94, 0.4);
}

.shape-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.lod-btn {
  width: auto;
  padding: 0 10px;
  gap: 4px;
}

.lod-icon {
  font-size: 14px;
}

.lod-text {
  font-weight: 600;
  font-size: 13px;
}

.reset-btn:hover, .lod-btn:hover {
  background: rgba(255, 255, 255, 1);
  transform: scale(1.05);
}

.lod-label {
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 6px;
}

.shape-badge {
  background: rgba(34, 197, 94, 0.35);
  color: #86efac;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  border: 1px solid rgba(34, 197, 94, 0.5);
}
</style>
