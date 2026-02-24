#!/usr/bin/env Rscript
# Tree Segmentation Script
# Segments individual trees from a LiDAR point cloud and outputs a LAS file with TreeID
# 
# Usage: Rscript segment_trees.R <input_las> <output_las> [crs_code] [--subset x1,y1,x2,y2]
#
# Arguments:
#   input_las   - Path to input LAS/LAZ file
#   output_las  - Path to output LAS file with TreeID column
#   crs_code    - EPSG code (default: 31370 for Belgian Lambert 72)
#   --subset    - Optional: Extract only a subset (xmin,ymin,xmax,ymax)

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
  library(data.table)
})

# Use all available CPU cores for lidR processing to speed up large files
set_lidr_threads(0)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript segment_trees.R <input_las> <output_las> [crs_code] [--subset x1,y1,x2,y2]\n")
  cat("\nExample:\n")
  cat("  Rscript segment_trees.R input.las output_segmented.las 31370\n")
  cat("  Rscript segment_trees.R input.las output_segmented.las 31370 --subset 149500,170500,149700,170700\n")
  quit(status = 1)
}

input_las_path <- args[1]
output_las_path <- args[2]
crs_code <- if (length(args) >= 3 && !startsWith(args[3], "--")) args[3] else "31370"
tile_crs <- paste0("epsg:", crs_code)

# Check for subset argument
subset_bounds <- NULL
for (i in seq_along(args)) {
  if (args[i] == "--subset" && i < length(args)) {
    bounds_str <- args[i + 1]
    bounds <- as.numeric(strsplit(bounds_str, ",")[[1]])
    if (length(bounds) == 4) {
      subset_bounds <- bounds
      names(subset_bounds) <- c("xmin", "ymin", "xmax", "ymax")
    }
  }
}

cat("\n========================================\n")
cat("TREE SEGMENTATION\n")
cat("========================================\n")
cat("Input:", input_las_path, "\n")
cat("Output:", output_las_path, "\n")
cat("CRS:", tile_crs, "\n")
if (!is.null(subset_bounds)) {
  cat("Subset:", paste(subset_bounds, collapse=", "), "\n")
}
cat("\n")

# Workflow parameters
cell_size <- 0.5
min_h_trees <- 2
max_h_trees <- 50
sigma_chm <- 0.5
vwf_fun <- function(x) { 2.51503 + 0.00901 * x^2 }

# =============================================================================
# STEP 1: Load data
# =============================================================================
cat("=== 1. LOADING DATA ===\n")

# Build filter string for subset
filter_str <- "-drop_class 7"  # Drop noise/outliers
if (!is.null(subset_bounds)) {
  filter_str <- paste0(filter_str, 
                       " -keep_xy ", subset_bounds["xmin"], " ", subset_bounds["ymin"],
                       " ", subset_bounds["xmax"], " ", subset_bounds["ymax"])
}

# Check if data has pre-classified vegetation (class 5)
las_header <- readLASheader(input_las_path)
cat("  Total points in file:", las_header@PHB[["Number of point records"]], "\n")

# Read a sample to check classifications
sample_filter <- paste0(filter_str, " -thin_random 0.001")
las_sample <- readLAS(input_las_path, filter = sample_filter)
available_classes <- sort(unique(las_sample@data$Classification))
cat("  Classification codes found:", paste(available_classes, collapse=", "), "\n")

has_vegetation_class <- 5 %in% available_classes
has_ground_class <- 2 %in% available_classes

if (!has_ground_class) {
  cat("  Ground points (class 2) not found. Auto-classifying ground using CSF...\n")
  
  # Read all valid points
  las_full <- readLAS(input_las_path, filter = filter_str)
  crs(las_full) <- tile_crs
  
  # Classify ground
  las_full <- classify_ground(las_full, algorithm = csf())
  
  # Extract ground points
  lidar_ground <- filter_poi(las_full, Classification == 2L)
  cat("  Ground points classified:", nrow(lidar_ground@data), "\n")
  
  # Extract vegetation points
  if (has_vegetation_class) {
    lidar_veg <- filter_poi(las_full, Classification == 5L)
  } else {
    lidar_veg <- filter_poi(las_full, Classification != 2L)
  }
  rm(las_full)
  gc(verbose = FALSE)
  
} else {
  # Read ground points for DTM
  cat("  Loading ground points (class 2)...\n")
  ground_filter <- paste0(filter_str, " -keep_class 2")
  lidar_ground <- readLAS(input_las_path, filter = ground_filter)
  crs(lidar_ground) <- tile_crs
  cat("  Ground points loaded:", nrow(lidar_ground@data), "\n")
  
  # Read vegetation/surface points
  if (has_vegetation_class) {
    cat("  Loading high vegetation points (class 5)...\n")
    veg_filter <- paste0(filter_str, " -keep_class 5")
  } else {
    cat("  No vegetation classification found. Loading all non-ground points...\n")
    veg_filter <- paste0(filter_str, " -drop_class 2")
  }
  
  lidar_veg <- readLAS(input_las_path, filter = veg_filter)
  crs(lidar_veg) <- tile_crs
  cat("  Vegetation/surface points loaded:", nrow(lidar_veg@data), "\n")
}

if (nrow(lidar_veg@data) == 0) {
  stop("No vegetation points found!")
}

# Get tile extent
tile_ext <- ext(lidar_veg)
cat("  Tile extent: X[", xmin(tile_ext), "-", xmax(tile_ext), "] Y[", 
    ymin(tile_ext), "-", ymax(tile_ext), "]\n")

# =============================================================================
# STEP 2: Create DTM and normalize heights
# =============================================================================
cat("\n=== 2. TERRAIN NORMALIZATION ===\n")

cat("  Creating DTM...\n")
dtm_grid <- rasterize_terrain(lidar_ground, resolution = cell_size, algorithm = tin())
dtm_grid <- ifel(is.na(dtm_grid), 0, dtm_grid)

rm(lidar_ground)
gc(verbose = FALSE)

cat("  Normalizing vegetation heights...\n")
# Preserve original orthometric elevation (TAW) before normalization.
lidar_veg@data[, Z_TAW := Z]
lidar_veg <- normalize_height(lidar_veg, dtm_grid)
lidar_veg@data[Z < 0, Z := 0]

# Filter by height
original_count <- nrow(lidar_veg@data)
lidar_veg <- filter_poi(lidar_veg, Z >= min_h_trees & Z <= max_h_trees)
cat("  Points after height filtering:", nrow(lidar_veg@data), 
    "(removed", original_count - nrow(lidar_veg@data), ")\n")

if (nrow(lidar_veg@data) < 100) {
  stop("Too few vegetation points after filtering!")
}

# =============================================================================
# STEP 3: Create CHM and detect tree tops
# =============================================================================
cat("\n=== 3. CANOPY HEIGHT MODEL ===\n")

cat("  Creating CHM...\n")
template <- rast(extent = tile_ext, crs = tile_crs, resolution = cell_size)
chm_grid <- rasterize_canopy(lidar_veg, res = cell_size, 
                              algorithm = p2r(subcircle = 0.3, na.fill = NULL))
chm_grid <- ifel(chm_grid < min_h_trees, NA, chm_grid)

cat("  CHM cells with data:", sum(!is.na(values(chm_grid))), "\n")

# Gaussian smoothing
gf <- focalMat(chm_grid, sigma_chm, 'Gauss')
chm_smooth <- focal(chm_grid, w = gf)

# =============================================================================
# STEP 4: Individual Tree Detection
# =============================================================================
cat("\n=== 4. TREE DETECTION ===\n")

cat("  Detecting tree tops (variable window filter)...\n")
ttops <- locate_trees(chm_smooth, lmf(vwf_fun, hmin = min_h_trees, shape = 'circular'))
cat("  Tree tops found:", nrow(ttops), "\n")

if (nrow(ttops) == 0) {
  stop("No trees detected!")
}

# =============================================================================
# STEP 5: Crown Segmentation
# =============================================================================
cat("\n=== 5. CROWN SEGMENTATION ===\n")

cat("  Segmenting crowns (marker-controlled watershed)...\n")
# Use dalponte2016 algorithm for point-based segmentation
lidar_segmented <- segment_trees(lidar_veg, dalponte2016(chm_smooth, ttops, th_tree = min_h_trees))

# Get statistics
tree_ids <- unique(lidar_segmented@data$treeID)
tree_ids <- tree_ids[!is.na(tree_ids)]
cat("  Trees segmented:", length(tree_ids), "\n")

# Rename treeID to TreeID for DetailView compatibility
if ("treeID" %in% names(lidar_segmented@data)) {
  setnames(lidar_segmented@data, "treeID", "TreeID")
}

# Filter out unsegmented points (TreeID = NA)
lidar_segmented <- filter_poi(lidar_segmented, !is.na(TreeID))
cat("  Points with tree assignment:", nrow(lidar_segmented@data), "\n")

# =============================================================================
# STEP 6: Compute tree-level metrics
# =============================================================================
cat("\n=== 6. COMPUTING TREE METRICS ===\n")

# Calculate metrics per tree
tree_metrics <- lidar_segmented@data[, .(
  tree_H = max(Z),
  tree_X = mean(X),
  tree_Y = mean(Y),
  n_points = .N
), by = TreeID]

cat("  Trees with metrics:", nrow(tree_metrics), "\n")
cat("  Height range:", round(min(tree_metrics$tree_H), 1), "-", 
    round(max(tree_metrics$tree_H), 1), "m\n")
cat("  Mean height:", round(mean(tree_metrics$tree_H), 1), "m\n")

# Filter out very small trees (< 50 points)
small_trees <- tree_metrics[n_points < 50, TreeID]
if (length(small_trees) > 0) {
  cat("  Removing", length(small_trees), "trees with < 50 points\n")
  lidar_segmented <- filter_poi(lidar_segmented, !(TreeID %in% small_trees))
  tree_metrics <- tree_metrics[!(TreeID %in% small_trees)]
}

# Renumber tree IDs sequentially
old_ids <- sort(unique(lidar_segmented@data$TreeID))
new_ids <- seq_along(old_ids)
id_map <- data.table(old = old_ids, new = new_ids)
lidar_segmented@data <- merge(lidar_segmented@data, id_map, by.x = "TreeID", by.y = "old", all.x = TRUE)
lidar_segmented@data[, TreeID := new]
lidar_segmented@data[, new := NULL]

cat("  Final tree count:", length(unique(lidar_segmented@data$TreeID)), "\n")

# =============================================================================
# STEP 7: Save output
# =============================================================================
cat("\n=== 7. SAVING OUTPUT ===\n")

# Create output directory if needed
output_dir <- dirname(output_las_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Write segmented point cloud with TreeID as extra bytes attribute
if (!"TreeID" %in% names(lidar_segmented@data)) {
  stop("TreeID column not found in segmented data!")
}

# Create clean LAS with only standard columns; custom attributes are explicitly
# re-added as LAS extra bytes to guarantee persistence.
essential_cols <- c("X", "Y", "Z", "Intensity", "ReturnNumber", "NumberOfReturns",
                    "Classification", "R", "G", "B")
available_cols <- names(lidar_segmented@data)
cols_to_keep <- intersect(essential_cols, available_cols)
clean_data <- lidar_segmented@data[, ..cols_to_keep]

if (!"Z_TAW" %in% names(lidar_segmented@data)) {
  stop("Z_TAW column not found in segmented data (required for absolute elevation export)")
}

# Values for extra bytes attributes
treeid_values <- as.integer(lidar_segmented@data$TreeID)
ztaw_values <- as.numeric(lidar_segmented@data$Z_TAW)

# Create new LAS
lidar_clean <- LAS(clean_data)
crs(lidar_clean) <- tile_crs

# Add TreeID and original orthometric elevation (TAW) as extra bytes
lidar_clean <- add_lasattribute(lidar_clean, treeid_values, name = "TreeID", desc = "Tree ID")
lidar_clean <- add_lasattribute(lidar_clean, ztaw_values, name = "Z_TAW", desc = "Original orthometric elevation (TAW)")

# Write combined segmented point cloud
writeLAS(lidar_clean, output_las_path)
cat("  Saved combined LAS to:", output_las_path, "\n")

# Save tree metrics as CSV (required by DetailView)
metrics_path <- sub("\\.la[sz]$", "_tree_metrics.csv", output_las_path, ignore.case = TRUE)
tree_metrics_final <- lidar_segmented@data[, .(
  tree_H = max(Z),
  tree_X = mean(X),
  tree_Y = mean(Y),
  n_points = .N
), by = TreeID]

# Create CSV in DetailView format (filename, species_id, tree_H)
# species_id = -999 means unknown (to be predicted)
tree_csv <- data.table(
  filename = paste0("tree_", tree_metrics_final$TreeID, ".las"),
  species_id = -999,
  tree_H = tree_metrics_final$tree_H
)
fwrite(tree_csv, metrics_path)
cat("  Saved tree metadata to:", metrics_path, "\n")

# Save individual tree point clouds
individual_dir <- file.path(output_dir, "individual_trees")
dir.create(individual_dir, showWarnings = FALSE, recursive = TRUE)

cat("  Saving individual tree files to:", individual_dir, "\n")
tree_ids <- sort(unique(lidar_clean@data$TreeID))

for (tid in tree_ids) {
  tree_las <- filter_poi(lidar_clean, TreeID == tid)
  tree_path <- file.path(individual_dir, paste0("tree_", tid, ".las"))
  writeLAS(tree_las, tree_path)
}
cat("  Saved", length(tree_ids), "individual tree files\n")

cat("\n========================================\n")
cat("SEGMENTATION COMPLETE!\n")
cat("========================================\n")
cat("Total trees:", nrow(tree_metrics_final), "\n")
cat("Total points:", nrow(lidar_segmented@data), "\n")
cat("\nOutput files:\n")
cat("  - Segmented LAS:", output_las_path, "\n")
cat("  - Tree metrics:", metrics_path, "\n")

