"""
FastAPI backend for Tree Species Identifier
"""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from models.schemas import (
    UploadResponse, ProcessingStatus, ProcessingResult,
    TreeInfo, TreeMetrics, SpeciesPrediction, PointCloudInfo
)
from services.segmentation import (
    run_segmentation, get_point_cloud_info, convert_to_web_format
)
from services.species_prediction import run_species_prediction, get_all_species
from services.root_prediction import predict_roots
from services.shape_prediction import generate_tree_shape
from database import init_db, close_db, get_db, async_session_maker
from database.repository import TreeRepository

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job storage (in-memory for simplicity, use Redis in production)
jobs: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Tree Species Identifier API")
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (run docker-compose up): {e}")
    yield
    logger.info("Shutting down Tree Species Identifier API")
    await close_db()


app = FastAPI(
    title="Tree Species Identifier API",
    description="API for tree segmentation and species identification from LiDAR point clouds",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (output data)
app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Tree Species Identifier API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "status": "/api/jobs/{job_id}",
            "result": "/api/jobs/{job_id}/result",
            "trees": "/api/jobs/{job_id}/trees",
            "species": "/api/species",
            "all_trees": "/api/trees",
            "tree_stats": "/api/trees/stats"
        }
    }


@app.get("/api/species")
async def list_species():
    """List all supported tree species"""
    return get_all_species()


# ============ Database-backed Tree Endpoints ============

@app.get("/api/trees")
async def get_all_trees_from_db(
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    with_species: bool = Query(default=False, description="Only return trees with species predictions")
):
    """Get all trees from the database (lightweight, excludes full root mesh data)"""
    try:
        async with async_session_maker() as session:
            repo = TreeRepository(session)
            trees = await repo.get_all_trees(limit=limit, offset=offset, with_species_only=with_species)
            # Use include_root_mesh=False to reduce payload size
            return {
                "trees": [tree.to_dict(include_root_mesh=False) for tree in trees],
                "count": len(trees),
                "limit": limit,
                "offset": offset
            }
    except Exception as e:
        logger.error(f"Failed to fetch trees: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/api/trees/stats")
async def get_trees_stats():
    """Get statistics about trees in the database"""
    try:
        async with async_session_maker() as session:
            repo = TreeRepository(session)
            stats = await repo.get_trees_stats()
            return stats
    except Exception as e:
        logger.error(f"Failed to fetch tree stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/api/trees/bbox")
async def get_trees_in_bbox(
    min_lon: float = Query(..., description="Minimum longitude"),
    min_lat: float = Query(..., description="Minimum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    max_lat: float = Query(..., description="Maximum latitude")
):
    """Get trees within a bounding box (lightweight, excludes full root mesh data)"""
    try:
        async with async_session_maker() as session:
            repo = TreeRepository(session)
            trees = await repo.get_trees_in_bbox(min_lon, min_lat, max_lon, max_lat)
            # Use include_root_mesh=False to reduce payload size
            return {
                "trees": [tree.to_dict(include_root_mesh=False) for tree in trees],
                "count": len(trees),
                "bbox": {
                    "min_lon": min_lon,
                    "min_lat": min_lat,
                    "max_lon": max_lon,
                    "max_lat": max_lat
                }
            }
    except Exception as e:
        logger.error(f"Failed to fetch trees in bbox: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/api/trees/{tree_id}")
async def get_tree_from_db(tree_id: int):
    """Get a specific tree from the database"""
    try:
        async with async_session_maker() as session:
            repo = TreeRepository(session)
            tree = await repo.get_tree_by_id(tree_id)
            if not tree:
                raise HTTPException(status_code=404, detail=f"Tree {tree_id} not found")
            return tree.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch tree {tree_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/api/trees/{tree_id}/pointcloud")
async def get_db_tree_pointcloud(tree_id: int, format: str = Query(default="json")):
    """Get point cloud data for a tree from database"""
    try:
        async with async_session_maker() as session:
            repo = TreeRepository(session)
            tree = await repo.get_tree_by_id(tree_id)
            if not tree:
                raise HTTPException(status_code=404, detail=f"Tree {tree_id} not found")
            
            if not tree.point_cloud_path:
                raise HTTPException(status_code=404, detail="Point cloud path not available")
            
            tree_las = Path(tree.point_cloud_path)
            if not tree_las.exists():
                raise HTTPException(status_code=404, detail="Point cloud file not found")
            
            if format == "las":
                return FileResponse(
                    str(tree_las),
                    media_type="application/octet-stream",
                    filename=f"tree_{tree_id}.las"
                )
            
            # Convert to JSON for web
            import laspy
            import numpy as np
            
            las = laspy.read(str(tree_las))
            
            x = np.array(las.x)
            y = np.array(las.y)
            z = np.array(las.z)
            
            # Use local coordinates centered at the tree (in meters)
            center_x, center_y, center_z = x.mean(), y.mean(), z.mean()
            local_x = x - center_x
            local_y = y - center_y
            local_z = z - center_z
            
            # Get colors
            if hasattr(las, 'red'):
                colors = np.column_stack([
                    las.red / 65535 * 255,
                    las.green / 65535 * 255,
                    las.blue / 65535 * 255
                ]).astype(np.uint8).tolist()
            else:
                colors = [[34, 139, 34]] * len(x)
            
            return JSONResponse({
                "tree_id": tree_id,
                "positions": [[float(lx), float(ly), float(lz)] for lx, ly, lz in zip(local_x, local_y, local_z)],
                "colors": colors,
                "count": len(x),
                "center": [0.0, 0.0, 0.0],
                "bbox_min": [float(local_x.min()), float(local_y.min()), float(local_z.min())],
                "bbox_max": [float(local_x.max()), float(local_y.max()), float(local_z.max())]
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pointcloud for tree {tree_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


# ============ Job-based Processing Endpoints ============

@app.post("/api/upload", response_model=UploadResponse)
async def upload_point_cloud(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    crs: str = Query(default="31370", description="EPSG code for the input CRS"),
    n_aug: int = Query(default=5, description="Number of augmentations for species prediction")
):
    """
    Upload a point cloud file for processing.
    
    Returns a job_id that can be used to track processing status.
    """
    # Validate file extension
    if not file.filename.lower().endswith(('.las', '.laz')):
        raise HTTPException(status_code=400, detail="Only LAS/LAZ files are supported")
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Create job directory
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    upload_path = job_dir / file.filename
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize job status
    now = datetime.now()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Upload complete, queued for processing",
        "created_at": now,
        "updated_at": now,
        "input_file": str(upload_path),
        "crs": crs,
        "n_aug": n_aug,
        "filename": file.filename
    }
    
    # Start processing in background
    background_tasks.add_task(process_point_cloud, job_id)
    
    return UploadResponse(
        job_id=job_id,
        message=f"File '{file.filename}' uploaded successfully. Processing started."
    )


async def process_point_cloud(job_id: str):
    """Background task to process point cloud"""
    job = jobs.get(job_id)
    if not job:
        return
    
    job_dir = OUTPUT_DIR / job_id
    input_file = Path(job["input_file"])
    crs = job["crs"]
    n_aug = job["n_aug"]
    filename = job["filename"]
    
    point_cloud_id = None
    
    try:
        # Step 1: Get point cloud info
        job["status"] = "analyzing"
        job["progress"] = 5
        job["message"] = "Analyzing point cloud..."
        job["updated_at"] = datetime.now()
        
        pc_info = get_point_cloud_info(input_file, crs)
        job["point_cloud_info"] = pc_info
        
        # Step 2: Run segmentation
        job["status"] = "segmenting"
        job["progress"] = 10
        job["message"] = "Segmenting individual trees..."
        job["updated_at"] = datetime.now()
        
        # Run segmentation in thread pool
        loop = asyncio.get_event_loop()
        seg_result = await loop.run_in_executor(
            None,
            lambda: run_segmentation(input_file, job_dir, crs)
        )
        
        job["progress"] = 50
        job["message"] = f"Segmented {len(seg_result['trees'])} trees"
        job["updated_at"] = datetime.now()
        job["segmentation_result"] = seg_result
        
        # Step 2.5: Save trees to database
        try:
            async with async_session_maker() as session:
                repo = TreeRepository(session)
                
                # Create or get point cloud record
                pc_center = pc_info.get("center", [None, None])
                pc = await repo.get_or_create_point_cloud(
                    filename=filename,
                    crs=f"EPSG:{crs}",
                    bbox_min=(pc_info.get("bbox_min", [None, None])[0], pc_info.get("bbox_min", [None, None])[1]) if pc_info.get("bbox_min") else None,
                    bbox_max=(pc_info.get("bbox_max", [None, None])[0], pc_info.get("bbox_max", [None, None])[1]) if pc_info.get("bbox_max") else None,
                    n_points=pc_info.get("n_points")
                )
                point_cloud_id = pc.id
                job["point_cloud_db_id"] = point_cloud_id
                
                # Add point cloud path to tree data
                individual_dir = job_dir / "individual_trees"
                for tree_data in seg_result["trees"]:
                    tree_data["point_cloud_path"] = str(individual_dir / f"tree_{tree_data['tree_id']}.las")
                
                # Bulk insert trees
                db_trees = await repo.bulk_upsert_trees(point_cloud_id, seg_result["trees"])
                
                # Update point cloud tree count
                pc.n_trees = len(db_trees)
                
                await repo.commit()
                logger.info(f"Saved {len(db_trees)} trees to database")
                
                # Store database IDs mapping
                job["tree_db_ids"] = {t.tree_id: t.id for t in db_trees}
                
        except Exception as db_error:
            logger.warning(f"Database save failed (continuing without DB): {db_error}")
        
        # Step 3: Convert to web format
        job["status"] = "converting"
        job["progress"] = 55
        job["message"] = "Converting point cloud for web display..."
        job["updated_at"] = datetime.now()
        
        segmented_las = Path(seg_result["segmented_las"])
        web_json = job_dir / "pointcloud.json"
        await loop.run_in_executor(
            None,
            lambda: convert_to_web_format(segmented_las, web_json, crs)
        )
        job["web_pointcloud"] = str(web_json)
        
        # Step 4: Run species prediction
        job["status"] = "predicting"
        job["progress"] = 60
        job["message"] = "Predicting tree species..."
        job["updated_at"] = datetime.now()
        
        predictions = await loop.run_in_executor(
            None,
            lambda: run_species_prediction(segmented_las, job_dir / "predictions", n_aug=n_aug)
        )
        
        job["progress"] = 90
        job["predictions"] = predictions
        
        # Step 4.5: Update database with species predictions
        if point_cloud_id:
            try:
                async with async_session_maker() as session:
                    repo = TreeRepository(session)
                    
                    for pred in predictions:
                        await repo.update_species_prediction(
                            tree_id=pred["tree_id"],
                            point_cloud_id=point_cloud_id,
                            species_id=pred["species_id"],
                            species_name=pred["species_name"],
                            confidence=pred["confidence"],
                            probabilities=pred.get("probabilities")
                        )
                    
                    await repo.commit()
                    logger.info(f"Updated species predictions for {len(predictions)} trees in database")
                    
            except Exception as db_error:
                logger.warning(f"Database species update failed: {db_error}")
        
        # Step 4.6: Generate root predictions and shape predictions
        job["progress"] = 92
        job["message"] = "Generating root and shape predictions..."
        job["updated_at"] = datetime.now()
        
        # Get tree_db_ids mapping from job
        tree_db_ids = job.get("tree_db_ids", {})
        
        root_predictions = {}
        shape_predictions = {}
        for tree in seg_result["trees"]:
            tree_id = tree["tree_id"]
            db_id = tree_db_ids.get(tree_id)
            
            # Generate roots based on tree dimensions
            try:
                root_result = predict_roots(
                    tree_height=tree["height"],
                    crown_diameter=tree.get("crown_diameter", tree["height"] * 0.5),  # Estimate if not available
                    root_type=None,  # Random selection
                    seed=db_id if db_id else tree_id  # Use DB ID for reproducibility
                )
                root_predictions[tree_id] = root_result
            except Exception as root_error:
                logger.warning(f"Root prediction failed for tree {tree_id}: {root_error}")
            
            # Generate 3D shape mesh
            try:
                tree_las_path = individual_dir / f"tree_{tree_id}.las"
                if tree_las_path.exists():
                    import laspy
                    import numpy as np
                    
                    las = laspy.read(str(tree_las_path))
                    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
                    
                    # Convert to local coordinates centered at tree
                    center_x, center_y, center_z = x.mean(), y.mean(), z.mean()
                    local_points = np.column_stack([x - center_x, y - center_y, z - center_z])
                    
                    shape_result = generate_tree_shape(local_points, max_faces=100)
                    shape_predictions[tree_id] = shape_result
            except Exception as shape_error:
                logger.warning(f"Shape prediction failed for tree {tree_id}: {shape_error}")
        
        job["root_predictions"] = root_predictions
        job["shape_predictions"] = shape_predictions
        logger.info(f"Generated root predictions for {len(root_predictions)} trees and shape predictions for {len(shape_predictions)} trees")
        
        # Step 4.7: Update database with root and shape predictions
        if point_cloud_id and (root_predictions or shape_predictions):
            try:
                async with async_session_maker() as session:
                    repo = TreeRepository(session)
                    
                    for tree_id, root_data in root_predictions.items():
                        db_id = tree_db_ids.get(tree_id)
                        if db_id:
                            await repo.update_root_prediction(
                                db_id=db_id,
                                root_type=root_data["root_type"],
                                root_lod0=root_data["lod0"],
                                root_lod1=root_data["lod1"],
                                root_lod2=root_data["lod2"],
                                root_lod3=root_data["lod3"],
                                root_seed=root_data["metadata"].get("seed")
                            )
                    
                    for tree_id, shape_data in shape_predictions.items():
                        db_id = tree_db_ids.get(tree_id)
                        if db_id:
                            await repo.update_shape_prediction(
                                db_id=db_id,
                                shape_mesh=shape_data
                            )
                    
                    await repo.commit()
                    logger.info(f"Saved root and shape predictions to database")
                    
            except Exception as db_error:
                logger.warning(f"Database root/shape prediction update failed: {db_error}")
        
        # Step 5: Merge results
        job["status"] = "finalizing"
        job["progress"] = 95
        job["message"] = "Finalizing results..."
        job["updated_at"] = datetime.now()
        
        # Build complete tree info
        trees = []
        pred_by_id = {p["tree_id"]: p for p in predictions}
        tree_db_ids = job.get("tree_db_ids", {})
        
        for tree in seg_result["trees"]:
            tree_id = tree["tree_id"]
            db_id = tree_db_ids.get(tree_id)
            
            metrics = TreeMetrics(
                tree_id=tree_id,
                height=tree["height"],
                crown_diameter=tree.get("crown_diameter"),
                n_points=tree["n_points"],
                x=tree["x"],
                y=tree["y"],
                z_min=tree["z_min"],
                z_max=tree["z_max"],
                bbox_min=tree["bbox_min"],
                bbox_max=tree["bbox_max"],
                longitude=tree.get("longitude"),
                latitude=tree.get("latitude")
            )
            
            species = None
            if tree_id in pred_by_id:
                p = pred_by_id[tree_id]
                species = SpeciesPrediction(
                    species_id=p["species_id"],
                    species_name=p["species_name"],
                    confidence=p["confidence"],
                    probabilities=p.get("probabilities")
                )
            
            tree_info = TreeInfo(
                tree_id=tree_id,
                metrics=metrics,
                species=species,
                point_cloud_url=f"/api/jobs/{job_id}/trees/{tree_id}/pointcloud",
                db_id=db_id
            )
            tree_dict = tree_info.model_dump()
            
            # Add root data if available
            if tree_id in root_predictions:
                tree_dict["root"] = root_predictions[tree_id]
            
            # Add shape data if available
            if tree_id in shape_predictions:
                tree_dict["shape"] = shape_predictions[tree_id]
            
            trees.append(tree_dict)
        
        job["trees"] = trees
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = f"Processing complete. Identified {len(trees)} trees."
        job["updated_at"] = datetime.now()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Processing failed: {e}"
        job["updated_at"] = datetime.now()


@app.get("/api/jobs/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProcessingStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job.get("message"),
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the complete result of a processing job"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] not in ["completed", "failed"]:
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": "Processing not yet complete"
        }
    
    if job["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": job.get("error")
        }
    
    # Return complete result
    pc_info = job.get("point_cloud_info", {})
    
    return {
        "job_id": job_id,
        "status": "completed",
        "point_cloud": {
            "filename": job["filename"],
            "n_points": pc_info.get("n_points"),
            "crs": pc_info.get("crs"),
            "bbox_min": pc_info.get("bbox_min"),
            "bbox_max": pc_info.get("bbox_max"),
            "center": pc_info.get("center"),
            "data_url": f"/api/jobs/{job_id}/pointcloud"
        },
        "trees": job.get("trees", []),
        "tree_count": len(job.get("trees", []))
    }


@app.get("/api/jobs/{job_id}/trees")
async def get_trees(job_id: str):
    """Get all trees for a job"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet complete")
    
    return job.get("trees", [])


@app.get("/api/jobs/{job_id}/trees/{tree_id}")
async def get_tree(job_id: str, tree_id: int):
    """Get details for a specific tree"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet complete")
    
    trees = job.get("trees", [])
    tree = next((t for t in trees if t["tree_id"] == tree_id), None)
    
    if not tree:
        raise HTTPException(status_code=404, detail=f"Tree {tree_id} not found")
    
    return tree


@app.get("/api/jobs/{job_id}/trees/{tree_id}/pointcloud")
async def get_tree_pointcloud(job_id: str, tree_id: int, format: str = Query(default="json")):
    """Get point cloud data for a specific tree"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = OUTPUT_DIR / job_id
    tree_las = job_dir / "individual_trees" / f"tree_{tree_id}.las"
    
    if not tree_las.exists():
        raise HTTPException(status_code=404, detail=f"Point cloud for tree {tree_id} not found")
    
    if format == "las":
        return FileResponse(
            str(tree_las),
            media_type="application/octet-stream",
            filename=f"tree_{tree_id}.las"
        )
    
    # Convert to JSON for web - use LOCAL coordinates (meters) for 3D tree viewer
    # WGS84 would make the tree look like a vertical line due to tiny decimal variations
    import laspy
    import numpy as np
    
    las = laspy.read(str(tree_las))
    
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    
    # Use local coordinates centered at the tree (in meters)
    # This preserves the proper shape for 3D viewing
    center_x, center_y, center_z = x.mean(), y.mean(), z.mean()
    local_x = x - center_x
    local_y = y - center_y
    local_z = z - center_z
    
    # Get colors
    if hasattr(las, 'red'):
        colors = np.column_stack([
            las.red / 65535 * 255,
            las.green / 65535 * 255,
            las.blue / 65535 * 255
        ]).astype(np.uint8).tolist()
    else:
        colors = [[34, 139, 34]] * len(x)
    
    return JSONResponse({
        "tree_id": tree_id,
        # Positions in local meters (centered at tree)
        "positions": [[float(lx), float(ly), float(lz)] for lx, ly, lz in zip(local_x, local_y, local_z)],
        "colors": colors,
        "count": len(x),
        # Center is at origin in local coords
        "center": [0.0, 0.0, 0.0],
        "bbox_min": [float(local_x.min()), float(local_y.min()), float(local_z.min())],
        "bbox_max": [float(local_x.max()), float(local_y.max()), float(local_z.max())]
    })


@app.get("/api/jobs/{job_id}/pointcloud")
async def get_full_pointcloud(job_id: str):
    """Get the full point cloud data for web display"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    web_json = job.get("web_pointcloud")
    if not web_json or not Path(web_json).exists():
        raise HTTPException(status_code=404, detail="Point cloud data not ready")
    
    return FileResponse(
        web_json,
        media_type="application/json",
        filename="pointcloud.json"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
