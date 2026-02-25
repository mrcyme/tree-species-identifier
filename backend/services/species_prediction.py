"""
Species prediction service using DetailView
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import gc
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Add DetailView to path
PROJECT_DIR = Path(__file__).parent.parent.parent
DETAILVIEW_DIR = PROJECT_DIR / "DetailView"
sys.path.insert(0, str(DETAILVIEW_DIR))

# Species lookup
SPECIES_LOOKUP = {
    0: "Abies_alba",
    1: "Acer_campestre",
    2: "Acer_pseudoplatanus",
    3: "Acer_saccharum",
    4: "Betula_pendula",
    5: "Carpinus_betulus",
    6: "Corylus_avellana",
    7: "Crataegus_monogyna",
    8: "Eucalyptus_miniata",
    9: "Euonymus_europaeus",
    10: "Fagus_sylvatica",
    11: "Fraxinus_angustifolia",
    12: "Fraxinus_excelsior",
    13: "Larix_decidua",
    14: "Picea_abies",
    15: "Picea_glauca",
    16: "Pinus_contorta",
    17: "Pinus_nigra",
    18: "Pinus_pinaster",
    19: "Pinus_radiata",
    20: "Pinus_resinosa",
    21: "Pinus_sylvestris",
    22: "Populus_deltoides",
    23: "Populus_tremuloides",
    24: "Prunus_avium",
    25: "Pseudotsuga_menziesii",
    26: "Quercus_faginea",
    27: "Quercus_ilex",
    28: "Quercus_petraea",
    29: "Quercus_robur",
    30: "Quercus_rubra",
    31: "Tilia_cordata",
    32: "Ulmus_laevis"
}


def run_species_prediction(
    segmented_las: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    n_aug: int = 3  # Reduced from 5 for lower VRAM usage
) -> List[Dict[str, Any]]:
    """
    Run species prediction on segmented point cloud.
    
    Args:
        segmented_las: Path to segmented LAS file with TreeID
        output_dir: Directory for output files
        model_path: Path to model weights (optional, will download if not provided)
        n_aug: Number of augmentations for prediction
        
    Returns:
        List of species predictions per tree
    """
    try:
        import torch
        from predict import run_predict
    except ImportError as e:
        logger.error(f"Failed to import DetailView: {e}")
        raise RuntimeError(f"DetailView not available: {e}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default model path
    if model_path is None:
        model_path = PROJECT_DIR / "models" / "model_202305171452_60"
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}, will attempt to download")
    
    # Check CUDA availability and clear cache
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            # Check if CUDA device is actually available (not busy)
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Try to get device properties to verify it's accessible
                torch.cuda.get_device_properties(0)
                torch.cuda.empty_cache()
                logger.info(f"CUDA available: {device_count} device(s)")
            else:
                cuda_available = False
                logger.warning("CUDA device count is 0, falling back to CPU")
        except Exception as e:
            logger.warning(f"CUDA device check failed: {e}, falling back to CPU")
            cuda_available = False
    
    if not cuda_available:
        logger.warning("CUDA not available or device is busy, using CPU (this will be slower)")
        # Force CPU mode
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    logger.info(f"Running species prediction on {segmented_las}")
    logger.info(f"Model: {model_path}, Augmentations: {n_aug}, Device: {'CUDA' if cuda_available else 'CPU'}")
    
    # Retry logic for CUDA errors
    max_retries = 3
    retry_delay = 2  # seconds
    joined_df = None
    probs_df = None
    
    for attempt in range(max_retries):
        try:
            # Clear CUDA cache before each attempt
            if cuda_available and attempt > 0:
                torch.cuda.empty_cache()
                import time
                time.sleep(retry_delay)
                logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
            
            outfile, outfile_probs, joined_df, probs_df = run_predict(
                prediction_data=str(segmented_las),
                path_las="",
                model_path=str(model_path),
                tree_id_col="TreeID",
                n_aug=n_aug,
                output_dir=str(output_dir),
                path_csv_lookup=str(DETAILVIEW_DIR / "lookup.csv"),
                projection_backend="numpy",
                output_type="csv"
            )
            break  # Success, exit retry loop
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"CUDA error on attempt {attempt + 1}: {error_msg}")
                    logger.info(f"Will retry in {retry_delay} seconds...")
                    continue
                else:
                    # Last attempt failed, try CPU fallback
                    logger.error(f"CUDA failed after {max_retries} attempts: {error_msg}")
                    logger.info("Attempting CPU fallback...")
                    import os
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    cuda_available = False
                    # Aggressively release GPU memory before CPU fallback.
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    gc.collect()
                    # Try one more time with CPU
                    try:
                        # Force run_predict() to select CPU device even when CUDA
                        # remains visible to the current process.
                        orig_is_available = torch.cuda.is_available
                        torch.cuda.is_available = lambda: False
                        outfile, outfile_probs, joined_df, probs_df = run_predict(
                            prediction_data=str(segmented_las),
                            path_las="",
                            model_path=str(model_path),
                            tree_id_col="TreeID",
                            n_aug=n_aug,
                            output_dir=str(output_dir),
                            path_csv_lookup=str(DETAILVIEW_DIR / "lookup.csv"),
                            projection_backend="numpy",
                            output_type="csv"
                        )
                        torch.cuda.is_available = orig_is_available
                        logger.info("CPU fallback successful")
                        break
                    except Exception as cpu_error:
                        try:
                            torch.cuda.is_available = orig_is_available
                        except Exception:
                            pass
                        logger.error(f"CPU fallback also failed: {cpu_error}")
                        raise RuntimeError(f"Species prediction failed on both CUDA and CPU: {cpu_error}")
            else:
                # Non-CUDA error, don't retry
                raise
        except Exception as e:
            # Other errors, don't retry
            raise
    
    if joined_df is None:
        raise RuntimeError("Species prediction failed: No results obtained")
    
    # Parse results
    predictions = []
    for _, row in joined_df.iterrows():
        # Extract tree_id from filename (format: "tree_123")
        try:
            tree_id = int(row["filename"].split("_")[-1])
        except:
            tree_id = int(row["filename"])
        
        pred = {
            "tree_id": tree_id,
            "species_id": int(row["species_id"]),
            "species_name": row["species"],
            "confidence": float(row["species_prob"])
        }
        
        # Add probabilities if available
        if probs_df is not None:
            tree_row = probs_df[probs_df["File"] == row["filename"]]
            if len(tree_row) > 0:
                probs = {}
                for col in probs_df.columns[1:]:  # Skip "File" column
                    probs[col] = float(tree_row[col].values[0])
                pred["probabilities"] = probs
        
        predictions.append(pred)
    
    logger.info(f"Species prediction complete for {len(predictions)} trees")
    
    # Clear CUDA cache after predictions to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return predictions


def get_species_name(species_id: int) -> str:
    """Get species name from ID"""
    return SPECIES_LOOKUP.get(species_id, "Unknown")


def get_all_species() -> List[Dict[str, Any]]:
    """Get list of all supported species"""
    return [
        {"id": k, "name": v, "common_name": v.replace("_", " ")}
        for k, v in SPECIES_LOOKUP.items()
    ]

