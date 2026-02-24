#!/usr/bin/env python3
"""
Tree Species Identification Pipeline

This script integrates:
1. Tree segmentation from LiDAR point clouds (R-based)
2. Species identification using DetailView (PyTorch-based)

Usage:
    python identify_species.py <input_las> [options]

Options:
    --output-dir    Output directory (default: ./output)
    --crs           EPSG code for the input data (default: 31370)
    --subset        Extract subset: xmin,ymin,xmax,ymax
    --skip-segmentation  Use pre-segmented LAS file
    --n-aug         Number of augmentations for species prediction (default: 10)
    --gpu           Use GPU for inference (default: auto-detect)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add DetailView to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DETAILVIEW_DIR = PROJECT_DIR / "DetailView"
sys.path.insert(0, str(DETAILVIEW_DIR))


def run_segmentation(input_las: Path, output_las: Path, crs: str, 
                     subset: str = None) -> bool:
    """Run R-based tree segmentation."""
    print("\n" + "="*60)
    print("STEP 1: TREE SEGMENTATION")
    print("="*60)
    
    r_script = SCRIPT_DIR / "segment_trees.R"
    
    cmd = ["Rscript", str(r_script), str(input_las), str(output_las), crs]
    
    if subset:
        cmd.extend(["--subset", subset])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error during segmentation: {e}")
        return False
    except FileNotFoundError:
        print("Error: Rscript not found. Please install R.")
        return False


def run_species_identification(segmented_las: Path, output_dir: Path,
                                n_aug: int = 10, 
                                model_path: Path = None) -> dict:
    """Run DetailView species identification."""
    print("\n" + "="*60)
    print("STEP 2: SPECIES IDENTIFICATION")
    print("="*60)
    
    # Import DetailView predict module
    try:
        from predict import run_predict
    except ImportError as e:
        print(f"Error importing DetailView: {e}")
        print("Make sure DetailView is properly set up.")
        return None
    
    # Set default model path
    if model_path is None:
        model_path = PROJECT_DIR / "models" / "model_202305171452_60"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run prediction
    print(f"Input: {segmented_las}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_path}")
    print(f"Augmentations: {n_aug}")
    
    try:
        outfile, outfile_probs, joined_df, probs_df = run_predict(
            prediction_data=str(segmented_las),
            path_las="",
            model_path=str(model_path),
            tree_id_col="TreeID",
            n_aug=n_aug,
            output_dir=str(output_dir),
            path_csv_lookup=str(DETAILVIEW_DIR / "lookup.csv"),
            projection_backend="numpy",  # More stable
            output_type="both"  # CSV and LAS output
        )
        
        return {
            "predictions_csv": outfile,
            "probabilities_csv": outfile_probs,
            "predictions_df": joined_df,
            "probabilities_df": probs_df
        }
    except Exception as e:
        print(f"Error during species identification: {e}")
        import traceback
        traceback.print_exc()
        return None


def summarize_results(results: dict, output_dir: Path):
    """Print summary of species identification results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if results is None or "predictions_df" not in results:
        print("No results available.")
        return
    
    df = results["predictions_df"]
    
    print(f"\nTotal trees identified: {len(df)}")
    print(f"\nSpecies distribution:")
    
    species_counts = df["species"].value_counts()
    for species, count in species_counts.items():
        pct = count / len(df) * 100
        print(f"  {species}: {count} ({pct:.1f}%)")
    
    print(f"\nMean prediction confidence: {df['species_prob'].mean():.2%}")
    print(f"Median prediction confidence: {df['species_prob'].median():.2%}")
    
    # High confidence predictions
    high_conf = df[df["species_prob"] >= 0.7]
    print(f"\nHigh confidence predictions (>70%): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")
    
    print(f"\nOutput files:")
    print(f"  - Predictions: {results['predictions_csv']}")
    print(f"  - Probabilities: {results['probabilities_csv']}")


def main():
    parser = argparse.ArgumentParser(
        description="Tree Species Identification from LiDAR Point Clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with subset
  python identify_species.py input.las --subset 149500,170500,149700,170700

  # Use pre-segmented file
  python identify_species.py segmented.las --skip-segmentation

  # Specify output directory
  python identify_species.py input.las --output-dir ./my_output
        """
    )
    
    parser.add_argument("input_las", type=str,
                        help="Input LAS/LAZ file")
    parser.add_argument("--output-dir", "-o", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--crs", type=str, default="31370",
                        help="EPSG code for input data (default: 31370)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Extract subset: xmin,ymin,xmax,ymax")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip segmentation (input is pre-segmented)")
    parser.add_argument("--n-aug", type=int, default=10,
                        help="Number of augmentations for prediction (default: 10)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to DetailView model weights")
    
    args = parser.parse_args()
    
    # Setup paths
    input_las = Path(args.input_las).absolute()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("TREE SPECIES IDENTIFICATION PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {input_las}")
    print(f"Output: {output_dir}")
    
    if not input_las.exists():
        print(f"Error: Input file not found: {input_las}")
        sys.exit(1)
    
    # Step 1: Segmentation
    if args.skip_segmentation:
        print("\nSkipping segmentation (using pre-segmented input)")
        segmented_las = input_las
    else:
        segmented_las = output_dir / f"{input_las.stem}_segmented.las"
        
        success = run_segmentation(
            input_las=input_las,
            output_las=segmented_las,
            crs=args.crs,
            subset=args.subset
        )
        
        if not success or not segmented_las.exists():
            print("Error: Segmentation failed!")
            sys.exit(1)
    
    # Step 2: Species identification
    model_path = Path(args.model_path) if args.model_path else None
    
    results = run_species_identification(
        segmented_las=segmented_las,
        output_dir=output_dir,
        n_aug=args.n_aug,
        model_path=model_path
    )
    
    # Summary
    if results:
        summarize_results(results, output_dir)
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
    else:
        print("\nPipeline completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()









