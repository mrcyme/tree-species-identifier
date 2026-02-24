# 🌳 Tree Species Identifier

A full-stack web application for individual tree segmentation and species classification from LiDAR point clouds.

![Tree Species Identifier](https://img.shields.io/badge/LiDAR-Point%20Cloud-green)
![Vue 3](https://img.shields.io/badge/Vue-3-42b883)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)
![deck.gl](https://img.shields.io/badge/deck.gl-9.0-blue)

## Overview

This tool provides:
1. **Tree Segmentation** - Uses the lidR package (R) to segment individual trees from urban LiDAR point clouds
2. **Species Identification** - Uses DetailView (PyTorch) deep learning model to classify tree species
3. **Web Interface** - Vue 3 frontend with deck.gl for interactive 3D visualization
4. **REST API** - FastAPI backend for processing and serving data

## Features

- 📤 **Upload LAS/LAZ point clouds** directly in the browser
- 🌲 **Automatic tree segmentation** using marker-controlled watershed
- 🔬 **AI-powered species identification** with confidence scores
- 🗺️ **Interactive 3D map** with Mapbox satellite imagery
- 🎯 **Click on trees** to view detailed information
- 📊 **Species probability distribution** for each tree
- 💾 **Download individual tree point clouds**

## Supported Species

The model can identify 33 tree species common in European forests:

| ID | Species | ID | Species |
|----|---------|----|---------| 
| 0 | Abies alba | 17 | Pinus nigra |
| 1 | Acer campestre | 18 | Pinus pinaster |
| 2 | Acer pseudoplatanus | 19 | Pinus radiata |
| 3 | Acer saccharum | 20 | Pinus resinosa |
| 4 | Betula pendula | 21 | Pinus sylvestris |
| 5 | Carpinus betulus | 22 | Populus deltoides |
| 6 | Corylus avellana | 23 | Populus tremuloides |
| 7 | Crataegus monogyna | 24 | Prunus avium |
| 8 | Eucalyptus miniata | 25 | Pseudotsuga menziesii |
| 9 | Euonymus europaeus | 26 | Quercus faginea |
| 10 | Fagus sylvatica | 27 | Quercus ilex |
| 11 | Fraxinus angustifolia | 28 | Quercus petraea |
| 12 | Fraxinus excelsior | 29 | Quercus robur |
| 13 | Larix decidua | 30 | Quercus rubra |
| 14 | Picea abies | 31 | Tilia cordata |
| 15 | Picea glauca | 32 | Ulmus laevis |
| 16 | Pinus contorta | | |

## Quick Start

```bash
# Clone and enter project
cd tree-species-identifier

# Start both backend and frontend
chmod +x start.sh
./start.sh
```

Then open http://localhost:3000 in your browser.

## Installation

### Prerequisites

1. **R (≥ 4.0)** with packages:
   ```r
   install.packages(c("lidR", "terra", "data.table"))
   ```

2. **Python (≥ 3.10)**

3. **Node.js (≥ 18)** with pnpm:
   ```bash
   npm install -g pnpm
   ```

4. **Mapbox Token** (optional, for satellite imagery):
   Get one at https://www.mapbox.com/

### Backend Setup

```bash
cd tree-species-identifier

# Create Python environment
python -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Download model weights (if not already present)
wget -O models/model_202305171452_60 \
  "https://freidata.uni-freiburg.de/records/xw42t-6mt03/files/model_202305171452_60?download=1"
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# (Optional) Configure Mapbox token
echo "VITE_MAPBOX_TOKEN=your_token_here" > .env
```

## Usage

### Web Application

1. **Start the backend**:
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

2. **Start the frontend** (in another terminal):
   ```bash
   cd frontend
   pnpm dev
   ```

3. **Open** http://localhost:3000

4. **Upload** a LAS/LAZ file and watch the magic happen!

### CLI Pipeline

```bash
# Basic usage
python src/identify_species.py input_pointcloud.las

# With options
python src/identify_species.py input_pointcloud.las \
  --output-dir ./output \
  --crs 31370 \
  --subset 149500,170500,149700,170700 \
  --n-aug 10
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory | `./output` |
| `--crs` | EPSG code for input data | `31370` |
| `--subset` | Extract subset: xmin,ymin,xmax,ymax | None |
| `--skip-segmentation` | Use pre-segmented LAS file | False |
| `--n-aug` | Number of augmentations | 10 |
| `--model-path` | Path to model weights | Auto-download |

### Segmentation Only

```bash
Rscript src/segment_trees.R input.las output_segmented.las 31370

# With subset
Rscript src/segment_trees.R input.las output_segmented.las 31370 \
  --subset 149500,170500,149700,170700
```

### API Documentation

When the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## Input Requirements

- **Format**: LAS or LAZ files
- **Classification**: Ground points must be classified (class 2). Vegetation classification (class 5) is recommended but not required.
- **Density**: Minimum 4 pts/m² recommended
- **CRS**: Any projected CRS (specify with `--crs`)

## Output Files

| File | Description |
|------|-------------|
| `*_segmented.las` | Point cloud with TreeID attribute |
| `*_tree_metrics.csv` | Tree-level metrics (height, position, point count) |
| `predictions_*.csv` | Species predictions per tree |
| `predictions_probs_*.csv` | Full probability distribution per species |
| `predictions_*.laz` | Point cloud with species_id and species_prob attributes |

## Project Structure

```
tree-species-identifier/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── services/
│   │   ├── segmentation.py # Tree segmentation service
│   │   └── species_prediction.py
│   ├── models/
│   │   └── schemas.py      # Pydantic models
│   └── requirements.txt
├── frontend/               # Vue 3 frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── MapView.vue     # deck.gl map
│   │   │   ├── TreePanel.vue   # Tree details panel
│   │   │   ├── TreeList.vue    # Tree list sidebar
│   │   │   ├── TreeViewer.vue  # 3D point cloud viewer
│   │   │   └── UploadPanel.vue # File upload
│   │   ├── services/
│   │   │   └── api.ts      # API client
│   │   ├── types/
│   │   │   └── index.ts    # TypeScript types
│   │   ├── App.vue
│   │   └── main.ts
│   ├── package.json
│   └── vite.config.ts
├── src/                    # CLI tools
│   ├── segment_trees.R     # R segmentation script
│   └── identify_species.py # CLI pipeline
├── DetailView/             # Species classification model
├── models/                 # Model weights
├── data/                   # Test data
├── output/                 # Processing results
├── start.sh               # Quick start script
└── README.md
```

## Credits

### Tree Segmentation
Based on the [lidR package](https://github.com/r-lidar/lidR) and workflows from [LiDAR-3D-Urban-Forest-Mapping](https://github.com/markusmnzngr/lidar-3d-urban-forest-mapping).

### Species Classification
Uses [DetailView](https://github.com/JulFrey/DetailView) by Zoe Schindler and Julian Frey (University of Freiburg).

**Citation:**
> Puliti, S., et al. (2024) Benchmarking tree species classification from proximally-sensed laser scanning data: introducing the FOR-species20K dataset. [ArXiv](https://www.arxiv.org/abs/2408.06507)

## License

MIT License (see LICENSE file)



## To do : 

- Elevation
- tree root LOD1 + LOD3 when clicking