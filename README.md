# Tree Species Identifier

Full-stack tooling for segmenting individual trees from LAS/LAZ point clouds, predicting species with the `DetailView` model, storing results in PostgreSQL, and browsing processed trees in a Vue/FastAPI web app.

## Current Scope

This repo currently supports:

- Uploading a LAS/LAZ file through the web UI and running segmentation + species prediction.
- Persisting segmented trees in PostgreSQL for later browsing.
- Viewing database-backed trees on a 2D satellite map with point-cloud inspection in the side panel.
- Generating procedural root systems and lightweight tree shape meshes.
- Batch ingestion from the Brussels Atom feed plus optional 3D Tiles export scripts.

## Stack

- `frontend/`: Vue 3 + Vite + MapLibre + deck.gl
- `backend/`: FastAPI + SQLAlchemy + async PostgreSQL access
- `src/segment_trees.R`: R/lidR segmentation pipeline
- `DetailView/`: species classification model code as a git submodule
- `docker-compose.yml`: local PostgreSQL/PostGIS service

## Repository Layout

```text
backend/      FastAPI app, DB models, services, batch scripts
frontend/     Vue app
src/          Standalone R/Python segmentation and species CLI pipeline
DetailView/   Git submodule used for species inference
data/         Local sample data and intermediate tree files (ignored by git)
models/       Local model weights (ignored by git)
output/       Generated processing output (ignored by git)
uploads/      Uploaded LAS/LAZ files (ignored by git)
```

## Prerequisites

- Python 3.10+
- R 4.0+ with `lidR`, `terra`, `data.table`
- Node.js 18+ and `pnpm`
- Docker + Docker Compose
- `uv` is recommended if you want `./start.sh` to bootstrap the virtualenv automatically
- Optional: CUDA-capable PyTorch environment for faster species inference

## Clone

The repo depends on the `DetailView` submodule:

```bash
git clone --recurse-submodules <repo-url>
cd tree-species-identifier
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Install Dependencies

### 1. Start the database

```bash
docker compose up -d
```

The backend defaults to:

```bash
DATABASE_URL=postgresql+asyncpg://treeapp:treeapp_secret@localhost:5433/trees
```

### 2. Install R packages

```bash
Rscript -e 'install.packages(c("lidR","terra","data.table"), repos="https://cloud.r-project.org")'
```

### 3. Install backend dependencies

Using `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r backend/requirements.txt
```

Using standard `venv`/`pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend
pnpm install
cd ..
```

### 5. Download model weights

The species service expects the model at `models/model_202305171452_60`:

```bash
mkdir -p models
wget -O models/model_202305171452_60 \
  "https://freidata.uni-freiburg.de/records/xw42t-6mt03/files/model_202305171452_60?download=1"
```

## Run The App

### Option A: one-command startup

```bash
chmod +x start.sh
./start.sh
```

This starts:

- frontend: `http://localhost:3000`
- backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

`start.sh` will reuse `.venv` if it already exists. If it does not exist, the script tries to create it with `uv`.

### Option B: manual startup

Backend:

```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
pnpm dev
```

The Vite dev server is configured for port `3000` and proxies `/api` to the backend.

## Web Workflow

1. Open `http://localhost:3000`
2. Upload a `.las` or `.laz` file
3. Choose the input EPSG code
4. Wait for segmentation and species prediction to complete
5. Inspect trees on the map or from the database-backed view

Notes:

- The frontend uses Esri World Imagery through MapLibre. No Mapbox token is required in the current version.
- On startup, the UI tries to load trees already stored in the database.

## Batch Processing Scripts

Run these from `backend/` with the virtualenv active.

### Ingest Brussels Atom feed tiles

```bash
python scripts/process_atom_feed_pointclouds.py \
  --feed-url "https://urbisdownload.datastore.brussels/atomfeed/ff1124e1-424e-11ee-b156-00090ffe0001-en.xml" \
  --tile-size-m 200 \
  --continue-on-error
```

Useful flags:

- `--limit 1` to test on a single item
- `--reprocess-existing` to force a rerun
- `--downloads-dir`, `--work-dir`, `--tiles-output` to move local output

### Add roots for older trees already in the DB

```bash
python scripts/predict_roots_for_existing_trees.py
```

### Generate shape meshes for stored tree point clouds

```bash
python scripts/predict_shape.py --batch-size 100 --max-faces 100
```

### Export DB shapes as batched 3D Tiles / GLB

```bash
python scripts/export_shape_tiles_to_gltf.py
```

## Standalone CLI Pipeline

The older non-web pipeline is still present in `src/identify_species.py`:

```bash
python src/identify_species.py input_pointcloud.las \
  --output-dir ./output \
  --crs 31370 \
  --n-aug 10
```

You can also pass `--subset xmin,ymin,xmax,ymax` or `--skip-segmentation`.

## Git Notes

The root Git config now ignores local environments, generated outputs, datasets, model weights, and large geospatial / ML artifacts. In this working tree those big directories were already local-only, not tracked:

- `data/`
- `models/`
- `output/`
- `uploads/`
- `.venv/`
- `.pnpm-store/`
- `frontend/node_modules/`

If you intentionally want to version a fixture from one of those locations later, move it into a dedicated tracked path first instead of overriding the ignore rules ad hoc.
