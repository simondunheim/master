# PyGMTSAR Web Dashboard — Setup & Installation Guide

**Open-Source Software Development of Persistent Scatterer Interferometry with PyGMTSAR**

Master Thesis — M.Sc. Geoinformation, Berliner Hochschule für Technik
Author: Simon Eric Korfmacher

---

## Overview

This application provides a browser-based interface for Persistent Scatterer Interferometry (PSI) and Small Baseline Subset (SBAS) processing using PyGMTSAR. The system consists of two Docker containers:

- **Backend** — FastAPI server with PyGMTSAR processing engine, Dask distributed computing, and ASF data access
- **Frontend** — React/Vite application with OpenLayers-based interactive GeoTIFF viewer, Sentinel-1 burst search, and processing controls

The entire application runs via Docker Compose — no local Python, GMTSAR, or Node.js installation required.

---

## System Requirements

### Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| CPU | 4 cores | 8 cores |
| Disk Space | 50 GB free | 100 GB+ free |
| Docker | v20.10+ | Latest stable |
| Docker Compose | v2.0+ | Latest stable |
| OS | Linux, macOS, Windows (WSL2) | Ubuntu 22.04+ |

> **Important:** InSAR processing is memory-intensive. The backend container is configured with a 12 GB RAM limit and 4 GB reservation. Systems with less than 12 GB total RAM may experience out-of-memory errors during processing.

### Required Accounts

- **ASF Earthdata Account** (free): Required for Sentinel-1 data download
  - Register at: https://urs.earthdata.nasa.gov/users/new
  - This account provides access to the Alaska Satellite Facility data archive

---

## Project Structure

```
project-root/
├── docker-compose.yml          # Container orchestration
├── backend/
│   ├── Dockerfile.backend      # Backend container image
│   ├── main.py                 # FastAPI application entry point
│   ├── insar_processor.py      # PyGMTSAR processing engine
│   ├── config.txt              # ASF credentials configuration
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── Dockerfile              # Frontend container image
│   ├── src/
│   │   ├── App.tsx             # Main React application
│   │   └── App.css             # Application styles
│   ├── package.json            # Node.js dependencies
│   └── vite.config.ts          # Vite build configuration
├── data/                       # Sentinel-1 SLC data (auto-created)
├── results/                    # Processing results (auto-created)
├── processing/                 # Temporary processing files (auto-created)
└── dem_cache/                  # DEM cache (auto-created)
```

---

## Installation

### Step 1: Install Docker

**Ubuntu/Debian:**

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin

# Add your user to the docker group (avoids sudo for docker commands)
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
docker compose version
```

**macOS:**

Download and install Docker Desktop from https://www.docker.com/products/docker-desktop/

In Docker Desktop Settings → Resources, allocate at least **12 GB RAM** and **4 CPUs**.

**Windows (WSL2):**

1. Install WSL2: `wsl --install` in PowerShell (Admin)
2. Install Docker Desktop with WSL2 backend enabled
3. In Docker Desktop Settings → Resources → WSL Integration, enable your Linux distribution
4. Allocate at least **12 GB RAM** in Docker Desktop Settings → Resources

### Step 2: Clone the Repository

Install git on your system:
https://git-scm.com/install/windows

Varirfy in CMD / terminal:
git --version

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 3: Configure ASF Credentials

Edit `backend/config.txt` with your ASF Earthdata credentials:

```
asf_username=YOUR_ASF_USERNAME
asf_password=YOUR_ASF_PASSWORD
```

> **Security Note:** Do not commit this file with real credentials to a public repository. Add `backend/config.txt` to your `.gitignore`.

### Step 4: Create Data Directories

```bash
mkdir -p data results processing app_results dem_cache
```

### Step 5: Set File Permissions

The backend container runs as user `1000:1000`. Ensure the data directories are writable:

```bash
# Linux/macOS
sudo chown -R 1000:1000 data results processing app_results dem_cache

# Alternative: make directories world-writable (less secure, but works on all systems)
chmod -R 777 data results processing app_results dem_cache
```

---

## Starting the Application

### Start All Services

Notice: on Windows have DockerDesktop open

```bash
docker compose up --build
```

On first run, Docker will build both container images. This may take **10–20 minutes** depending on your internet connection (PyGMTSAR and its dependencies are large).

### Verify Startup

Wait until you see:

```
berlin-insar-backend   | INFO:     Uvicorn running on http://0.0.0.0:8000
berlin-insar-frontend  | VITE v5.x.x  ready in xxx ms
```

Then open your browser:

- **Frontend UI:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **Health Check:** http://localhost:8000/health

The frontend will show a green "Server connected" indicator in the top bar when the backend is reachable.

### Start in Background (Detached Mode)

```bash
docker compose up --build -d
```

### View Logs

```bash
# All services
docker compose logs -f

# Backend only
docker compose logs -f backend

# Frontend only
docker compose logs -f frontend
```

### Stop the Application

```bash
docker compose down
```

---

## Usage Workflow

### 1. Search Sentinel-1 Data

1. Open http://localhost:5173
2. Switch to the **"Data Search"** tab
3. Click **"Draw Bounding Box"** and draw your area of interest on the map
4. Set date range, orbit direction, and other search parameters
5. Click **"Search Available Burst Data"**
6. Review results showing available Sentinel-1 bursts with metadata (path, subswath, burst ID)

### 2. Download Data

1. Select desired bursts from search results using checkboxes
2. Click **"Download Selected"**
3. Monitor download progress in the status indicators
4. Downloaded data is stored in the `data/` directory

> **Note:** Sentinel-1 burst data can be several GB per scene. Ensure sufficient disk space.

### 3. Run InSAR Processing

1. Switch to the **"InSAR Processing"** tab
2. Click **"Perform SBAS/PSI Analysis"**
3. Monitor processing progress through the progress bar and status messages

The processing chain executes these steps automatically:

1. Scene scanning and validation
2. SBAS stack initialization
3. DEM download (SRTM)
4. Scene reframing to AOI
5. Image alignment (coregistration)
6. Geocoding transform computation
7. Baseline pair selection
8. Persistent Scatterer Function computation
9. Multi-look interferogram generation
10. SNAPHU 2D phase unwrapping
11. Topographic trend correction (7-variable regression)
12. Atmospheric turbulence correction (iterative)
13. Coherence-weighted least-squares displacement inversion
14. LOS velocity estimation
15. GeoTIFF and PNG export

Processing time depends on the number of scenes and area size. Typical: **2–8 hours** for 30 scenes.

### 4. View Results

Results are available in multiple ways:

- **In-browser:** GeoTIFF Viewer tab with interactive velocity field display
- **Server files:** Click "Load from Server" to browse generated GeoTIFF files
- **Drag & drop:** Drag any GeoTIFF onto the map for instant visualization
- **External:** Results in `results/` directory can be opened in QGIS, ArcGIS, or any GIS software

The GeoTIFF Viewer provides:

- Dynamic color scale (blue = subsidence, red = uplift)
- Adjustable velocity threshold filter
- Click-to-query pixel values (mm/year, coordinates)
- Automatic extent detection and zoom

---

## Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Check Docker memory allocation
docker stats

# Restart with fresh build
docker compose down
docker compose up --build
```

### Out of Memory errors during processing

The backend requires significant RAM for InSAR processing. If you see `Killed` or `MemoryError`:

1. Increase Docker memory limit in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G  # Increase from 12G
   ```

2. Or increase Docker Desktop memory allocation in Settings → Resources

3. Reduce the area of interest (smaller bounding box = less memory)

### Frontend can't connect to backend

- Ensure both containers are running: `docker compose ps`
- Check that `VITE_API_BASE_URL=http://localhost:8000` is set in `docker-compose.yml`
- Try accessing http://localhost:8000/health directly in the browser
- Check firewall settings (ports 5173 and 8000 must be accessible)

### Permission errors on data directories

```bash
# Fix ownership
sudo chown -R 1000:1000 data results processing app_results dem_cache

# Or use broader permissions
chmod -R 777 data results processing app_results dem_cache
```

### DEM download fails

The SRTM DEM download requires internet access from within the Docker container. If it fails:

- Check your internet connection
- Verify DNS resolution inside container: `docker compose exec backend ping google.com`
- The system has automatic retry logic with progressively smaller AOI

### Processing hangs at a specific step

Check backend logs for the last processing step:

```bash
docker compose logs -f backend | tail -50
```

Common bottleneck steps:

- **SNAPHU unwrapping:** Memory-intensive, can take 30+ minutes per interferogram
- **GeoTIFF export:** Switches to synchronous scheduler automatically to avoid memory spikes
- **Turbulence correction:** Has automatic fallback if primary method fails

---

## Configuration

### Docker Compose Resource Limits

Edit `docker-compose.yml` to adjust resources:

```yaml
deploy:
  resources:
    limits:
      memory: 12G    # Maximum RAM for backend
      cpus: '4.0'    # Maximum CPU cores
    reservations:
      memory: 4G     # Guaranteed minimum RAM
      cpus: '2.0'    # Guaranteed minimum CPU
```

### Processing Parameters

Key parameters are configurable in the processing request:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `referenceDate` | Auto (middle date) | Reference scene for stack |
| `orbit` | `A` (Ascending) | Orbit direction |
| `demType` | `SRTM` | Digital Elevation Model source |
| `temporal_baseline` | 50 days | Maximum temporal baseline for pairs |
| `perpendicular_baseline` | 150 m | Maximum perpendicular baseline |

### Network Configuration

The application uses a custom Docker bridge network (`insar-network`, subnet `172.20.0.0/16`). If this conflicts with your network, modify the `networks` section in `docker-compose.yml`.

---

## Output Files

After successful processing, the following files are generated in `results/<job_id>/`:

| File | Description |
|------|-------------|
| `velocity_sbas.tif` | LOS velocity field (mm/year), GeoTIFF |
| `displacement_sbas.tif` | Cumulative LOS displacement (mm), GeoTIFF |
| `psfunction.tif` | Persistent Scatterer Function, GeoTIFF |
| `velocity.png` | Velocity field visualization |
| `displacement.png` | Displacement time series visualization |
| `psfunction.png` | PS Function visualization |
| `topography.png` | DEM / topography of study area |
| `baseline_plot.png` | Temporal-perpendicular baseline network |
| `interferograms.png` | Sample wrapped interferograms |
| `correlations.png` | Coherence maps |
| `processing_summary.json` | Full processing metadata and parameters |

All GeoTIFF outputs are in **WGS84 (EPSG:4326)** projection with LOS velocity/displacement values in **millimeters per year**.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| InSAR Engine | PyGMTSAR + GMTSAR | PSI/SBAS processing |
| Distributed Computing | Dask | Parallel task execution, memory management |
| Backend Framework | FastAPI | REST API, async processing |
| Frontend Framework | React + Vite + TypeScript | User interface |
| Map Visualization | OpenLayers + WebGL | Interactive GeoTIFF rendering |
| Raster Rendering | ol/source/GeoTIFF + WebGLTileLayer | Client-side GeoTIFF display |
| Containerization | Docker + Docker Compose | Deployment, reproducibility |
| Data Access | ASF API | Sentinel-1 burst search and download |
| DEM Source | SRTM (via PyGMTSAR Tiles) | Topographic correction |

---

## License

This software was developed as part of a Master's thesis at Berliner Hochschule für Technik (BHT). PyGMTSAR is licensed under the BSD-3-Clause License. GMTSAR is licensed under the GPL-3.0 License.

---

## Citation

If you use this software in your research, please cite:

```
Korfmacher, S. E. (2025). Open-Source Software Development of Persistent Scatterer
Interferometry with PyGMTSAR. Master's Thesis, M.Sc. Geoinformation,
Berliner Hochschule für Technik.
```
