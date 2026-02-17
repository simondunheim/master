#!/usr/bin/env python3
"""
FastAPI Backend for Berlin InSAR Processing - ENHANCED VERSION
Handles SBAS/PSI analysis, serves GeoTIFF results, AND provides Sentinel-1 burst search/download
"""

import os
import asyncio
import json
import uuid
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess
import logging

from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Berlin InSAR Processing API - Enhanced with Search & Download",
    description="API for SBAS and PS InSAR analysis with Sentinel-1 burst search and download capabilities",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"]
)

def read_config_file(config_path="config.txt"):
    """Read credentials from config file"""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        logger.info(f"Config file loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not read config file: {str(e)}")
        return {}

config = read_config_file()

# NASA Earthdata Login credentials
EARTHDATA_USERNAME = config.get("EARTHDATA_USERNAME") or os.environ.get("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = config.get("EARTHDATA_PASSWORD") or os.environ.get("EARTHDATA_PASSWORD", "")
EARTHDATA_TOKEN = config.get("EARTHDATA_TOKEN") or os.environ.get("EARTHDATA_TOKEN", "")

processing_jobs: Dict[str, Dict] = {}

def setup_results_directory():
    """Setup results directory with fallback options"""
    result_dirs = [
        Path("results"),           # Primary choice
        Path("/app/results"),      # Explicit path
        Path("/tmp/results"),      # Fallback to tmp
        Path.home() / "results"    # User home fallback
    ]
    
    for results_dir in result_dirs:
        try:
            results_dir.mkdir(exist_ok=True)
            # Test write permissions
            test_file = results_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"Using results directory: {results_dir}")
            return results_dir
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot use {results_dir}: {e}")
            continue
    
    logger.warning("Using processing directory for results (no separate results dir)")
    return Path("/app/processing")

RESULTS_DIR = setup_results_directory()

# Download directory for Sentinel-1 data
DOWNLOAD_DIR = Path("./sentinel1_burst_downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

download_statuses = {}

# ===============================
# DATA MODELS 
# ===============================

class ProcessingRequest(BaseModel):
    analysis_type: str = "complete"  # "sbas", "ps", or "complete"
    email_notification: Optional[str] = None

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    progress: float
    message: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    results: Optional[Dict] = None

class MapLayer(BaseModel):
    name: str
    type: str  # "velocity", "displacement", "psf", etc.
    method: str  # "sbas" or "ps"
    file_path: str
    bounds: List[float]  # [minx, miny, maxx, maxy] in WGS84
    color_range: List[float]  # [min_value, max_value]
    projection: Optional[str] = None  # CRS identifier

# NEW models for search and download
class BoundingBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

class SearchRequest(BaseModel):
    boundingBox: BoundingBox
    startDate: str
    endDate: str
    productType: str = "SLC"
    polarization: Optional[str] = None
    orbitDirection: Optional[str] = None
    path: Optional[int] = None
    frame: Optional[int] = None
    subswath: Optional[str] = None
    burstID: Optional[str] = None
    fullBurstID: Optional[str] = None

class SentinelProduct(BaseModel):
    id: str
    title: str
    date: str
    footprint: str
    size: str
    thumbnailUrl: Optional[str] = None
    metadata: Dict[str, Any]
    burstID: Optional[str] = None
    subswath: Optional[str] = None
    burstIndex: Optional[int] = None

class SearchResponse(BaseModel):
    products: List[SentinelProduct]
    totalResults: int
    asfUrl: str

class DownloadStatus(BaseModel):
    productId: str
    status: str  # 'started', 'downloading', 'completed', 'failed'
    message: str
    progress: Optional[float] = None

# ===============================
# HELPER FUNCTIONS 
# ===============================

def generate_asf_url(request: SearchRequest) -> str:
    """Generate ASF Alaska data search tool URL for burst data based on search parameters"""
    base_url = "https://search.asf.alaska.edu/#/"
    
    # Calculate center point from bounding box
    center_lon = (request.boundingBox.west + request.boundingBox.east) / 2
    center_lat = (request.boundingBox.south + request.boundingBox.north) / 2
    
    lon_diff = abs(request.boundingBox.east - request.boundingBox.west)
    lat_diff = abs(request.boundingBox.north - request.boundingBox.south)
    max_diff = max(lon_diff, lat_diff)
    
    if max_diff > 10:
        zoom = 6
    elif max_diff > 5:
        zoom = 7
    elif max_diff > 2:
        zoom = 8
    elif max_diff > 1:
        zoom = 9
    elif max_diff > 0.5:
        zoom = 10
    else:
        zoom = 11
    
    # Create polygon string
    polygon = f"POLYGON(({request.boundingBox.west} {request.boundingBox.south},{request.boundingBox.east} {request.boundingBox.south},{request.boundingBox.east} {request.boundingBox.north},{request.boundingBox.west} {request.boundingBox.north},{request.boundingBox.west} {request.boundingBox.south}))"
    
    # Convert dates to ISO format with Z timezone
    start_date = f"{request.startDate}T00:00:00Z"
    end_date = f"{request.endDate}T23:59:59Z"
    
    params = {
        'zoom': str(zoom),
        'center': f"{center_lon},{center_lat}",
        'polygon': polygon,
        'start': start_date,
        'end': end_date,
        'dataset': 'SENTINEL-1 BURSTS',
        'beamModes': 'IW',
        'resultsLoaded': 'true'
    }
    
    # Add optional parameters
    if request.orbitDirection and request.orbitDirection.lower() != "both":
        flight_dir = request.orbitDirection.capitalize()
        params['flightDirs'] = flight_dir
    
    if request.polarization and request.polarization.lower() != "all":
        pol = request.polarization.upper()
        if pol in ['VV', 'VH', 'HH', 'HV']:
            params['polarizations'] = pol
        elif pol == 'DUAL':
            params['polarizations'] = 'VV+VH'
    
    if request.path is not None:
        params['relativeOrbit'] = str(request.path)
        params['path'] = f"{request.path}-"
    
    if request.fullBurstID:
        params['fullBurstIDs'] = request.fullBurstID
        logger.info(f"Adding fullBurstIDs parameter: {request.fullBurstID}")
    
    encoded_params = []
    for key, value in params.items():
        encoded_value = urllib.parse.quote(str(value), safe='')
        encoded_params.append(f"{key}={encoded_value}")
    
    # Construct final URL
    asf_url = f"{base_url}?{'&'.join(encoded_params)}"
    
    return asf_url

async def download_file_with_asf(product_id: str):
    """Background task to download a burst file using ASF"""
    try:
        # Check if asf_search and pygmtsar are available
        try:
            import asf_search as asf
            from pygmtsar import ASF as PyGMTSAR_ASF
        except ImportError as e:
            logger.error(f"Required dependencies not found: {e}")
            download_statuses[product_id].status = "failed"
            download_statuses[product_id].message = f"Required dependencies not installed: {e}"
            return
        
        download_statuses[product_id].status = "downloading"
        download_statuses[product_id].message = "Setting up ASF downloader..."
        
        if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
            raise Exception("Earthdata credentials not configured")
        
        asf_downloader = PyGMTSAR_ASF(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        
        download_path = str(DOWNLOAD_DIR)
        logger.info(f"Downloading to: {download_path}")
        
        os.makedirs(download_path, exist_ok=True)
        
        # Update status
        download_statuses[product_id].message = "Downloading burst file..."
        download_statuses[product_id].progress = 10.0
        
        # Download using PyGMTSAR's ASF class
        logger.info(f"Starting download for burst: {product_id}")
        
        # Download the burst
        asf_downloader.download(download_path, [product_id])
        
        files_found = False
        for root, dirs, files in os.walk(download_path):
            if files:
                files_found = True
                logger.info(f"Downloaded files in {root}: {files}")
                break
        
        if not files_found:
            logger.warning(f"No files found after download for {product_id}")
        
        download_statuses[product_id].status = "completed"
        download_statuses[product_id].message = f"Download completed to {DOWNLOAD_DIR}"
        download_statuses[product_id].progress = 100.0
        
        logger.info(f"Successfully downloaded burst {product_id}")
            
    except Exception as e:
        logger.error(f"Error downloading burst {product_id}: {str(e)}")
        download_statuses[product_id].status = "failed"
        download_statuses[product_id].message = f"Download failed: {str(e)}"

# ===============================
# ORIGINAL ENDPOINTS 
# ===============================

@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    """Handle CORS preflight requests"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Berlin InSAR Processing API - Enhanced with Search & Download",
        "version": "2.0.0",
        "results_dir": str(RESULTS_DIR),
        "download_dir": str(DOWNLOAD_DIR),
        "earthdata_credentials": "configured" if EARTHDATA_USERNAME and EARTHDATA_PASSWORD else "missing",
        "endpoints": {
            "start_processing": "/api/process",
            "check_status": "/api/status/{job_id}",
            "get_results": "/api/results/{job_id}",
            "list_layers": "/api/layers/{job_id}",
            "serve_geotiff_simple": "/api/raster/simple/{filename}",
            "list_backups": "/api/list-backups",
            "search_sentinel": "/search",
            "download_product": "/download/{product_id}",
            "download_status": "/download/status/{product_id}",
            "scan_slc": "/scan-slc",
            "health": "/health"
        }
    }

@app.get("/api/list-backups")
async def list_backups():
    """List all backed up result files"""
    try:
        backup_dir = Path("/app/app_results")
        
        if not backup_dir.exists():
            return {
                "status": "info",
                "message": "No backup directory found",
                "files": []
            }
        
        backup_files = []
        
        for file_path in backup_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                backup_files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(file_path)
                })
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "status": "success",
            "backup_directory": str(backup_dir),
            "total_files": len(backup_files),
            "files": backup_files,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        return {
            "status": "error",
            "message": f"Failed to list backups: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    credentials_status = "configured" if EARTHDATA_USERNAME and EARTHDATA_PASSWORD else "missing"
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "earthdata_credentials": credentials_status
    }

@app.get("/api/test-script")
async def test_script():
    """Test if the InSAR script can be executed"""
    try:
        script_path = Path("/app/berlin_insar.py")
        python_exec = "/usr/bin/python3"
        
        if not script_path.exists():
            return {"status": "error", "message": f"Script not found at {script_path}"}
        
        if not Path(python_exec).exists():
            python_exec = "python3"
        
        process = await asyncio.create_subprocess_exec(
            python_exec, "-c", f"import py_compile; py_compile.compile('{script_path}', doraise=True)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                "status": "success", 
                "message": "Script syntax is valid",
                "python_path": python_exec,
                "script_path": str(script_path)
            }
        else:
            return {
                "status": "error",
                "message": f"Script syntax error: {stderr.decode('utf-8')}",
                "python_path": python_exec,
                "script_path": str(script_path)
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Test failed: {str(e)}"}

@app.get("/api/check-data")
async def check_data():
    """Check if required data directories exist and have content"""
    try:
        data_dir = Path("/app/data_berlin_A")
        work_dir = Path("/app/raw_berlin_A")
        
        data_files = list(data_dir.glob("*.SAFE")) if data_dir.exists() else []
        work_files = list(work_dir.glob("*")) if work_dir.exists() else []
        
        return {
            "data_directory": {
                "path": str(data_dir),
                "exists": data_dir.exists(),
                "scene_count": len(data_files),
                "scenes": [f.name for f in data_files[:5]]  # First 5 scenes
            },
            "work_directory": {
                "path": str(work_dir),
                "exists": work_dir.exists(),
                "file_count": len(work_files)
            },
            "results_directory": {
                "path": str(RESULTS_DIR),
                "exists": RESULTS_DIR.exists(),
                "writable": os.access(RESULTS_DIR, os.W_OK)
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Check failed: {str(e)}"}

@app.post("/api/process", response_model=ProcessingStatus)
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start InSAR processing job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Processing job queued",
        "started_at": datetime.now(),
        "completed_at": None,
        "analysis_type": request.analysis_type,
        "results": None
    }
    
    background_tasks.add_task(run_insar_processing, job_id, request.analysis_type)
    
    return ProcessingStatus(**processing_jobs[job_id])

@app.get("/api/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProcessingStatus(**processing_jobs[job_id])

@app.get("/api/results/{job_id}")
async def get_processing_results(job_id: str):
    """Get processing results and available layers"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    return job["results"]

@app.get("/api/layers/{job_id}")
async def get_map_layers(job_id: str) -> List[MapLayer]:
    """Get available map layers for visualization"""
    
    # Handle special case for existing results
    if job_id == "existing-results":
        return await get_existing_layers()
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        return []
    
    layers = []
    result_files = job["results"]["files"]
    
    # Define layer configurations
    layer_configs = {
        "berlin_velocity.tif": {
            "name": "SBAS Velocity",
            "type": "velocity",
            "method": "sbas",
            "unit": "mm/year"
        },
        "berlin_ps_velocity.tif": {
            "name": "PS Velocity", 
            "type": "velocity",
            "method": "ps",
            "unit": "mm/year"
        },
        "berlin_displacement.tif": {
            "name": "SBAS Displacement",
            "type": "displacement", 
            "method": "sbas",
            "unit": "mm"
        },
        "berlin_ps_displacement.tif": {
            "name": "PS Displacement",
            "type": "displacement",
            "method": "ps", 
            "unit": "mm"
        },
        "berlin_psfunction.tif": {
            "name": "PS Function",
            "type": "psf",
            "method": "sbas",
            "unit": "coherence"
        }
    }
    
    for filename, config in layer_configs.items():
        if filename in result_files:
            file_path = result_files[filename]
            if Path(file_path).exists():
                try:
                    # Get raster bounds and statistics
                    bounds, color_range, projection = get_raster_info_fixed(file_path)
                    
                    layer = MapLayer(
                        name=config["name"],
                        type=config["type"],
                        method=config["method"], 
                        file_path=f"/api/raster/simple/{filename}",
                        bounds=bounds,
                        color_range=color_range,
                        projection=projection
                    )
                    layers.append(layer)
                    logger.info(f"Added layer: {config['name']} with bounds {bounds}")
                    
                except Exception as e:
                    logger.error(f"Error processing layer {filename}: {e}")
                    continue
    
    logger.info(f"Returning {len(layers)} layers for job {job_id}")
    return layers

async def get_existing_layers() -> List[MapLayer]:
    """Get layers from existing files in /app or backup directory"""
    layers = []
    
    # Search directories for existing files
    search_dirs = [
        Path("/app"),
        Path("/app/app_results"),
        Path("/app/processing")
    ]
    
    # Define layer configurations
    layer_configs = {
        "berlin_velocity.tif": {
            "name": "SBAS Velocity",
            "type": "velocity",
            "method": "sbas",
            "unit": "mm/year"
        },
        "berlin_ps_velocity.tif": {
            "name": "PS Velocity", 
            "type": "velocity",
            "method": "ps",
            "unit": "mm/year"
        },
        "berlin_displacement.tif": {
            "name": "SBAS Displacement",
            "type": "displacement", 
            "method": "sbas",
            "unit": "mm"
        },
        "berlin_ps_displacement.tif": {
            "name": "PS Displacement",
            "type": "displacement",
            "method": "ps", 
            "unit": "mm"
        },
        "berlin_psfunction.tif": {
            "name": "PS Function",
            "type": "psf",
            "method": "sbas",
            "unit": "coherence"
        }
    }
    
    for filename, config in layer_configs.items():
        file_found = False
        
        for search_dir in search_dirs:
            file_path = search_dir / filename
            if file_path.exists():
                try:
                    logger.info(f"Found existing file: {file_path}")
                    
                    bounds, color_range, projection = get_raster_info_fixed(str(file_path))
                    
                    layer = MapLayer(
                        name=config["name"],
                        type=config["type"],
                        method=config["method"], 
                        file_path=f"/api/raster/simple/{filename}",
                        bounds=bounds,
                        color_range=color_range,
                        projection=projection
                    )
                    layers.append(layer)
                    logger.info(f"Added existing layer: {config['name']} with bounds {bounds}")
                    file_found = True
                    break  # Stop searching once found
                    
                except Exception as e:
                    logger.error(f"Error processing existing layer {filename}: {e}")
                    continue
        
        if not file_found:
            logger.info(f"File not found: {filename}")
    
    logger.info(f"Found {len(layers)} existing layers")
    return layers

@app.get("/api/test-file/{filename}")
async def test_file_access(filename: str):
    """Test endpoint to verify file can be accessed"""
    search_dirs = [
        Path("/app"),
        Path("/app/app_results"), 
        Path("/app/processing")
    ]
    
    file_path = None
    for search_dir in search_dirs:
        candidate_path = search_dir / filename
        if candidate_path.exists():
            file_path = candidate_path
            break
    
    if not file_path:
        return {"status": "error", "message": f"File not found: {filename}"}
    
    file_stat = file_path.stat()
    
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(1024)
            
        return {
            "status": "success", 
            "file_path": str(file_path),
            "file_size": file_stat.st_size,
            "file_modified": file_stat.st_mtime,
            "first_bytes_length": len(first_bytes),
            "is_tiff": first_bytes.startswith(b'II*\x00') or first_bytes.startswith(b'MM\x00*'),
            "accessible": True
        }
    except Exception as e:
        return {"status": "error", "message": f"Cannot read file: {str(e)}"}

@app.get("/api/raster/existing-results/{filename}")
async def redirect_old_endpoint(filename: str):
    """Redirect old endpoint to simple endpoint"""
    return RedirectResponse(url=f"/api/raster/simple/{filename}", status_code=307)

@app.get("/api/raster/simple/{filename}")
async def serve_simple_raster_file(filename: str):
    """Serve GeoTIFF files without range request support for optimal performance"""
    
    logger.info(f"Serving simple raster: {filename}")
    
    # Search directories for the file
    search_dirs = [
        Path("/app"),
        Path("/app/app_results"), 
        Path("/app/processing")
    ]
    
    file_path = None
    for search_dir in search_dirs:
        candidate_path = search_dir / filename
        if candidate_path.exists():
            file_path = candidate_path
            logger.info(f"Found file: {file_path}")
            break
    
    if not file_path:
        logger.error(f"File not found in any location: {filename}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Expose-Headers": "Content-Length, Content-Type, Cache-Control, Content-Disposition",
        "Content-Type": "image/tiff",
        "Cache-Control": "public, max-age=3600",
        "Content-Disposition": f'inline; filename="{filename}"',
        # Add these for better GeoTIFF compatibility
        "X-Content-Type-Options": "nosniff",
        "Cross-Origin-Resource-Policy": "cross-origin",
    }
    
    return FileResponse(
        str(file_path),
        media_type="image/tiff",
        headers=headers,
        filename=filename
    )

@app.get("/api/comparison/{job_id}")
async def get_comparison_plot(job_id: str):
    """Serve comparison plot image"""
    
    if job_id == "existing-results":
        return await serve_existing_comparison_plot()
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    plot_path = job["results"]["files"].get("berlin_comparison_plot.png")
    if not plot_path or not Path(plot_path).exists():
        raise HTTPException(status_code=404, detail="Comparison plot not found")
    
    return FileResponse(
        plot_path, 
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
        }
    )

async def serve_existing_comparison_plot():
    """Serve existing comparison plot from various locations"""
    
    search_dirs = [
        Path("/app"),
        Path("/app/app_results"),
        Path("/app/processing")
    ]
    
    for search_dir in search_dirs:
        plot_path = search_dir / "berlin_comparison_plot.png"
        if plot_path.exists():
            logger.info(f"Serving existing comparison plot: {plot_path}")
            return FileResponse(
                str(plot_path), 
                media_type="image/png",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=3600",
                }
            )
    
    # Plot not found
    raise HTTPException(status_code=404, detail="Comparison plot not found")

# ===============================
# SEARCH & DOWNLOAD ENDPOINTS
# ===============================

@app.post("/search", response_model=SearchResponse)
async def search_sentinel_data(request: SearchRequest):
    """Search for Sentinel-1 SLC burst data"""
    try:
        if not EARTHDATA_TOKEN and (not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD):
            raise Exception("Earthdata credentials not configured. Please set EARTHDATA_TOKEN or EARTHDATA_USERNAME and EARTHDATA_PASSWORD.")

        logger.info(f"Starting burst search with parameters: {request.dict()}")

        # Generate ASF URL
        asf_url = generate_asf_url(request)
        logger.info(f"Generated ASF URL: {asf_url}")

        # Import required libraries
        try:
            import asf_search as asf
        except ImportError:
            raise Exception("asf_search library not found. Please install with: pip install asf_search")

        session = asf.ASFSession()
        if EARTHDATA_TOKEN:
            logger.info("Authenticating with Earthdata token")
            session.auth_with_token(EARTHDATA_TOKEN)
        else:
            logger.info("Authenticating with Earthdata username/password")
            session.auth_with_creds(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        
        opts = {
            'dataset': asf.DATASET.SLC_BURST,
            'start': datetime.strptime(request.startDate, "%Y-%m-%d"),
            'end': datetime.strptime(request.endDate, "%Y-%m-%d"),
        }
        
        logger.info(f"Using dataset: {asf.DATASET.SLC_BURST}")
        
        # Add bounding box
        bbox = request.boundingBox
        opts['intersectsWith'] = f"POLYGON(({bbox.west} {bbox.south}, {bbox.east} {bbox.south}, " \
                                f"{bbox.east} {bbox.north}, {bbox.west} {bbox.north}, " \
                                f"{bbox.west} {bbox.south}))"
        
        if request.orbitDirection and request.orbitDirection.lower() != "both":
            opts['flightDirection'] = request.orbitDirection.upper()
            
        if request.path is not None:
            opts['relativeOrbit'] = request.path
            
        if request.polarization is not None and request.polarization.lower() != "all":
            opts['polarization'] = request.polarization
            
        if request.subswath is not None:
            opts['subswath'] = request.subswath
            
        if request.burstID is not None:
            opts['burstID'] = request.burstID
            
        # Handle Full Burst ID filtering
        filter_by_full_burst_id = request.fullBurstID is not None
            
        # Perform search
        results = asf.search(**opts)
        logger.info(f"Found {len(results)} burst results from ASF")
        
        # Format results
        products = []
        
        for result in results:
            try:
                # Get file size properly
                size_bytes = result.properties.get('bytes', 0)
                if size_bytes == 0:
                    size_bytes = result.properties.get('size', 0)
                    if size_bytes == 0:
                        size_bytes = result.properties.get('fileSizeBytes', 0)
                
                if isinstance(size_bytes, str):
                    try:
                        size_bytes = float(size_bytes)
                    except (ValueError, TypeError):
                        size_bytes = 0
                        
                size_mb = size_bytes / (1024**2) if size_bytes else 0
                
                # Extract burst and subswath information
                burst_id = result.properties.get("burstID", "")
                subswath = result.properties.get("subswath", "")
                
                if not subswath:
                    file_id = result.properties.get("fileID", "")
                    if "IW1" in file_id:
                        subswath = "IW1"
                    elif "IW2" in file_id:
                        subswath = "IW2"
                    elif "IW3" in file_id:
                        subswath = "IW3"
                
                if not burst_id:
                    file_id = result.properties.get("fileID", "")
                    parts = file_id.split("_")
                    if len(parts) > 1:
                        burst_id = parts[1]
                
                # Handle burst index
                burst_index = 0
                full_burst_id = ""
                
                # Generate Full Burst ID
                path_number = result.properties.get("pathNumber", result.properties.get("relativeOrbit", 44))
                if burst_id and subswath and path_number:
                    full_burst_id = f"{path_number:03d}_{burst_id}_{subswath}"
                    
                    # Derive burst index from burst ID
                    try:
                        burst_num = int(burst_id)
                        last_digit = burst_num % 10
                        if last_digit == 0:
                            burst_index = 5
                        elif last_digit == 1:
                            burst_index = 7
                        elif last_digit == 9:
                            burst_index = 3
                        else:
                            burst_index = (last_digit * 2) % 10 + 1
                    except (ValueError, TypeError):
                        burst_index = 1
                
                api_full_burst_id = result.properties.get("fullBurstID") or result.properties.get("full_burst_id")
                if api_full_burst_id:
                    full_burst_id = api_full_burst_id
                
                scene_name = result.properties.get("sceneName", "")
                file_id = result.properties.get("fileID", "")
                
                if burst_id and subswath:
                    title = f"Burst-{subswath}-{burst_id}"
                elif scene_name:
                    title = scene_name
                else:
                    title = file_id
                
                # Parse date
                start_time = result.properties.get("startTime", "")
                date_str = start_time.split("T")[0] if start_time and "T" in start_time else ""
                
                # Get orbit/path information
                rel_orbit = result.properties.get("relativeOrbit", None)
                path_number = result.properties.get("pathNumber", None)
                path = rel_orbit if rel_orbit is not None else path_number if path_number is not None else 0
                
                # Create the product
                product = SentinelProduct(
                    id=result.properties.get("fileID", ""),
                    title=title,
                    date=date_str,
                    footprint=str(result.geometry),
                    size=f"{size_mb:.2f} MB",
                    thumbnailUrl=result.properties.get("browse", ""),
                    burstID=burst_id,
                    subswath=subswath,
                    burstIndex=burst_index,
                    metadata={
                        "mission": "Sentinel-1B" if "S1B" in scene_name else "Sentinel-1A",
                        "mode": result.properties.get("beamMode", "IW"),
                        "orbitDirection": result.properties.get("flightDirection", ""),
                        "path": path,
                        "absoluteOrbit": result.properties.get("absoluteOrbit", 0),
                        "startTime": start_time,
                        "stopTime": result.properties.get("stopTime", ""),
                        "url": result.properties.get("url", ""),
                        "burstID": burst_id,
                        "subswath": subswath,
                        "burstIndex": burst_index,
                        "fullBurstID": full_burst_id,
                        "polarization": result.properties.get("polarization", ""),
                        "frame": result.properties.get("frameNumber", None),
                        "burstIdentifier": file_id.split("_")[-1].split("-")[0] if "-" in file_id else ""
                    }
                )
                products.append(product)
                
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
        
        # Filter by Full Burst ID if specified
        if filter_by_full_burst_id and request.fullBurstID:
            logger.info(f"Filtering results by Full Burst ID: {request.fullBurstID}")
            filtered_products = []
            for product in products:
                if product.metadata.get("fullBurstID") == request.fullBurstID:
                    filtered_products.append(product)
            products = filtered_products
            logger.info(f"After Full Burst ID filtering: {len(products)} products remain")
        
        return SearchResponse(
            products=products,
            totalResults=len(products),
            asfUrl=asf_url
        )
        
    except Exception as e:
        logger.error(f"Error in ASF burst search: {str(e)}")
        raise HTTPException(status_code=503, detail=f"ASF search failed: {str(e)}")

@app.post("/download/{product_id}")
async def download_product(product_id: str, background_tasks: BackgroundTasks):
    """Initiate download of a Sentinel-1 burst product"""
    try:
        if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
            raise HTTPException(status_code=401, detail="Earthdata credentials not configured")
        
        # Initialize download status
        download_statuses[product_id] = DownloadStatus(
            productId=product_id,
            status="started",
            message="Download initiated",
            progress=0.0
        )
        
        # Start download in background
        background_tasks.add_task(download_file_with_asf, product_id)
        
        return download_statuses[product_id]
        
    except Exception as e:
        logger.error(f"Error initiating download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/download/status/{product_id}")
async def get_download_status(product_id: str):
    """Get download status for a product"""
    if product_id not in download_statuses:
        raise HTTPException(status_code=404, detail="Download not found")
    
    return download_statuses[product_id]

@app.get("/scan-slc")
async def scan_slc():
    """Scan for Sentinel-1 SLC data in the download directory"""
    try:
        # Check for SLC data files
        scene_count = 0
        product_ids = []
        
        if DOWNLOAD_DIR.exists():
            # Look for .SAFE directories or .zip files
            for item in DOWNLOAD_DIR.iterdir():
                if item.is_dir() and item.name.endswith('.SAFE'):
                    scene_count += 1
                    product_ids.append(item.name)
                elif item.is_file() and item.name.endswith('.zip'):
                    scene_count += 1
                    product_ids.append(item.name)
        
        has_sufficient_data = scene_count >= 2
        
        return {
            "success": True,
            "scene_count": scene_count,
            "has_sufficient_data": has_sufficient_data,
            "product_ids": product_ids,
            "download_directory": str(DOWNLOAD_DIR)
        }
    except Exception as e:
        logger.error(f"Error scanning for SLC data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ===============================
# ORIGINAL PROCESSING FUNCTIONS 
# ===============================

async def run_insar_processing(job_id: str, analysis_type: str):
    """Background task to run InSAR processing"""
    job = processing_jobs[job_id]
    
    try:
        # Update status to running
        job["status"] = "running"
        job["message"] = "Starting InSAR processing..."
        job["progress"] = 5.0
        
        processing_dir = Path("/app/processing")
        processing_dir.mkdir(exist_ok=True)
        
        try:
            os.chmod(processing_dir, 0o755)
            test_file = processing_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"Processing directory permissions verified: {processing_dir}")
        except Exception as perm_error:
            logger.warning(f"Could not fix processing directory permissions: {perm_error}")
            processing_dir = Path("/app")
        
        # Full path to script
        script_path = Path("/app/berlin_insar.py")
        if not script_path.exists():
            raise FileNotFoundError(f"InSAR processing script not found at {script_path}")
        
        python_exec = "/usr/bin/python3"
        if not Path(python_exec).exists():
            python_exec = "python3"
        
        # Start the processing
        job["message"] = "Running SBAS analysis..."
        job["progress"] = 10.0
        
        logger.info(f"Starting InSAR processing: {python_exec} {script_path}")
        
        env = dict(os.environ)
        env['INSAR_LOG_DIR'] = str(processing_dir)
        
        process = await asyncio.create_subprocess_exec(
            python_exec, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(processing_dir),
            env=env
        )
        
        # Progress tracking keywords
        progress_map = {
            "SBAS processing": 20.0,
            "Starting SBAS analysis": 25.0,
            "Computing multi-look interferograms": 30.0,
            "Phase unwrapping": 40.0,
            "Atmospheric corrections": 45.0,
            "STARTING PS ANALYSIS": 50.0,
            "Computing single-look interferograms": 55.0,
            "PS 1D phase unwrapping": 65.0,
            "Computing PS displacements": 75.0,
            "COMPARING SBAS vs PS RESULTS": 85.0,
            "EXPORTING 3D VISUALIZATION": 92.0,
            "COMPLETE PROCESSING FINISHED": 100.0
        }
        
        # Read output line by line and update progress
        output_lines = []
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    output_lines.append(line_str)
                    logger.info(f"Processing: {line_str}")
                    
                    # Update progress based on log messages
                    for keyword, progress in progress_map.items():
                        if keyword in line_str:
                            job["progress"] = min(progress, 99.0)
                            job["message"] = f"Processing: {keyword}"
                            logger.info(f"Progress updated: {progress}% - {keyword}")
                            break
                    
                    # Update generic progress based on time elapsed
                    elapsed = (datetime.now() - job["started_at"]).total_seconds()
                    if elapsed > 300:  # After 5 minutes, gradually increase
                        time_progress = min(30 + (elapsed - 300) / 60, 95)
                        if job["progress"] < time_progress:
                            job["progress"] = time_progress
                            
            except asyncio.TimeoutError:
                elapsed = (datetime.now() - job["started_at"]).total_seconds()
                if elapsed < 3600:  # Keep waiting for up to 1 hour
                    continue
                else:
                    logger.warning(f"Processing timeout after {elapsed/60:.1f} minutes")
                    break
        
        returncode = await process.wait()
        
        if returncode == 0:
            # Processing completed successfully
            job["status"] = "completed"
            job["progress"] = 100.0  
            job["message"] = "Processing completed successfully"
            job["completed_at"] = datetime.now()
            
            # Collect result files from wherever they actually are
            result_files = collect_result_files_robust(processing_dir)
            
            # Generate statistics
            stats = generate_processing_statistics(result_files)
            
            job["results"] = {
                "files": result_files,
                "statistics": stats,
                "processing_time": (job["completed_at"] - job["started_at"]).total_seconds() / 60,
                "output_log": output_lines[-50:] if len(output_lines) > 50 else output_lines
            }
            
            logger.info(f"Processing completed successfully for job {job_id}")
            
        else:
            # Processing failed
            error_output = "\n".join(output_lines[-10:]) if output_lines else "No output captured"
            job["status"] = "failed"
            job["message"] = f"Processing failed with return code {returncode}"
            logger.error(f"Processing failed for job {job_id}: {error_output}")
            
    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Processing error: {str(e)}"
        logger.error(f"Processing error for job {job_id}: {str(e)}")

def collect_result_files_robust(processing_dir: Path) -> Dict[str, str]:
    """Collect all result files with persistent storage backup"""
    result_files = {}
    
    search_dirs = [Path("/app"), processing_dir]
    
    # Persistent storage directory
    persistent_dir = Path("/app/app_results")
    persistent_dir.mkdir(exist_ok=True)
    
    # Expected result files
    expected_files = [
        "berlin_velocity.tif",
        "berlin_ps_velocity.tif", 
        "berlin_displacement.tif",
        "berlin_ps_displacement.tif",
        "berlin_psfunction.tif",
        "berlin_comparison_plot.png"
    ]
    
    for filename in expected_files:
        for search_dir in search_dirs:
            file_path = search_dir / filename
            if file_path.exists():
                # Copy to persistent storage for backup
                try:
                    persistent_path = persistent_dir / filename
                    import shutil
                    shutil.copy2(file_path, persistent_path)
                    result_files[filename] = str(file_path)
                    logger.info(f"Found and backed up: {filename} -> {persistent_path}")
                except Exception as backup_error:
                    logger.warning(f"Could not backup {filename}: {backup_error}")
                    result_files[filename] = str(file_path)
                break
    
    # Collect time series files
    for search_dir in search_dirs:
        for ts_file in search_dir.glob("berlin_*displacement_timeseries*.tif"):
            if ts_file.name not in result_files:
                try:
                    persistent_path = persistent_dir / ts_file.name
                    import shutil
                    shutil.copy2(ts_file, persistent_path)
                    result_files[ts_file.name] = str(ts_file)
                    logger.info(f"Found and backed up time series: {ts_file.name}")
                except Exception as backup_error:
                    logger.warning(f"Could not backup {ts_file.name}: {backup_error}")
                    result_files[ts_file.name] = str(ts_file)
    
    logger.info(f"Total files found: {len(result_files)}")
    return result_files

def get_raster_info_fixed(file_path: str) -> tuple:
    """Get bounds and value range from raster file - FIXED VERSION"""
    try:
        with rasterio.open(file_path) as src:
            # Get bounds in the source CRS first
            bounds_native = src.bounds
            
            # Get the CRS
            crs = src.crs
            projection = crs.to_string() if crs else "EPSG:4326"
            
            logger.info(f"Raster {file_path}: CRS={projection}, native bounds={bounds_native}")
            
            # Convert bounds to WGS84 if not already
            if crs and crs != CRS.from_epsg(4326):
                try:
                    from rasterio.warp import transform_bounds
                    bounds_wgs84 = transform_bounds(crs, CRS.from_epsg(4326), *bounds_native)
                    bounds = [bounds_wgs84[0], bounds_wgs84[1], bounds_wgs84[2], bounds_wgs84[3]]
                    logger.info(f"Transformed to WGS84: {bounds}")
                except Exception as transform_error:
                    logger.warning(f"Could not transform bounds: {transform_error}, using native bounds")
                    bounds = [bounds_native.left, bounds_native.bottom, bounds_native.right, bounds_native.top]
            else:
                bounds = [bounds_native.left, bounds_native.bottom, bounds_native.right, bounds_native.top]
            
            # Validate bounds are finite
            if not all(np.isfinite(bounds)):
                logger.warning(f"Non-finite bounds detected: {bounds}")
                bounds = [13.35, 52.47, 13.45, 52.57]
                logger.info(f"Using default Berlin bounds: {bounds}")
            
            try:
                data = src.read(1, masked=True)
                
                # Handle masked arrays and NaN values properly
                if hasattr(data, 'mask'):
                    valid_data = data.compressed()
                else:
                    valid_data = data[~np.isnan(data)]
                
                if len(valid_data) > 0:
                    # Calculate percentiles for reasonable display range
                    p2 = np.percentile(valid_data, 2)
                    p98 = np.percentile(valid_data, 98)
                    
                    # Handle infinite values
                    if not np.isfinite(p2):
                        p2 = float(np.nanmin(valid_data)) if len(valid_data) > 0 else -10.0
                    if not np.isfinite(p98):
                        p98 = float(np.nanmax(valid_data)) if len(valid_data) > 0 else 10.0
                    
                    # Ensure values are JSON-safe and reasonable
                    color_range = [
                        float(p2) if np.isfinite(p2) else -10.0,
                        float(p98) if np.isfinite(p98) else 10.0
                    ]
                    
                    # Ensure min != max
                    if abs(color_range[1] - color_range[0]) < 1e-6:
                        color_range = [color_range[0] - 1.0, color_range[1] + 1.0]
                        
                else:
                    color_range = [-10.0, 10.0]
                    
            except Exception as data_error:
                logger.warning(f"Could not read raster data for range calculation: {data_error}")
                color_range = [-10.0, 10.0]
                
            logger.info(f"Final raster info for {file_path}: bounds={bounds}, range={color_range}, projection={projection}")
            return bounds, color_range, projection
            
    except Exception as e:
        logger.error(f"Error reading raster info for {file_path}: {e}")
        # Return sensible defaults for Berlin
        return [13.35, 52.47, 13.45, 52.57], [-10.0, 10.0], "EPSG:4326"

def generate_processing_statistics(result_files: Dict[str, str]) -> Dict:
    """Generate processing statistics from result files"""
    stats = {
        "total_files": len(result_files),
        "sbas_files": len([f for f in result_files.keys() if "ps_" not in f and f.endswith(".tif")]),
        "ps_files": len([f for f in result_files.keys() if "ps_" in f and f.endswith(".tif")]),
        "time_series_files": len([f for f in result_files.keys() if "timeseries" in f])
    }
    
    # Try to extract velocity statistics
    try:
        sbas_velocity_file = result_files.get("berlin_velocity.tif")
        ps_velocity_file = result_files.get("berlin_ps_velocity.tif")
        
        if sbas_velocity_file and Path(sbas_velocity_file).exists():
            try:
                with rasterio.open(sbas_velocity_file) as src:
                    data = src.read(1, masked=True)
                    if hasattr(data, 'mask'):
                        valid_data = data.compressed()
                    else:
                        valid_data = data[~np.isnan(data)]
                    
                    if len(valid_data) > 0:
                        vmin, vmax, vmean = float(valid_data.min()), float(valid_data.max()), float(valid_data.mean())
                        if all(np.isfinite([vmin, vmax, vmean])):
                            stats["sbas_velocity_range"] = [vmin, vmax]
                            stats["sbas_velocity_mean"] = vmean
            except Exception as e:
                logger.warning(f"Could not read SBAS velocity stats: {e}")
        
        if ps_velocity_file and Path(ps_velocity_file).exists():
            try:
                with rasterio.open(ps_velocity_file) as src:
                    data = src.read(1, masked=True)
                    if hasattr(data, 'mask'):
                        valid_data = data.compressed()
                    else:
                        valid_data = data[~np.isnan(data)]
                    
                    if len(valid_data) > 0:
                        vmin, vmax, vmean = float(valid_data.min()), float(valid_data.max()), float(valid_data.mean())
                        if all(np.isfinite([vmin, vmax, vmean])):
                            stats["ps_velocity_range"] = [vmin, vmax]
                            stats["ps_velocity_mean"] = vmean
            except Exception as e:
                logger.warning(f"Could not read PS velocity stats: {e}")
                    
    except Exception as e:
        logger.error(f"Error computing statistics: {e}")
    
    return stats

if __name__ == "__main__":
    # Print credentials status on startup
    if EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
        print(f"✓ Earthdata credentials found for user: {EARTHDATA_USERNAME}")
        print(f"✓ Source: {'Config file' if EARTHDATA_USERNAME in config else 'Environment variables'}")
    else:
        print("✗ Earthdata credentials not found!")
        print("To use ASF API, set credentials in config.txt or environment variables:")
        print("Option 1: Create a config.txt file with:")
        print("EARTHDATA_USERNAME=your_username")
        print("EARTHDATA_PASSWORD=your_password")
        print("\nOption 2: Set environment variables:")
        print("export EARTHDATA_USERNAME='your_username'")
        print("export EARTHDATA_PASSWORD='your_password'")
    
    print(f"Download directory: {DOWNLOAD_DIR.absolute()}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)