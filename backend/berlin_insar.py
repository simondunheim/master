#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Berlin InSAR Processing Script - COMPLETE VERSION
Converted from working Colab notebook
Processes 30 Sentinel-1 bursts covering Berlin (Jan-Dec 2023)
Includes SBAS + PS analysis + comparison + 3D visualization exports
"""

import os
import sys
import time
import logging
import warnings
import traceback
from pathlib import Path

# Set up logging first - fix permission issue
log_file_path = '/app/berlin_insar.log'  # Write to /app instead of current directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_environment():
    """Setup environment variables and paths"""
    logger.info("Setting up environment...")
    
    # Set GMTSAR path
    path = os.environ.get('PATH', '')
    if 'GMTSAR' not in path:
        os.environ['PATH'] = path + ':/usr/local/GMTSAR/bin/'
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = [12, 4]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    logger.info("Environment setup complete")

def import_dependencies():
    """Import all required dependencies"""
    logger.info("Importing dependencies...")
    
    global xr, np, pd, gpd, json, Client, dask, S1, Stack, ASF, Tiles, XYZTiles, pv
    
    import xarray as xr
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import json
    from dask.distributed import Client
    import dask
    from pygmtsar import S1, Stack, ASF, Tiles, XYZTiles
    
    # Try to import pyvista for 3D visualization
    try:
        import pyvista as pv
        pv.set_plot_theme("document")
        logger.info("PyVista imported successfully")
    except ImportError:
        logger.warning("PyVista not available - 3D visualization exports will be skipped")
        pv = None
    
    # Set pandas options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 100)
    
    logger.info("Dependencies imported successfully")

def define_dataset():
    """Define Berlin dataset parameters"""
    logger.info("Defining Berlin dataset...")
    
    # Your complete Berlin burst dataset (30 bursts)
    bursts = """
    S1_092670_IW3_20230106T170056_VV_9886-BURST
    S1_092670_IW3_20230118T170055_VV_2D16-BURST
    S1_092670_IW3_20230130T170055_VV_14E6-BURST
    S1_092670_IW3_20230211T170055_VV_47FA-BURST
    S1_092670_IW3_20230223T170054_VV_2E24-BURST
    S1_092670_IW3_20230307T170054_VV_D567-BURST
    S1_092670_IW3_20230319T170054_VV_F01D-BURST
    S1_092670_IW3_20230331T170055_VV_79B8-BURST
    S1_092670_IW3_20230412T170055_VV_D98E-BURST
    S1_092670_IW3_20230424T170056_VV_F02A-BURST
    S1_092670_IW3_20230506T170056_VV_D8C8-BURST
    S1_092670_IW3_20230518T170057_VV_AAA4-BURST
    S1_092670_IW3_20230530T170057_VV_DEA0-BURST
    S1_092670_IW3_20230611T170058_VV_F493-BURST
    S1_092670_IW3_20230623T170059_VV_2D12-BURST
    S1_092670_IW3_20230705T170059_VV_C327-BURST
    S1_092670_IW3_20230717T170100_VV_0FFD-BURST
    S1_092670_IW3_20230729T170101_VV_1423-BURST
    S1_092670_IW3_20230810T170101_VV_7F3E-BURST
    S1_092670_IW3_20230822T170102_VV_BED0-BURST
    S1_092670_IW3_20230903T170103_VV_7C9A-BURST
    S1_092670_IW3_20230915T170103_VV_0597-BURST
    S1_092670_IW3_20230927T170103_VV_9B77-BURST
    S1_092670_IW3_20231009T170103_VV_D06D-BURST
    S1_092670_IW3_20231021T170104_VV_3288-BURST
    S1_092670_IW3_20231102T170103_VV_DE5C-BURST
    S1_092670_IW3_20231114T170103_VV_663E-BURST
    S1_092670_IW3_20231126T170102_VV_858A-BURST
    S1_092670_IW3_20231208T170102_VV_A704-BURST
    S1_092670_IW3_20231220T170101_VV_FCFF-BURST
    """
    
    burst_list = list(filter(None, bursts.split('\n')))
    burst_list = [b.strip() for b in burst_list]
    
    # Processing parameters
    config = {
        'bursts': burst_list,
        'orbit': 'A',
        'reference': '2023-08-22',  # Middle of time series
        'workdir': 'raw_berlin_A',
        'datadir': 'data_berlin_A',
        'dem': 'dem_berlin.nc',
        'temporal_baseline': 45,  # days
        'perpendicular_baseline': 120,  # meters
        'buffer': 0.05,  # ~5.5km radius around Berlin center
        'ps_corr_limit': 0.30  # PS correlation threshold
    }
    
    logger.info(f"Dataset defined: {len(burst_list)} bursts from Jan-Dec 2023")
    return config

def define_aoi():
    """Define Area of Interest - Berlin (FIXED VERSION)"""
    logger.info("Defining Berlin AOI...")
    
    # Berlin bounding box (more robust than point + buffer)
    # Approximate 10km x 10km around Berlin center
    buffer_deg = 0.05  # ~5.5km
    center_lon, center_lat = 13.404954, 52.520008
    
    berlin_geojson = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [center_lon - buffer_deg, center_lat - buffer_deg],  # SW
                [center_lon + buffer_deg, center_lat - buffer_deg],  # SE  
                [center_lon + buffer_deg, center_lat + buffer_deg],  # NE
                [center_lon - buffer_deg, center_lat + buffer_deg],  # NW
                [center_lon - buffer_deg, center_lat - buffer_deg]   # Close polygon
            ]]
        }
    }
    
    # Brandenburg Gate as POI
    brandenburg_geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Point", 
            "coordinates": [13.377704, 52.516275]
        },
        "properties": {
            "name": "Brandenburg Gate"
        }
    }
    
    # Create GeoDataFrames with explicit CRS
    aoi = gpd.GeoDataFrame.from_features([berlin_geojson], crs='EPSG:4326')
    poi = gpd.GeoDataFrame.from_features([brandenburg_geojson], crs='EPSG:4326')
    
    logger.info(f"AOI defined: Berlin bounding box")
    logger.info(f"AOI bounds: {aoi.bounds}")
    logger.info(f"AOI area: {aoi.area.iloc[0]:.6f} deg²")
    
    return aoi, poi

def setup_dask():
    """Initialize Dask client"""
    logger.info("Setting up Dask client...")
    
    try:
        client = Client(
            processes=False,  # Use threads
            threads_per_worker=2,
            n_workers=1,
            memory_limit='4GB',
            silence_logs=False,
            dashboard_address=None
        )
        
        # Configure Dask
        dask.config.set({
            'distributed.worker.memory.target': 0.5,
            'distributed.worker.memory.spill': 0.6,
            'distributed.worker.memory.pause': 0.7,
            'distributed.worker.memory.terminate': 0.85,
            'array.chunk-size': '64MB'
        })
        
        logger.info(f"Dask client initialized: {client}")
        return client
    except Exception as e:
        logger.warning(f"Could not initialize Dask client: {e}")
        return None

def check_existing_data(config):
    """Check what data already exists"""
    data_dir = Path(config['datadir'])
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        return []
    
    existing_scenes = []
    for safe_dir in data_dir.glob("*.SAFE"):
        if safe_dir.is_dir():
            existing_scenes.append(safe_dir.name)
    
    return existing_scenes

def filter_needed_bursts(config):
    """Filter bursts to only download missing ones"""
    existing_scenes = check_existing_data(config)
    
    if not existing_scenes:
        logger.info("No existing data found - will download all bursts")
        return config['bursts']
    
    logger.info(f"Found {len(existing_scenes)} existing scenes:")
    for scene in existing_scenes[:5]:  # Show first 5
        logger.info(f"  - {scene}")
    if len(existing_scenes) > 5:
        logger.info(f"  ... and {len(existing_scenes) - 5} more")
    
    # Convert burst names to expected .SAFE directory names
    needed_bursts = []
    for burst in config['bursts']:
        # Extract date and scene info from burst name
        # S1_092670_IW3_20230106T170056_VV_9886-BURST -> S1A_IW_SLC__1SDV_20230106T170056_*
        parts = burst.split('_')
        if len(parts) >= 4:
            date_part = parts[3]  # 20230106T170056
            expected_pattern = f"*{date_part}*.SAFE"
            
            # Check if this scene already exists
            matching_scenes = list(Path(config['datadir']).glob(expected_pattern))
            if not matching_scenes:
                needed_bursts.append(burst)
            else:
                logger.info(f"Skipping {burst} - already exists as {matching_scenes[0].name}")
    
    logger.info(f"Need to download {len(needed_bursts)} new bursts out of {len(config['bursts'])} total")
    return needed_bursts

def download_data(config):
    """Download Sentinel-1 data and orbits with duplicate prevention"""
    logger.info("Starting data download...")
    
    # Check existing data and filter needed bursts
    needed_bursts = filter_needed_bursts(config)
    
    if not needed_bursts:
        logger.info("All data already exists - skipping download")
    else:
        asf_username = 'simonericmoon49'
        asf_password = 'Simon4998Korfmacher?'
        
        asf = ASF(asf_username, asf_password)
        
        # Download only needed bursts
        logger.info(f"Downloading {len(needed_bursts)} new bursts...")
        logger.info("Download progress may show as 0% but files are being downloaded...")
        logger.info("Check data directory to see actual progress!")
        
        try:
            # Set environment variable to reduce joblib verbosity
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
            
            download_result = asf.download(config['datadir'], needed_bursts)
            logger.info(f"Download completed: {download_result}")
        except KeyboardInterrupt:
            logger.warning("Download interrupted by user")
            logger.info("Checking what was actually downloaded...")
            existing_after = check_existing_data(config)
            logger.info(f"Found {len(existing_after)} scenes after interruption")
        except Exception as e:
            logger.error(f"Download error: {e}")
            logger.info("Continuing with available data...")
    
    # Scan and download missing orbits
    logger.info("Scanning for scenes and downloading orbits...")
    scenes = S1.scan_slc(config['datadir'])
    logger.info(f"Found {len(scenes)} scenes for orbit download")
    
    if len(scenes) > 0:
        S1.download_orbits(config['datadir'], scenes)
        logger.info("Orbit download complete")
    else:
        logger.warning("No scenes found - cannot download orbits")
    
    # Final data summary
    final_scenes = check_existing_data(config)
    logger.info(f"Data download summary: {len(final_scenes)} scenes available for processing")
    
    return len(final_scenes) > 0

def download_dem(aoi, config):
    """Download DEM for AOI with enhanced error handling"""
    logger.info("Downloading DEM...")
    
    # Check if DEM already exists
    if Path(config['dem']).exists():
        logger.info(f"DEM file {config['dem']} already exists - skipping download")
        return None
    
    try:
        # Validate AOI before DEM download
        logger.info("Validating AOI for DEM download...")
        logger.info(f"AOI CRS: {aoi.crs}")
        logger.info(f"AOI bounds: {aoi.bounds}")
        logger.info(f"AOI geometry types: {aoi.geometry.type.unique()}")
        
        # Ensure AOI has valid geometry
        if not aoi.geometry.is_valid.all():
            logger.warning("Invalid geometry detected, attempting to fix...")
            aoi = aoi.copy()
            aoi['geometry'] = aoi.geometry.buffer(0)  # Fix invalid geometries
        
        # Validate final AOI dimensions
        bounds = aoi.bounds
        width = bounds.maxx.iloc[0] - bounds.minx.iloc[0]
        height = bounds.maxy.iloc[0] - bounds.miny.iloc[0]
        
        logger.info(f"DEM download area - Width: {width:.6f}°, Height: {height:.6f}°")
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid AOI dimensions: width={width}, height={height}")
        
        if width > 1.0 or height > 1.0:
            logger.warning(f"Large DEM area requested: {width:.3f}° x {height:.3f}°")
        
        # Download DEM with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"DEM download attempt {attempt + 1}/{max_retries}")
                dem = Tiles().download_dem_srtm(aoi, filename=config['dem'])
                logger.info(f"DEM downloaded successfully: {config['dem']}")
                return dem
                
            except Exception as e:
                logger.warning(f"DEM download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Try with smaller area
                    logger.info("Reducing AOI size for retry...")
                    reduced_buffer = config['buffer'] * (0.8 ** (attempt + 1))
                    center_lon, center_lat = 13.404954, 52.520008
                    
                    smaller_geojson = {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [center_lon - reduced_buffer, center_lat - reduced_buffer],
                                [center_lon + reduced_buffer, center_lat - reduced_buffer],
                                [center_lon + reduced_buffer, center_lat + reduced_buffer],
                                [center_lon - reduced_buffer, center_lat + reduced_buffer],
                                [center_lon - reduced_buffer, center_lat - reduced_buffer]
                            ]]
                        }
                    }
                    aoi = gpd.GeoDataFrame.from_features([smaller_geojson], crs='EPSG:4326')
                else:
                    raise
                    
    except Exception as e:
        logger.error(f"Failed to download DEM: {e}")
        
        # Fallback: create a minimal AOI and try again
        logger.info("Attempting fallback DEM download with minimal area...")
        try:
            # Create very small bounding box around Berlin center
            center_lon, center_lat = 13.404954, 52.520008
            minimal_buffer = 0.01  # ~1km
            
            minimal_geojson = {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [center_lon - minimal_buffer, center_lat - minimal_buffer],
                        [center_lon + minimal_buffer, center_lat - minimal_buffer],
                        [center_lon + minimal_buffer, center_lat + minimal_buffer],
                        [center_lon - minimal_buffer, center_lat + minimal_buffer],
                        [center_lon - minimal_buffer, center_lat - minimal_buffer]
                    ]]
                }
            }
            
            minimal_aoi = gpd.GeoDataFrame.from_features([minimal_geojson], crs='EPSG:4326')
            dem = Tiles().download_dem_srtm(minimal_aoi, filename=config['dem'])
            logger.info(f"Fallback DEM download successful: {config['dem']}")
            return dem
            
        except Exception as fallback_error:
            logger.error(f"Fallback DEM download also failed: {fallback_error}")
            raise

def check_processing_status(config):
    """Check what processing steps have already been completed"""
    status = {
        'data_downloaded': False,
        'dem_downloaded': False,
        'stack_initialized': False,
        'sbas_completed': False,
        'ps_completed': False,
        'results_exported': False
    }
    
    # Check data download
    existing_scenes = check_existing_data(config)
    status['data_downloaded'] = len(existing_scenes) >= 20  # Reasonable threshold
    
    # Check DEM
    status['dem_downloaded'] = Path(config['dem']).exists()
    
    # Check work directory
    work_dir = Path(config['workdir'])
    status['stack_initialized'] = work_dir.exists() and any(work_dir.iterdir())
    
    # Check SBAS results
    status['sbas_completed'] = (
        Path('berlin_psfunction.tif').exists() and
        Path('berlin_velocity.tif').exists() and
        Path('berlin_displacement.tif').exists()
    )
    
    # Check PS results
    status['ps_completed'] = (
        Path('berlin_ps_velocity.tif').exists() and
        Path('berlin_ps_displacement.tif').exists()
    )
    
    # Check comparison results
    status['results_exported'] = (
        status['sbas_completed'] and status['ps_completed'] and
        Path('berlin_comparison_plot.png').exists()
    )
    
    logger.info("Processing status check:")
    for step, completed in status.items():
        logger.info(f"  {step}: {'✓' if completed else '✗'}")
    
    return status

def initialize_stack(config):
    """Initialize SBAS processing stack"""
    logger.info("Initializing SBAS stack...")
    
    # Scan scenes
    scenes = S1.scan_slc(config['datadir'])
    logger.info(f"Found {len(scenes)} scenes")
    
    # Create stack
    sbas = Stack(config['workdir'], drop_if_exists=True).set_scenes(scenes).set_reference(config['reference'])
    
    logger.info(f"Stack initialized with {len(sbas.to_dataframe())} scenes")
    logger.info(f"Reference date: {config['reference']}")
    
    return sbas

def process_geometry(sbas, aoi, config):
    """Process geometric corrections with improved validation"""
    logger.info("Processing geometric corrections...")
    
    # Use AOI directly since it's already a proper polygon
    logger.info(f"AOI bounds: {aoi.bounds}")
    logger.info(f"AOI geometry type: {aoi.geometry.type.iloc[0]}")
    logger.info(f"AOI area: {aoi.area.iloc[0]:.6f} deg²")
    
    # Validate the AOI geometry
    if not aoi.geometry.is_valid.all():
        logger.error("Invalid AOI geometry!")
        raise ValueError("AOI geometry is invalid")
    
    if aoi.area.iloc[0] <= 0:
        logger.error("AOI has zero or negative area!")
        raise ValueError("AOI area must be positive")
    
    # Reframe scenes to AOI
    logger.info("Reframing scenes to AOI...")
    sbas.compute_reframe(aoi)
    
    # Load DEM with error handling
    logger.info("Loading DEM...")
    try:
        sbas.load_dem(config['dem'], aoi)
    except Exception as e:
        logger.error(f"Failed to load DEM: {e}")
        # Try to regenerate DEM with smaller area
        logger.info("Attempting to regenerate DEM with reduced buffer...")
        smaller_buffer = config['buffer'] * 0.5
        center_lon, center_lat = 13.404954, 52.520008
        
        smaller_geojson = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon", 
                "coordinates": [[
                    [center_lon - smaller_buffer, center_lat - smaller_buffer],
                    [center_lon + smaller_buffer, center_lat - smaller_buffer],
                    [center_lon + smaller_buffer, center_lat + smaller_buffer],
                    [center_lon - smaller_buffer, center_lat + smaller_buffer],
                    [center_lon - smaller_buffer, center_lat - smaller_buffer]
                ]]
            }
        }
        aoi_smaller = gpd.GeoDataFrame.from_features([smaller_geojson], crs='EPSG:4326')
        
        # Re-download DEM with smaller area
        dem = Tiles().download_dem_srtm(aoi_smaller, filename=config['dem'])
        sbas.load_dem(config['dem'], aoi_smaller)
        aoi = aoi_smaller
    
    # Align images
    logger.info("Aligning images...")
    sbas.compute_align()
    
    # Backup and cleanup
    logger.info("Creating backup and cleaning up...")
    sbas.backup('backup')
    import shutil
    if Path('backup').exists():
        shutil.rmtree('backup')
    
    # Geocoding
    logger.info("Computing geocoding transform...")
    sbas.compute_geocode(1)
    
    topo = sbas.get_topo()
    logger.info("Geometric processing complete")
    
    return sbas, topo, aoi

def compute_baselines(sbas, config):
    """Compute baseline pairs"""
    logger.info("Computing baseline pairs...")
    
    baseline_pairs = sbas.sbas_pairs(
        days=config['temporal_baseline'], 
        meters=config['perpendicular_baseline']
    )
    baseline_pairs = sbas.sbas_pairs_limit(baseline_pairs, limit=2, iterations=2)
    
    logger.info(f"Selected {len(baseline_pairs)} baseline pairs")
    return baseline_pairs

def compute_psf(sbas):
    """Compute Persistent Scatterers Function"""
    logger.info("Computing Persistent Scatterers Function...")
    
    sbas.compute_ps()
    psfunction = sbas.psfunction()
    
    logger.info("PSF computation complete")
    return psfunction

def sbas_analysis(sbas, baseline_pairs, psfunction):
    """Perform SBAS analysis"""
    logger.info("Starting SBAS analysis...")
    
    # Multi-look interferograms
    logger.info("Computing multi-look interferograms...")
    sbas.compute_interferogram_multilook(baseline_pairs, 'intf_mlook', wavelength=30, weight=psfunction)
    
    # Open stack
    ds_sbas = sbas.open_stack('intf_mlook')
    intf_sbas = ds_sbas.phase
    corr_sbas = ds_sbas.correlation
    
    logger.info(f"Generated {len(intf_sbas.pair)} interferograms")
    
    # Quality check
    logger.info("Performing quality check...")
    baseline_pairs['corr'] = corr_sbas.mean(['y', 'x'])
    baseline_pairs_best = sbas.sbas_pairs_covering_correlation(baseline_pairs, 3)
    baseline_pairs_best = sbas.sbas_pairs_limit(baseline_pairs_best, limit=2, iterations=2)
    
    # Filter to best pairs
    intf_sbas = intf_sbas.sel(pair=baseline_pairs_best.pair.values)
    corr_sbas = corr_sbas.sel(pair=baseline_pairs_best.pair.values)
    
    logger.info(f"Quality filtering: {len(baseline_pairs)} -> {len(baseline_pairs_best)} pairs")
    
    return intf_sbas, corr_sbas, baseline_pairs_best

def unwrap_phases(sbas, intf_sbas, corr_sbas):
    """Phase unwrapping"""
    logger.info("Starting phase unwrapping...")
    
    unwrap_sbas = sbas.unwrap_snaphu(intf_sbas, corr_sbas)
    unwrap_sbas = sbas.sync_cube(unwrap_sbas, 'unwrap_sbas')
    
    logger.info("Phase unwrapping complete")
    return unwrap_sbas

def correct_phases(sbas, unwrap_sbas, corr_sbas):
    """Apply atmospheric corrections"""
    logger.info("Applying atmospheric corrections...")
    
    # Trend correction
    logger.info("Computing trend correction...")
    decimator = sbas.decimator(resolution=15, grid=(1,1))
    topo = decimator(sbas.get_topo())
    inc = decimator(sbas.incidence_angle())
    yy, xx = xr.broadcast(topo.y, topo.x)
    trend_sbas = sbas.regression(unwrap_sbas.phase, [topo, topo*yy, topo*xx, topo*yy*xx, yy, xx, inc], corr_sbas)
    trend_sbas = sbas.sync_cube(trend_sbas, 'trend_sbas')
    
    # Turbulence correction
    logger.info("Computing turbulence corrections...")
    turbo_sbas = sbas.polyfit(unwrap_sbas.phase - trend_sbas, corr_sbas)
    turbo_sbas = sbas.sync_cube(turbo_sbas, 'turbo_sbas')
    
    turbo2_sbas = sbas.polyfit(unwrap_sbas.phase - trend_sbas - turbo_sbas, corr_sbas)
    turbo2_sbas = sbas.sync_cube(turbo2_sbas, 'turbo2_sbas')
    
    logger.info("Atmospheric corrections complete")
    return trend_sbas, turbo_sbas, turbo2_sbas

def compute_displacement(sbas, unwrap_sbas, trend_sbas, turbo_sbas, turbo2_sbas, corr_sbas):
    """Compute displacement time series"""
    logger.info("Computing displacement time series...")
    
    # SBAS displacement
    disp_sbas = sbas.lstsq(unwrap_sbas.phase - trend_sbas - turbo_sbas - turbo2_sbas, corr_sbas)
    disp_sbas = sbas.sync_cube(disp_sbas, 'disp_sbas')
    
    # Velocity
    velocity_sbas = sbas.velocity(disp_sbas)
    velocity_sbas = sbas.sync_cube(velocity_sbas, 'velocity_sbas')
    
    logger.info("Displacement computation complete")
    return disp_sbas, velocity_sbas

def ps_analysis(sbas, baseline_pairs, trend_sbas, turbo_sbas, turbo2_sbas, config):
    """Perform PS (Persistent Scatterer) analysis - NEW ADDITION"""
    logger.info("=" * 40)
    logger.info("STARTING PS ANALYSIS")
    logger.info("=" * 40)
    
    # Single-look interferograms for PS
    logger.info("Computing single-look interferograms...")
    sbas.compute_interferogram_singlelook(
        baseline_pairs, 'intf_slook', 
        wavelength=30, 
        weight=sbas.psfunction(),
        phase=trend_sbas + turbo_sbas + turbo2_sbas
    )
    
    # Open PS dataset
    logger.info("Opening PS interferogram stack...")
    ds_ps = sbas.open_stack('intf_slook')
    intf_ps = ds_ps.phase
    corr_ps = ds_ps.correlation
    
    logger.info("PS correlation analysis...")
    corr_ps_stack = corr_ps.mean('pair')
    corr_ps_stack = sbas.sync_cube(corr_ps_stack, 'corr_ps_stack')
    
    # 1D phase unwrapping (computationally intensive!)
    logger.info("PS 1D phase unwrapping... (this may take a while)")
    disp_ps_pairs = sbas.unwrap1d(intf_ps, corr_ps)
    disp_ps_pairs = sbas.sync_cube(disp_ps_pairs, 'disp_ps_pairs')
    
    # PS displacement computation
    logger.info("Computing PS displacements...")
    disp_ps = sbas.lstsq(disp_ps_pairs, corr_ps)
    disp_ps = sbas.sync_cube(disp_ps, 'disp_ps')
    
    # PS velocity
    logger.info("Computing PS velocity...")
    velocity_ps = sbas.velocity(disp_ps)
    velocity_ps = sbas.sync_cube(velocity_ps, 'velocity_ps')
    
    # PS RMSE analysis
    logger.info("Computing PS RMSE...")
    rmse_ps = sbas.rmse(disp_ps_pairs, disp_ps, corr_ps)
    rmse_ps = sbas.sync_cube(rmse_ps, 'rmse_ps')
    
    logger.info("PS analysis complete")
    return velocity_ps, disp_ps, rmse_ps, corr_ps_stack

def compare_sbas_ps(sbas, velocity_sbas, velocity_ps, aoi, config):
    """Compare SBAS vs PS results - NEW ADDITION"""
    logger.info("=" * 40)
    logger.info("COMPARING SBAS vs PS RESULTS")
    logger.info("=" * 40)
    
    try:
        # Convert displacements to millimeters, geocode, crop AOI
        logger.info("Preparing data for comparison...")
        points_sbas = sbas.as_geo(sbas.ra2ll(sbas.los_displacement_mm(velocity_sbas))).rio.clip(aoi.geometry)
        points_ps = sbas.as_geo(sbas.ra2ll(sbas.los_displacement_mm(velocity_ps))).rio.clip(aoi.geometry)
        points_ps = points_ps.interp_like(points_sbas, method='nearest').values.ravel()
        points_sbas = points_sbas.values.ravel()
        
        # Remove NaN values
        nanmask = np.isnan(points_sbas) | np.isnan(points_ps)
        points_sbas_clean = points_sbas[~nanmask]
        points_ps_clean = points_ps[~nanmask]
        
        logger.info(f"Comparison data points: {len(points_sbas_clean)}")
        
        # Create comparison plot
        logger.info("Creating comparison plot...")
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Scatter plot with multiple layers for visual effect
        plt.scatter(points_sbas_clean, points_ps_clean, c='silver', alpha=1,   s=1)
        plt.scatter(points_sbas_clean, points_ps_clean, c='b',      alpha=0.5, s=1)
        plt.scatter(points_sbas_clean, points_ps_clean, c='g',      alpha=0.1, s=0.5)
        plt.scatter(points_sbas_clean, points_ps_clean, c='y',      alpha=0.1, s=0.2)
        
        # Add 1:1 diagonal line
        limits_sbas = np.nanquantile(points_sbas_clean, [0.0001, 0.9999])
        limits_ps = np.nanquantile(points_ps_clean, [0.0001, 0.9999])
        plt.plot(limits_ps, limits_ps, 'k--', linewidth=2, label='1:1 Line')
        
        # Calculate correlation
        correlation = np.corrcoef(points_sbas_clean, points_ps_clean)[0, 1]
        
        plt.xlabel('Velocity SBAS, mm/year', fontsize=16)
        plt.ylabel('Velocity PS, mm/year', fontsize=16)
        plt.title(f'Cross-Comparison between SBAS and PS Velocity\nCorrelation: {correlation:.3f}', fontsize=18)
        plt.xlim(limits_sbas)
        plt.ylim(limits_ps)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save comparison plot
        plt.tight_layout()
        plt.savefig('berlin_comparison_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved comparison plot: berlin_comparison_plot.png")
        
        # Calculate and log statistics
        rmse = np.sqrt(np.mean((points_sbas_clean - points_ps_clean)**2))
        mean_diff = np.mean(points_sbas_clean - points_ps_clean)
        std_diff = np.std(points_sbas_clean - points_ps_clean)
        
        logger.info("Comparison Statistics:")
        logger.info(f"  Correlation: {correlation:.3f}")
        logger.info(f"  RMSE: {rmse:.2f} mm/year")
        logger.info(f"  Mean difference: {mean_diff:.2f} mm/year")
        logger.info(f"  Std difference: {std_diff:.2f} mm/year")
        
        return {
            'correlation': correlation,
            'rmse': rmse,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        }
        
    except Exception as e:
        logger.error(f"Comparison analysis failed: {e}")
        return None

def export_3d_visualization(sbas, velocity_sbas, velocity_ps, aoi, config):
    """Export 3D visualization files - NEW ADDITION"""
    logger.info("=" * 40)
    logger.info("EXPORTING 3D VISUALIZATION")
    logger.info("=" * 40)
    
    if pv is None:
        logger.warning("PyVista not available - skipping 3D visualization exports")
        return
    
    try:
        logger.info("Preparing 3D visualization data...")
        
        # Get DEM and prepare coordinate grids
        dem = sbas.get_dem()
        buffer_reduced = config['buffer'] / 2
        
        # Convert to lat/lon and clip to reduced AOI
        velocity_sbas_ll = sbas.los_displacement_mm(sbas.ra2ll(velocity_sbas)).rename('trend')
        velocity_ps_ll = sbas.los_displacement_mm(sbas.ra2ll(velocity_ps)).rename('trend')
        
        velocity_sbas_ll = sbas.as_geo(velocity_sbas_ll).rio.clip(aoi.geometry.buffer(-buffer_reduced).envelope)
        velocity_ps_ll = sbas.as_geo(velocity_ps_ll).rio.clip(aoi.geometry.buffer(-buffer_reduced).envelope)
        
        # Download map tiles for texture
        logger.info("Downloading map tiles for texture...")
        try:
            gmap_tiles = XYZTiles().download(velocity_sbas_ll, 14)
        except Exception as e:
            logger.warning(f"Could not download map tiles: {e}")
            # Create dummy tiles
            gmap_tiles = xr.DataArray(
                np.ones((3, velocity_sbas_ll.sizes['lat'], velocity_sbas_ll.sizes['lon']), dtype=np.uint8) * 128,
                dims=['band', 'lat', 'lon'],
                coords={'band': [0, 1, 2], 'lat': velocity_sbas_ll.lat, 'lon': velocity_sbas_ll.lon}
            )
        
        # Export SBAS 3D
        logger.info("Creating SBAS 3D visualization...")
        gmap = gmap_tiles.interp_like(velocity_sbas_ll, method='cubic').round().astype(np.uint8)
        ds_sbas = xr.merge([
            dem.interp_like(velocity_sbas_ll, method='cubic').rename('z'),
            velocity_sbas_ll, 
            gmap.transpose('band', 'lat', 'lon')
        ])
        vtk_grid_sbas = pv.StructuredGrid(sbas.as_vtk(ds_sbas.rename({'lat': 'y', 'lon': 'x'})))
        vtk_grid_sbas.save('berlin_velocity_sbas_3d.vtk')
        logger.info("Saved: berlin_velocity_sbas_3d.vtk")
        
        # Export PS 3D
        logger.info("Creating PS 3D visualization...")
        gmap = gmap_tiles.interp_like(velocity_ps_ll, method='cubic').round().astype(np.uint8)
        ds_ps = xr.merge([
            dem.interp_like(velocity_ps_ll, method='cubic').rename('z'),
            velocity_ps_ll,
            gmap.transpose('band', 'lat', 'lon')
        ])
        vtk_grid_ps = pv.StructuredGrid(sbas.as_vtk(ds_ps.rename({'lat': 'y', 'lon': 'x'})))
        vtk_grid_ps.save('berlin_velocity_ps_3d.vtk')
        logger.info("Saved: berlin_velocity_ps_3d.vtk")
        
        # Export PS points with quality filtering
        logger.info("Creating PS point cloud...")
        velocity_ll = sbas.ra2ll(velocity_ps)  # Don't filter yet
        dem_interp = dem.interp_like(velocity_ll, method='linear')
        
        # Create coordinate meshgrid
        xx, yy = np.meshgrid(velocity_ll.lon.values, velocity_ll.lat.values)
        points = np.column_stack((xx.ravel(), yy.ravel(), dem_interp.data.ravel()))
        
        # Create point cloud
        grid = pv.PolyData(points)
        grid['values'] = velocity_ll.fillna(0).data.ravel()
        grid.save('berlin_velocity_ps_points.vtk')
        logger.info("Saved: berlin_velocity_ps_points.vtk")
        
        logger.info("3D visualization exports complete")
        logger.info("Files can be opened in ParaView, MayaVi, or other VTK-compatible software")
        
    except Exception as e:
        logger.error(f"3D visualization export failed: {e}")

def export_results(sbas, psfunction, velocity_sbas, disp_sbas):
    """Export SBAS results to files"""
    logger.info("Exporting SBAS results...")
    
    try:
        # Export PSF
        sbas.export_geotiff(psfunction, 'berlin_psfunction')
        logger.info("Exported PSF GeoTIFF")
        
        # Export velocity
        sbas.export_geotiff(velocity_sbas, 'berlin_velocity')
        logger.info("Exported velocity GeoTIFF")
        
        # Export displacement (latest)
        latest_disp = disp_sbas.isel(date=-1)
        sbas.export_geotiff(latest_disp, 'berlin_displacement')
        logger.info("Exported displacement GeoTIFF")
        
        # Export time series
        logger.info("Exporting full displacement time series...")
        sbas.export_geotiff(disp_sbas, 'berlin_displacement_timeseries')
        logger.info("Exported displacement time series GeoTIFF")
        
    except Exception as e:
        logger.error(f"SBAS export error: {e}")

def export_ps_results(sbas, velocity_ps, disp_ps):
    """Export PS results - NEW ADDITION"""
    logger.info("Exporting PS results...")
    
    try:
        # Export PS velocity
        sbas.export_geotiff(velocity_ps, 'berlin_ps_velocity')
        logger.info("Exported PS velocity GeoTIFF")
        
        # Export PS displacement
        latest_ps_disp = disp_ps.isel(date=-1)
        sbas.export_geotiff(latest_ps_disp, 'berlin_ps_displacement')
        logger.info("Exported PS displacement GeoTIFF")
        
        # Export PS time series
        logger.info("Exporting PS displacement time series...")
        sbas.export_geotiff(disp_ps, 'berlin_ps_displacement_timeseries')
        logger.info("Exported PS displacement time series GeoTIFF")
        
    except Exception as e:
        logger.error(f"PS export error: {e}")

def main():
    """Main processing function with complete SBAS + PS analysis"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("BERLIN INSAR COMPLETE PROCESSING STARTED")
    logger.info("INCLUDES: SBAS + PS + COMPARISON + 3D VISUALIZATION")
    logger.info("=" * 60)
    
    try:
        # Setup
        setup_environment()
        import_dependencies()
        config = define_dataset()
        aoi, poi = define_aoi()
        
        # Check what's already been done
        status = check_processing_status(config)
        
        if status['results_exported']:
            logger.info("All processing already complete! Results found:")
            logger.info("- SBAS: berlin_*.tif")
            logger.info("- PS: berlin_ps_*.tif") 
            logger.info("- Comparison: berlin_comparison_plot.png")
            logger.info("- 3D: berlin_*_3d.vtk")
            return
        
        client = setup_dask()
        
        # Data preparation (skip if already done)
        if not status['data_downloaded']:
            download_data(config)
        else:
            logger.info("Data download already complete - skipping")
            
        if not status['dem_downloaded']:
            download_dem(aoi, config)
        else:
            logger.info("DEM already exists - skipping download")
        
        # Initialize processing (skip if work dir exists)
        if not status['stack_initialized']:
            sbas = initialize_stack(config)
            sbas, topo, aoi_buffered = process_geometry(sbas, aoi, config)
        else:
            logger.info("Work directory exists - attempting to resume...")
            sbas = Stack(config['workdir']).set_reference(config['reference'])
            # Load existing geometry processing
            topo = sbas.get_topo()
            aoi_buffered = aoi  # Use original AOI
        
        # SBAS Analysis
        if not status['sbas_completed']:
            logger.info("Starting SBAS processing...")
            baseline_pairs = compute_baselines(sbas, config)
            psfunction = compute_psf(sbas)
            intf_sbas, corr_sbas, baseline_pairs_best = sbas_analysis(sbas, baseline_pairs, psfunction)
            unwrap_sbas = unwrap_phases(sbas, intf_sbas, corr_sbas)
            trend_sbas, turbo_sbas, turbo2_sbas = correct_phases(sbas, unwrap_sbas, corr_sbas)
            disp_sbas, velocity_sbas = compute_displacement(sbas, unwrap_sbas, trend_sbas, turbo_sbas, turbo2_sbas, corr_sbas)
            
            # Export SBAS results
            export_results(sbas, psfunction, velocity_sbas, disp_sbas)
        else:
            logger.info("SBAS results already exist - loading for PS analysis...")
            # Load existing results
            baseline_pairs_best = compute_baselines(sbas, config)
            psfunction = sbas.psfunction()
            
            # Load atmospheric corrections from disk
            trend_sbas = sbas.open_cube('trend_sbas')
            turbo_sbas = sbas.open_cube('turbo_sbas')
            turbo2_sbas = sbas.open_cube('turbo2_sbas')
            velocity_sbas = sbas.open_cube('velocity_sbas')
            disp_sbas = sbas.open_cube('disp_sbas')
        
        # PS Analysis (NEW)
        if not status['ps_completed']:
            velocity_ps, disp_ps, rmse_ps, corr_ps_stack = ps_analysis(
                sbas, baseline_pairs_best, trend_sbas, turbo_sbas, turbo2_sbas, config
            )
            
            # Export PS results
            export_ps_results(sbas, velocity_ps, disp_ps)
        else:
            logger.info("PS results already exist - loading for comparison...")
            velocity_ps = sbas.open_cube('velocity_ps')
            disp_ps = sbas.open_cube('disp_ps')
        
        # Comparison Analysis (NEW)
        logger.info("Performing SBAS vs PS comparison...")
        comparison_stats = compare_sbas_ps(sbas, velocity_sbas, velocity_ps, aoi_buffered, config)
        
        # 3D Visualization Export (NEW)
        logger.info("Exporting 3D visualization files...")
        export_3d_visualization(sbas, velocity_sbas, velocity_ps, aoi_buffered, config)
        
        # Success
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"BERLIN INSAR COMPLETE PROCESSING FINISHED!")
        logger.info(f"Total processing time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
        logger.info("")
        logger.info("RESULTS SUMMARY:")
        logger.info("SBAS Results:")
        logger.info("  - berlin_psfunction.tif")
        logger.info("  - berlin_velocity.tif")
        logger.info("  - berlin_displacement.tif")
        logger.info("  - berlin_displacement_timeseries.*.tif")
        logger.info("")
        logger.info("PS Results:")
        logger.info("  - berlin_ps_velocity.tif")
        logger.info("  - berlin_ps_displacement.tif")
        logger.info("  - berlin_ps_displacement_timeseries.*.tif")
        logger.info("")
        logger.info("Comparison & Visualization:")
        logger.info("  - berlin_comparison_plot.png")
        logger.info("  - berlin_velocity_sbas_3d.vtk")
        logger.info("  - berlin_velocity_ps_3d.vtk")
        logger.info("  - berlin_velocity_ps_points.vtk")
        
        if comparison_stats:
            logger.info("")
            logger.info("SBAS vs PS Comparison:")
            logger.info(f"  Correlation: {comparison_stats['correlation']:.3f}")
            logger.info(f"  RMSE: {comparison_stats['rmse']:.2f} mm/year")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("PROCESSING FAILED!")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        sys.exit(1)
    
    finally:
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    main()