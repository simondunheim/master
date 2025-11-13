import asf_search as asf
import logging
import os
from datetime import datetime, timedelta

# Set up logging
asf_logger = logging.getLogger("asf_search")
formatter = logging.Formatter('[ %(asctime)s (%(name)s) %(filename)s:%(lineno)d ] %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
asf_logger.addHandler(stream_handler)
asf_logger.setLevel(logging.INFO)  # Set to INFO to see search and download progress

def search_and_download_sentinel1_slc(bbox, start_date, end_date, output_dir, 
                                     polarization='VV', beam_mode=None, 
                                     max_results=25, username=None, password=None):
    """
    Search and download Sentinel-1 SLC data for a specified area and time range
    
    Parameters:
    -----------
    bbox : tuple
        Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
    start_date : str or datetime
        Start date for search (YYYY-MM-DD)
    end_date : str or datetime
        End date for search (YYYY-MM-DD)
    output_dir : str
        Directory to save downloaded data
    polarization : str or list, optional
        Polarization mode(s) to search for (e.g., 'VV', ['VV', 'VH'])
    beam_mode : str or list, optional
        Beam mode(s) to search for (e.g., 'IW', 'EW', 'SM')
    max_results : int, optional
        Maximum number of results to return
    username : str, optional
        NASA Earthdata Login username
    password : str, optional
        NASA Earthdata Login password
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert bbox to WKT
    min_lon, min_lat, max_lon, max_lat = bbox
    wkt = f'POLYGON(({min_lon} {min_lat}, {min_lon} {max_lat}, {max_lon} {max_lat}, {max_lon} {min_lat}, {min_lon} {min_lat}))'
    
    # Set up search parameters
    search_params = {
        'platform': asf.PLATFORM.SENTINEL1,
        'processingLevel': 'SLC',  # Single Look Complex
        'intersectsWith': wkt,
        'start': start_date,
        'end': end_date,
        'maxResults': max_results
    }
    
    # Add optional parameters if provided
    if polarization:
        search_params['polarization'] = polarization
    if beam_mode:
        search_params['beamMode'] = beam_mode
    
    # Perform the search
    print(f"Searching for Sentinel-1 SLC data...")
    print(f"Search area: {wkt}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Polarization: {polarization}")
    print(f"Beam mode: {beam_mode}")
    
    try:
        results = asf.geo_search(**search_params)
        
        # Print search results
        print(f"Found {len(results)} results matching criteria")
        for i, result in enumerate(results[:10]):  # Print first 10 results
            print(f"{i+1}. {result.properties['sceneName']} - {result.properties['startTime']}")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")
        
        # Download results if any found
        if len(results) > 0:
            if username and password:
                # Create authenticated session
                session = asf.ASFSession()
                session.auth_with_creds(username, password)
                
                # Download with multiple processes for speed
                print(f"Downloading {len(results)} files to {output_dir}...")
                results.download(path=output_dir, session=session, processes=4)
                print("Download completed!")
            else:
                print("NASA Earthdata Login credentials required for downloading.")
                print("Please provide username and password or authenticate separately.")
        
        return results
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []

if __name__ == "__main__":
    # Define multiple test areas
    # San Francisco Bay Area (slightly expanded)
    sf_bay_area_bbox = (-123.0, 37.5, -122.0, 38.0)
    
    # Los Angeles area
    la_area_bbox = (-119.0, 33.7, -118.0, 34.2)
    
    # Seattle area
    seattle_area_bbox = (-122.5, 47.4, -122.0, 47.8)
    
    # Choose which area to search
    search_bbox = sf_bay_area_bbox
    
    # Time range - last 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Format dates as strings
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Output directory
    output_dir = "./sentinel1_slc_data"
    
    # Your NASA Earthdata Login credentials
    # You need to register at https://urs.earthdata.nasa.gov/
    username = "simonericmoon49"  # Replace with your username
    password = "Simon4998Korfmacher?"  # Replace with your password
    
    # Try various parameter combinations if initial search returns no results
    search_params = [
        {"polarization": "VV", "beam_mode": "IW"},  # Most common for land
        {"polarization": "VV", "beam_mode": None},  # Any beam mode with VV
        {"polarization": ["VV", "VH"], "beam_mode": "IW"},  # Both polarizations
        {"polarization": None, "beam_mode": None}   # Any polarization, any beam mode
    ]
    
    # Try each parameter set until we get results
    for params in search_params:
        print("\n" + "="*50)
        print(f"Trying search with parameters: {params}")
        print("="*50)
        
        results = search_and_download_sentinel1_slc(
            search_bbox, 
            start_str, 
            end_str, 
            output_dir,
            polarization=params["polarization"],
            beam_mode=params["beam_mode"],
            username=username,
            password=password
        )
        
        if len(results) > 0:
            print(f"Success! Found {len(results)} results.")
            
            # Display details of the first result
            if len(results) > 0:
                first_result = results[0]
                print("\nFirst result details:")
                properties = first_result.properties
                
                # Display all available properties
                print("\nAvailable properties:")
                for key in sorted(properties.keys()):
                    print(f"{key}: {properties[key]}")
                
                # Break out of the loop since we found results
                break
        else:
            print("No results found with these parameters. Trying next set...")