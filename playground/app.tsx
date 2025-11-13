import React, { useEffect, useRef, useState } from 'react';
import './App.css';
import 'ol/ol.css';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import { Vector as VectorSource } from 'ol/source';
import { Vector as VectorLayer } from 'ol/layer';
import Draw from 'ol/interaction/Draw';
import { transformExtent } from 'ol/proj';
import { createBox } from 'ol/interaction/Draw';
import { Style, Stroke, Fill } from 'ol/style';

function App() {
  const mapRef = useRef();
  const [map, setMap] = useState(null);
  const [boundingBox, setBoundingBox] = useState(null);
  const [drawInteraction, setDrawInteraction] = useState(null);
  const [vectorSource] = useState(new VectorSource());
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [availablePaths, setAvailablePaths] = useState([]);
  const [availableSubswaths, setAvailableSubswaths] = useState([]);
  const [downloadStatuses, setDownloadStatuses] = useState({});
  const [selectedProducts, setSelectedProducts] = useState([]);
  const [isDownloadingSelection, setIsDownloadingSelection] = useState(false);
  const [downloadedProducts, setDownloadedProducts] = useState([]);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [asfUrl, setAsfUrl] = useState(''); // Add state for ASF URL
  const [processingParams, setProcessingParams] = useState({
    orbit: 'A',
    referenceDate: '',
    demType: 'SRTM'
  });

  useEffect(() => {
    const checkExistingSLCData = async () => {
      try {
        const response = await fetch('http://localhost:8000/scan-slc');
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.has_sufficient_data) {
            console.log(`Found ${data.scene_count} existing SLC scenes`);
            
            // Update downloadedProducts state with product IDs from backend
            if (data.product_ids && data.product_ids.length > 0) {
              setDownloadedProducts(data.product_ids);
            } else {
              // If no specific product IDs but data exists, use placeholder
              setDownloadedProducts(['existing_data']);
            }
          }
        }
      } catch (error) {
        console.error('Error checking existing SLC data:', error);
      }
    };
    
    checkExistingSLCData();
  }, []);

  // Initialize map when component mounts
  useEffect(() => {
    // Vector layer for drawing bounding box
    const vectorLayer = new VectorLayer({
      source: vectorSource,
      style: new Style({
        fill: new Fill({
          color: 'rgba(255, 255, 255, 0.2)'
        }),
        stroke: new Stroke({
          color: '#ffcc33',
          width: 2
        })
      })
    });
    
    // Initialize OpenLayers map
    const initialMap = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM()
        }),
        vectorLayer
      ],
      view: new View({
        center: [0, 0],
        zoom: 2
      })
    });
    
    setMap(initialMap);
    
    // Clean up on unmount
    return () => {
      if (initialMap) {
        initialMap.setTarget(null);
      }
    };
  }, [vectorSource]);

  // Effect to poll for processing status
  useEffect(() => {
    let interval;
    
    if (jobId && (processingStatus?.status === 'processing' || processingStatus?.status === 'starting')) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:8000/insar/status/${jobId}`);
          if (response.ok) {
            const statusData = await response.json();
            setProcessingStatus(statusData);
            
            // Stop polling if processing is completed or failed
            if (statusData.status === 'completed' || statusData.status === 'failed') {
              clearInterval(interval);
            }
          }
        } catch (error) {
          console.error('Error fetching processing status:', error);
        }
      }, 5000); // Poll every 5 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, processingStatus]);

  // Function to enable drawing mode
  const enableDrawMode = () => {
    if (map) {
      // Remove any existing draw interaction
      if (drawInteraction) {
        map.removeInteraction(drawInteraction);
      }
      
      // Clear any existing drawings
      vectorSource.clear();
      setBoundingBox(null);
      
      // Create new draw interaction for rectangles
      const draw = new Draw({
        source: vectorSource,
        type: 'Circle',
        geometryFunction: createBox()
      });
      
      // When drawing ends, get the bounding box coordinates
      draw.on('drawend', (event) => {
        const feature = event.feature;
        const geometry = feature.getGeometry();
        const extent = geometry.getExtent();
        
        // Convert from EPSG:3857 (Web Mercator) to EPSG:4326 (WGS84 - lat/lon)
        const wgs84Extent = transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
        
        // Format as [west, south, east, north]
        const formattedBBox = {
          west: wgs84Extent[0].toFixed(6),
          south: wgs84Extent[1].toFixed(6),
          east: wgs84Extent[2].toFixed(6),
          north: wgs84Extent[3].toFixed(6)
        };
        
        setBoundingBox(formattedBBox);
        
        // Remove draw interaction after drawing is complete
        map.removeInteraction(draw);
        setDrawInteraction(null);
      });
      
      // Add the interaction to the map
      map.addInteraction(draw);
      setDrawInteraction(draw);
    }
  };

  // Function to clear the drawing
  const clearBoundingBox = () => {
    if (map) {
      vectorSource.clear();
      setBoundingBox(null);
      setAsfUrl(''); // Clear ASF URL when clearing bounding box
      
      if (drawInteraction) {
        map.removeInteraction(drawInteraction);
        setDrawInteraction(null);
      }
    }
  };

  // Function to search for Sentinel data
  const searchSentinelData = async () => {
    if (!boundingBox) {
      alert('Please draw a bounding box first');
      return;
    }
    
    // Get values from form
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const orbit = document.getElementById('orbit').value;
    const polarization = document.getElementById('polarization').value;
    const path = document.getElementById('path').value;
    const subswath = document.getElementById('subswath').value;
    const fullBurstID = document.getElementById('fullBurstID').value;
    
    setIsSearching(true);
    
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          boundingBox: boundingBox,
          startDate: startDate,
          endDate: endDate,
          productType: 'SLC',
          orbitDirection: orbit !== 'both' ? orbit.toUpperCase() : null,
          polarization: polarization !== 'all' ? polarization.toUpperCase() : null,
          path: path ? parseInt(path) : null,
          subswath: subswath !== 'all' ? subswath : null,
          fullBurstID: fullBurstID || null
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Search results:', data);
        setSearchResults(data.products || []);
        setAsfUrl(data.asfUrl || ''); // Set ASF URL from response
        
        // Extract unique paths and subswaths from results
        if (data.products && data.products.length > 0) {
          const paths = [...new Set(data.products.map(p => p.metadata.path))].sort((a, b) => a - b);
          const subswaths = [...new Set(data.products.map(p => p.metadata.subswath))].filter(Boolean);
          setAvailablePaths(paths);
          setAvailableSubswaths(subswaths);
        }
      } else {
        console.error('Search failed:', response.statusText);
        alert('Failed to search for Sentinel data. Please try again.');
      }
    } catch (error) {
      console.error('Error searching for Sentinel burst data:', error);
      alert('An error occurred while searching for Sentinel burst data.');
    } finally {
      setIsSearching(false);
    }
  };

  // Function to download a single product
  const downloadProduct = async (productId) => {
    try {
      // Initialize download status
      setDownloadStatuses(prev => ({
        ...prev,
        [productId]: { status: 'starting', message: 'Starting download...' }
      }));

      const response = await fetch(`http://localhost:8000/download/${productId}`, {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        setDownloadStatuses(prev => ({
          ...prev,
          [productId]: data
        }));

        // Poll for download status
        const pollInterval = setInterval(async () => {
          const statusResponse = await fetch(`http://localhost:8000/download/status/${productId}`);
          if (statusResponse.ok) {
            const statusData = await statusResponse.json();
            setDownloadStatuses(prev => ({
              ...prev,
              [productId]: statusData
            }));

            // Stop polling if download is completed or failed
            if (statusData.status === 'completed' || statusData.status === 'failed') {
              clearInterval(pollInterval);
              // Add to downloaded products if completed
              if (statusData.status === 'completed') {
                setDownloadedProducts(prev => [...prev.filter(id => id !== productId), productId]);
              }
            }
          }
        }, 2000); // Poll every 2 seconds
      } else {
        throw new Error('Failed to start download');
      }
    } catch (error) {
      console.error('Error downloading product:', error);
      setDownloadStatuses(prev => ({
        ...prev,
        [productId]: { status: 'failed', message: error.message }
      }));
    }
  };

  // Function to toggle product selection
  const toggleProductSelection = (productId) => {
    setSelectedProducts(prev => {
      if (prev.includes(productId)) {
        return prev.filter(id => id !== productId);
      } else {
        return [...prev, productId];
      }
    });
  };

  // Function to select all products
  const selectAllProducts = () => {
    const allProductIds = searchResults.map(product => product.id);
    setSelectedProducts(allProductIds);
  };

  // Function to deselect all products
  const deselectAllProducts = () => {
    setSelectedProducts([]);
  };

  // Function to download multiple products
  const downloadSelectedProducts = async () => {
    if (selectedProducts.length === 0) {
      alert('Please select products to download');
      return;
    }

    setIsDownloadingSelection(true);

    // Start downloads one by one
    for (const productId of selectedProducts) {
      try {
        // Initialize download status if it doesn't exist
        if (!downloadStatuses[productId]) {
          setDownloadStatuses(prev => ({
            ...prev,
            [productId]: { status: 'queued', message: 'Queued for download...' }
          }));
        }

        // Initiate download
        const response = await fetch(`http://localhost:8000/download/${productId}`, {
          method: 'POST',
        });

        if (response.ok) {
          const data = await response.json();
          setDownloadStatuses(prev => ({
            ...prev,
            [productId]: data
          }));
        } else {
          throw new Error('Failed to start download');
        }
      } catch (error) {
        console.error(`Error downloading product ${productId}:`, error);
        setDownloadStatuses(prev => ({
          ...prev,
          [productId]: { status: 'failed', message: error.message }
        }));
      }
    }

    setIsDownloadingSelection(false);

    // Set up polling for all downloads
    const pollInterval = setInterval(async () => {
      let allCompleted = true;

      for (const productId of selectedProducts) {
        if (
          !downloadStatuses[productId] || 
          (downloadStatuses[productId].status !== 'completed' && 
           downloadStatuses[productId].status !== 'failed')
        ) {
          try {
            const statusResponse = await fetch(`http://localhost:8000/download/status/${productId}`);
            if (statusResponse.ok) {
              const statusData = await statusResponse.json();
              setDownloadStatuses(prev => ({
                ...prev,
                [productId]: statusData
              }));

              if (statusData.status !== 'completed' && statusData.status !== 'failed') {
                allCompleted = false;
              } else if (statusData.status === 'completed') {
                // Add to downloaded products if completed
                setDownloadedProducts(prev => [...prev.filter(id => id !== productId), productId]);
              }
            }
          } catch (error) {
            console.error(`Error checking status for ${productId}:`, error);
          }
        }
      }

      // Stop polling if all downloads are completed or failed
      if (allCompleted) {
        clearInterval(pollInterval);
      }
    }, 2000); // Poll every 2 seconds
  };

  // Handle InSAR parameter changes
  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setProcessingParams(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Start InSAR processing
  const startProcessing = async () => {
    if (!boundingBox || downloadedProducts.length === 0) {
      alert('Please define a bounding box and download data first');
      return;
    }
    
    try {
      setIsProcessing(true);
      
      // Convert bounding box to GeoJSON
      const aoiGeoJson = {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [[
            [parseFloat(boundingBox.west), parseFloat(boundingBox.south)],
            [parseFloat(boundingBox.east), parseFloat(boundingBox.south)],
            [parseFloat(boundingBox.east), parseFloat(boundingBox.north)],
            [parseFloat(boundingBox.west), parseFloat(boundingBox.north)],
            [parseFloat(boundingBox.west), parseFloat(boundingBox.south)]
          ]]
        }
      };
      
      const response = await fetch('http://localhost:8000/insar/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          aoi: JSON.stringify(aoiGeoJson),
          parameters: processingParams
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setJobId(data.job_id);
        setProcessingStatus({ status: 'started', message: 'Processing started' });
      } else {
        throw new Error('Failed to start InSAR processing');
      }
    } catch (error) {
      console.error('Error starting InSAR processing:', error);
      alert('Failed to start InSAR processing. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sentinel-1 InSAR Processing Tool</h1>
      </header>
      <main>
        <section className="map-container">
          <div className="map-controls">
            <button onClick={enableDrawMode}>Draw Bounding Box</button>
            <button onClick={clearBoundingBox}>Clear</button>
          </div>
          <div ref={mapRef} className="map"></div>
        </section>
        <section className="content-panel">
          <h2>Search Parameters</h2>
          
          {boundingBox && (
            <div className="bbox-info">
              <h3>Bounding Box</h3>
              <p>West: {boundingBox.west}°</p>
              <p>South: {boundingBox.south}°</p>
              <p>East: {boundingBox.east}°</p>
              <p>North: {boundingBox.north}°</p>
            </div>
          )}
          
          <div className="search-form">
            <div className="form-group">
              <label htmlFor="start-date">Start Date:</label>
              <input type="date" id="start-date" defaultValue="2023-01-01" />
            </div>
            
            <div className="form-group">
              <label htmlFor="end-date">End Date:</label>
              <input type="date" id="end-date" defaultValue="2023-12-31" />
            </div>
            
            <div className="form-group">
              <label htmlFor="orbit">Orbit Direction:</label>
              <select id="orbit">
                <option value="both">Both</option>
                <option value="ascending">Ascending</option>
                <option value="descending">Descending</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="polarization">Polarization:</label>
              <select id="polarization">
                <option value="all">All</option>
                <option value="vv">VV</option>
                <option value="vh">VH</option>
                <option value="hh">HH</option>
                <option value="hv">HV</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="path">Path (Relative Orbit):</label>
              <input 
                type="number" 
                id="path" 
                placeholder="Optional - filter by path"
                min="1"
                max="175"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="subswath">Subswath:</label>
              <select id="subswath">
                <option value="all">All</option>
                <option value="IW1">IW1</option>
                <option value="IW2">IW2</option>
                <option value="IW3">IW3</option>
              </select>
            </div>
            
            <div className="form-group">
              <label htmlFor="fullBurstID">Full Burst ID:</label>
              <input 
                type="text" 
                id="fullBurstID" 
                placeholder="e.g., 044_092670_IW3"
                title="Filter by specific Full Burst ID (format: path_burstID_subswath)"
              />
            </div>
            
            <button 
              onClick={searchSentinelData}
              disabled={!boundingBox}
              className="search-button"
            >
              Search Available Burst Data
            </button>
          </div>
          
          {/* ASF URL Display */}
          {asfUrl && (
            <div className="asf-url-section">
              <h3>View in ASF Data Search</h3>
              <div className="asf-url-container">
                <p>View the same search parameters in the Alaska Satellite Facility data search tool:</p>
                <a 
                  href={asfUrl} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="asf-url-link"
                >
                  Open in ASF Data Search Tool
                </a>
                <details className="asf-url-details">
                  <summary>Show URL</summary>
                  <div className="asf-url-text">
                    <input 
                      type="text" 
                      value={asfUrl} 
                      readOnly 
                      className="asf-url-input"
                      onClick={(e) => e.target.select()}
                    />
                  </div>
                </details>
              </div>
            </div>
          )}
          
          {availablePaths.length > 0 && (
            <div className="available-filters">
              <h3>Available Paths in Results</h3>
              <p>{availablePaths.join(', ')}</p>
            </div>
          )}
          
          {availableSubswaths.length > 0 && (
            <div className="available-filters">
              <h3>Available Subswaths in Results</h3>
              <p>{availableSubswaths.join(', ')}</p>
            </div>
          )}
          
          <h3>Search Results {searchResults.length > 0 && `(${searchResults.length})`}</h3>
          
          <div className="results-container">
            {isSearching ? (
              <p>Searching for available burst data...</p>
            ) : searchResults.length > 0 ? (
              <>
                <div className="selection-controls">
                  <button onClick={selectAllProducts} className="select-button">Select All</button>
                  <button onClick={deselectAllProducts} className="select-button">Deselect All</button>
                  <button 
                    onClick={downloadSelectedProducts} 
                    disabled={selectedProducts.length === 0 || isDownloadingSelection}
                    className="download-selection-button"
                  >
                    {isDownloadingSelection ? 'Downloading...' : `Download Selected (${selectedProducts.length})`}
                  </button>
                </div>
                <ul className="results-list">
                  {searchResults.map((product) => (
                    <li key={product.id} className={`result-item ${selectedProducts.includes(product.id) ? 'selected' : ''}`}>
                      <div className="result-header">
                        <label className="select-checkbox-label">
                          <input 
                            type="checkbox" 
                            checked={selectedProducts.includes(product.id)}
                            onChange={() => toggleProductSelection(product.id)}
                            className="select-checkbox"
                          />
                          <h4>{product.title}</h4>
                        </label>
                        <span className="result-date">{product.date}</span>
                      </div>
                      <div className="result-details">
                        <p><strong>ID:</strong> {product.id}</p>
                        <p><strong>Size:</strong> {product.size}</p>
                        <p><strong>Polarization:</strong> {product.metadata.polarization}</p>
                        <p><strong>Orbit:</strong> {product.metadata.orbitDirection}</p>
                        <p><strong>Path (Relative Orbit):</strong> {product.metadata.path}</p>
                        <p><strong>Subswath:</strong> {product.metadata.subswath || product.subswath || (product.id.includes("IW1") ? "IW1" : product.id.includes("IW2") ? "IW2" : product.id.includes("IW3") ? "IW3" : "")}</p>
                        <p><strong>Burst ID:</strong> {product.metadata.burstID || product.burstID || product.id.split("_")[1] || ""}</p>
                        <p><strong>Burst Index:</strong> {product.metadata.burstIndex || product.burstIndex}</p>
                        <p><strong>Full Burst ID:</strong> {product.metadata.fullBurstID || ""}</p>
                        {product.metadata.burstIdentifier && <p><strong>Burst Identifier:</strong> {product.metadata.burstIdentifier}</p>}
                      </div>
                      <div className="download-section">
                        <button 
                          className="download-button"
                          onClick={() => downloadProduct(product.id)}
                          disabled={downloadStatuses[product.id]?.status === 'downloading'}
                        >
                          {downloadStatuses[product.id]?.status === 'downloading' ? 'Downloading...' : 'Download'}
                        </button>
                        {downloadStatuses[product.id] && (
                          <div className={`download-status ${downloadStatuses[product.id].status}`}>
                            <p>Status: {downloadStatuses[product.id].status}</p>
                            <p>{downloadStatuses[product.id].message}</p>
                          </div>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>No results to display. Draw a bounding box and search to see results.</p>
            )}
          </div>
          
          {/* InSAR Processing Panel */}
          {downloadedProducts.length > 0 && (
            <div className="insar-processing-panel">
              <h2>InSAR Processing</h2>
              
              <div className="processing-form">
                <div className="form-group">
                  <label htmlFor="orbit">Orbit Direction:</label>
                  <select 
                    id="orbit-param"
                    name="orbit"
                    value={processingParams.orbit}
                    onChange={handleParamChange}
                  >
                    <option value="A">Ascending</option>
                    <option value="D">Descending</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label htmlFor="referenceDate">Reference Date (YYYY-MM-DD):</label>
                  <input 
                    type="date"
                    id="referenceDate"
                    name="referenceDate"
                    value={processingParams.referenceDate}
                    onChange={handleParamChange}
                    placeholder="Optional - automatically select if not provided"
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="demType">DEM Type:</label>
                  <select 
                    id="demType"
                    name="demType"
                    value={processingParams.demType}
                    onChange={handleParamChange}
                  >
                    <option value="SRTM">SRTM 1-arc second</option>
                    <option value="COPERNICUS">Copernicus 30m</option>
                  </select>
                </div>
                
                <button 
                  onClick={startProcessing}
                  disabled={isProcessing || !boundingBox || downloadedProducts.length === 0}
                  className="process-button"
                >
                  Start InSAR Processing
                </button>
              </div>
              
              {processingStatus && (
                <div className={`processing-status ${processingStatus.status}`}>
                  <h3>Processing Status</h3>
                  <p><strong>Status:</strong> {processingStatus.status}</p>
                  <p><strong>Message:</strong> {processingStatus.message}</p>
                  
                  {processingStatus.progress !== undefined && (
                    <div className="progress-bar">
                      <div 
                        className="progress-fill"
                        style={{ width: `${processingStatus.progress}%` }}
                      ></div>
                      <span>{processingStatus.progress.toFixed(1)}%</span>
                    </div>
                  )}
                  
                  {processingStatus.status === 'completed' && processingStatus.results && (
                    <div className="results-section">
                      <h3>Results</h3>
                      <div className="result-links">
                        <a 
                          href={`http://localhost:8000${processingStatus.results.velocity_tiff}`}
                          download
                          className="result-link"
                        >
                          Download Velocity GeoTIFF
                        </a>
                        <div className="result-preview">
                          <h4>Velocity Map</h4>
                          <img 
                            src={`http://localhost:8000${processingStatus.results.velocity_png}`}
                            alt="LOS Velocity Map"
                            className="result-image"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;