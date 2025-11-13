import React, { useState, useEffect, useRef } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import { fromLonLat, toLonLat } from 'ol/proj';
import 'ol/ol.css';
import './App.css';

// TypeScript interfaces
interface ProcessingStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  started_at?: string;
  completed_at?: string;
  results?: ProcessingResults;
}

interface MapLayer {
  name: string;
  type: string;
  method: string;
  file_path: string;
  bounds: number[];
  color_range: number[];
  projection?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const App: React.FC = () => {
  // State management
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [layers, setLayers] = useState<MapLayer[]>([]);
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [debugInfo, setDebugInfo] = useState<string[]>([]);

  // Map references
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);

  // Berlin coordinates
  const BERLIN_CENTER = fromLonLat([13.404954, 52.520008]);

  // Enhanced debug logging
  const addDebugInfo = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const debugMessage = `${timestamp}: ${message}`;
    console.log(`[DEBUG] ${debugMessage}`);
    setDebugInfo(prev => [...prev.slice(-4), debugMessage]);
  };

  // Enhanced fetch function
  const apiFetch = async (url: string, options: RequestInit = {}) => {
    try {
      addDebugInfo(`Fetching: ${url}`);
      
      const response = await fetch(url, {
        ...options,
        mode: 'cors',
        credentials: 'omit',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          ...options.headers,
        },
      });

      addDebugInfo(`Response status: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        addDebugInfo(`JSON response received: ${Object.keys(data).join(', ')}`);
        return data;
      } else {
        return await response.text();
      }
    } catch (err) {
      addDebugInfo(`Fetch error: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('API fetch error:', err);
      throw err;
    }
  };

  // Check backend connection on startup and auto-load results
  useEffect(() => {
    const checkConnection = async () => {
      try {
        addDebugInfo(`Checking connection to: ${API_BASE_URL}`);
        const response = await apiFetch(`${API_BASE_URL}/health`);
        
        if (response && typeof response === 'object' && response.status === 'healthy') {
          setConnectionStatus('connected');
          setError(null);
          addDebugInfo('Backend connected successfully');
          
          // Check for existing results first
          await checkForExistingResults();
          
          // Also check script and data status
          try {
            const scriptResult = await apiFetch(`${API_BASE_URL}/api/test-script`);
            const dataResult = await apiFetch(`${API_BASE_URL}/api/check-data`);
            
            addDebugInfo(`Script check: ${scriptResult.status}`);
            addDebugInfo(`Data check: ${dataResult.data_directory?.scene_count || 0} scenes`);
            
            if (scriptResult.status === 'error') {
              setError(`Script issue: ${scriptResult.message}`);
            } else if (dataResult.data_directory?.scene_count === 0) {
              setError('Warning: No Sentinel-1 data found. Processing will download data automatically (may take longer).');
            }
          } catch (debugErr) {
            addDebugInfo(`Debug checks failed: ${debugErr instanceof Error ? debugErr.message : 'Unknown error'}`);
          }
        } else {
          throw new Error('Backend not responding properly');
        }
      } catch (err) {
        addDebugInfo(`Backend connection failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setConnectionStatus('disconnected');
        setError(`Cannot connect to backend server at ${API_BASE_URL}. Make sure Docker containers are running and network is configured.`);
      }
    };

    checkConnection();
    
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Check for existing processing results
  const checkForExistingResults = async () => {
    try {
      addDebugInfo('Checking for existing results...');
      const backupResponse = await apiFetch(`${API_BASE_URL}/api/list-backups`);
      
      if (backupResponse.status === 'success' && backupResponse.total_files > 0) {
        addDebugInfo(`Found existing results: ${backupResponse.total_files} files`);
        
        const dummyJobId = 'existing-results';
        
        const dummyStatus: ProcessingStatus = {
          job_id: dummyJobId,
          status: 'completed',
          progress: 100,
          message: 'Loaded existing results',
          results: {
            files: {},
            statistics: {
              total_files: backupResponse.total_files,
              sbas_files: Math.floor(backupResponse.total_files / 2),
              ps_files: Math.floor(backupResponse.total_files / 2),
              time_series_files: backupResponse.total_files - 6
            },
            processing_time: 0
          }
        };
        
        setJobId(dummyJobId);
        setStatus(dummyStatus);
        
        // Try to load layers
        await loadExistingResults(dummyJobId);
      } else {
        addDebugInfo('No existing results found');
      }
    } catch (err) {
      addDebugInfo(`Could not check for existing results: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Load existing results by trying to get layers directly
  const loadExistingResults = async (jobId: string) => {
    try {
      addDebugInfo('Loading existing results...');
      
      const layersData = await apiFetch(`${API_BASE_URL}/api/layers/${jobId}`);
      
      if (layersData && layersData.length > 0) {
        addDebugInfo(`Loaded existing layers: ${layersData.length} layers`);
        setLayers(layersData);
      }
      
      // Try to load comparison plot
      await loadComparisonPlot(jobId);
    } catch (err) {
      addDebugInfo(`Could not load existing results: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setJobId(null);
      setStatus(null);
    }
  };

  // Initialize OpenLayers map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    addDebugInfo('Initializing OpenLayers map...');

    try {
      const map = new Map({
        target: mapRef.current,
        layers: [
          new TileLayer({
            source: new OSM(),
            opacity: 0.8
          })
        ],
        view: new View({
          center: BERLIN_CENTER,
          zoom: 11,
          projection: 'EPSG:3857',
          minZoom: 8,
          maxZoom: 18
        })
      });

      mapInstanceRef.current = map;
      addDebugInfo('Map initialized successfully');

      // Add click handler for debugging
      map.on('click', (evt) => {
        const coordinate = toLonLat(evt.coordinate);
        addDebugInfo(`Map clicked at: ${coordinate[0].toFixed(6)}, ${coordinate[1].toFixed(6)}`);
      });

    } catch (err) {
      addDebugInfo(`Map initialization failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setError(`Failed to initialize map: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.setTarget(undefined);
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Load comparison plot
  const loadComparisonPlot = async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/comparison/${jobId}`, {
        mode: 'cors',
        credentials: 'omit'
      });
      if (response.ok) {
        setComparisonImage(`${API_BASE_URL}/api/comparison/${jobId}`);
        addDebugInfo('Comparison plot loaded');
      }
    } catch (err) {
      addDebugInfo('No comparison plot available');
    }
  };

  // Poll processing status
  useEffect(() => {
    if (!jobId || !processing || connectionStatus !== 'connected') return;

    const pollStatus = async () => {
      try {
        const statusData: ProcessingStatus = await apiFetch(`${API_BASE_URL}/api/status/${jobId}`);
        setStatus(statusData);

        if (statusData.status === 'completed') {
          setProcessing(false);
          await loadResults(jobId);
        } else if (statusData.status === 'failed') {
          setProcessing(false);
          setError(statusData.message);
        }
      } catch (err) {
        console.error('Error polling status:', err);
        setError('Failed to check processing status');
      }
    };

    const interval = setInterval(pollStatus, 3000);
    return () => clearInterval(interval);
  }, [jobId, processing, connectionStatus]);

  // Start InSAR processing
  const startProcessing = async () => {
    if (connectionStatus !== 'connected') {
      setError('Backend server not available. Check Docker containers.');
      return;
    }

    try {
      setError(null);
      setProcessing(true);
      
      const result: ProcessingStatus = await apiFetch(`${API_BASE_URL}/api/process`, {
        method: 'POST',
        body: JSON.stringify({
          analysis_type: 'complete'
        })
      });

      setJobId(result.job_id);
      setStatus(result);
    } catch (err) {
      console.error('Processing start failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setProcessing(false);
    }
  };

  // Load processing results
  const loadResults = async (jobId: string) => {
    try {
      addDebugInfo(`Loading results for job ${jobId}`);
      
      // Load layers
      let layersData: MapLayer[] = [];
      let retries = 3;
      
      while (retries > 0) {
        try {
          layersData = await apiFetch(`${API_BASE_URL}/api/layers/${jobId}`);
          break;
        } catch (err) {
          addDebugInfo(`Layer loading attempt failed (${4 - retries}/3): ${err instanceof Error ? err.message : 'Unknown error'}`);
          retries--;
          
          if (retries > 0) {
            await new Promise(resolve => setTimeout(resolve, 2000));
          } else {
            throw err;
          }
        }
      }
      
      addDebugInfo(`Loaded layers: ${layersData.length} layers`);
      setLayers(layersData);
      
      // Load comparison plot if available
      await loadComparisonPlot(jobId);

    } catch (err) {
      addDebugInfo(`Error loading results: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setError(`Failed to load results: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Connection status indicator
  const ConnectionStatus: React.FC = () => (
    <div className={`connection-status ${connectionStatus}`}>
      <span className="status-dot"></span>
      <span className="status-text">
        {connectionStatus === 'checking' && 'Connecting to server...'}
        {connectionStatus === 'connected' && `Server connected (${API_BASE_URL})`}
        {connectionStatus === 'disconnected' && 'Server disconnected'}
      </span>
    </div>
  );

  // Progress bar component
  const ProgressBar: React.FC<{ progress: number }> = ({ progress }) => (
    <div className="progress-container">
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className="progress-text">{progress.toFixed(1)}%</span>
    </div>
  );

  // Layer control component - shows available TIF files
  const LayerControls: React.FC = () => (
    <div className="layer-controls">
      <h3>Available InSAR TIF Files</h3>
      {layers.length === 0 ? (
        <p>No TIF files available. {status?.status === 'completed' ? 'Processing completed but no files could be loaded.' : 'Run processing first.'}</p>
      ) : (
        <div className="layer-list">
          {layers.map((layer, index) => (
            <div key={index} className="layer-item">
              <div className="layer-info">
                <div className="layer-name">{layer.name}</div>
                <div className="layer-details">
                  <small>Type: {layer.type} | Method: {layer.method}</small>
                </div>
                <div className="layer-range">
                  <small>Range: {layer.color_range[0].toFixed(1)} to {layer.color_range[1].toFixed(1)} mm/year</small>
                </div>
                <div className="layer-path">
                  <small>File: {layer.file_path}</small>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // Debug panel component
  const DebugPanel: React.FC = () => (
    <div className="debug-panel">
      <h4>Debug Info</h4>
      <div className="debug-messages">
        {debugInfo.map((info, index) => (
          <div key={index} className="debug-message">{info}</div>
        ))}
      </div>
    </div>
  );

  // Results panel component
  const ResultsPanel: React.FC = () => {
    if (!status?.results) return null;

    const stats = status.results.statistics;
    
    return (
      <div className="results-panel">
        <h3>Processing Results</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <label>Processing Time:</label>
            <span>{status.results.processing_time.toFixed(1)} minutes</span>
          </div>
          <div className="stat-item">
            <label>Total Files:</label>
            <span>{stats.total_files}</span>
          </div>
          <div className="stat-item">
            <label>SBAS Files:</label>
            <span>{stats.sbas_files}</span>
          </div>
          <div className="stat-item">
            <label>PS Files:</label>
            <span>{stats.ps_files}</span>
          </div>
          {stats.sbas_velocity_mean && (
            <div className="stat-item">
              <label>SBAS Avg Velocity:</label>
              <span>{stats.sbas_velocity_mean.toFixed(2)} mm/year</span>
            </div>
          )}
          {stats.ps_velocity_mean && (
            <div className="stat-item">
              <label>PS Avg Velocity:</label>
              <span>{stats.ps_velocity_mean.toFixed(2)} mm/year</span>
            </div>
          )}
        </div>

        {comparisonImage && (
          <div className="comparison-section">
            <h4>SBAS vs PS Comparison</h4>
            <img 
              src={comparisonImage} 
              alt="SBAS vs PS Comparison" 
              className="comparison-plot"
            />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Berlin InSAR Analysis</h1>
        <p>SBAS and PS interferometric analysis for ground deformation monitoring</p>
        <ConnectionStatus />
      </header>

      <main className="app-main">
        <div className="control-panel">
          <div className="processing-controls">
            <button 
              className={`process-button ${processing ? 'processing' : ''}`}
              onClick={startProcessing}
              disabled={processing || connectionStatus !== 'connected'}
            >
              {processing ? 'Processing...' : 'Perform SBAS/PSI Analysis'}
            </button>

            {status && (
              <div className="status-display">
                <div className="status-info">
                  <span className={`status-badge ${status.status}`}>
                    {status.status.toUpperCase()}
                  </span>
                  <span className="status-message">{status.message}</span>
                </div>
                {processing && <ProgressBar progress={status.progress} />}
              </div>
            )}

            {error && (
              <div className="error-message">
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>

          <LayerControls />
          <ResultsPanel />
          <DebugPanel />
        </div>

        <div className="map-container">
          <div 
            ref={mapRef} 
            className="map"
            style={{ width: '100%', height: '100%' }}
          />
          
          <div className="map-info">
            <div className="map-legend">
              <h4>Berlin InSAR Map</h4>
              <div className="map-details">
                <small>Map Center: Berlin, Germany</small>
                <small>Click map to see coordinates</small>
                <small>TIF files listed in control panel →</small>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;