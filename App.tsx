// src/App.tsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import WebGLTileLayer from 'ol/layer/WebGLTile';
import GeoTIFF from 'ol/source/GeoTIFF';
import { fromLonLat, toLonLat } from 'ol/proj';
import 'ol/ol.css';

const BERLIN = fromLonLat([13.404954, 52.520008]);
const MIN = -27.01;
const MAX = 31.90;

interface BackupFile {
  name: string;
  size: number;
  modified: string;
  path: string;
}

interface BackupResponse {
  status: string;
  backup_directory?: string;
  total_files?: number;
  files?: BackupFile[];
  message?: string;
}

const App: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<Map | null>(null);
  const layerRef = useRef<WebGLTileLayer | null>(null);

  const [fileName, setFileName] = useState('');
  const [threshold, setThreshold] = useState(1);
  const [popup, setPopup] = useState<{ lon: number; lat: number; value: number } | null>(null);
  
  // New state for server files
  const [serverFiles, setServerFiles] = useState<BackupFile[]>([]);
  const [selectedServerFile, setSelectedServerFile] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [showFileSelector, setShowFileSelector] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const buildStyle = (t: number) => ({
    color: [
      'case',
      ['any', ['==', ['band', 1], -9999], ['<=', ['abs', ['band', 1]], t]],
      [0, 0, 0, 0],
      ['interpolate', ['linear'], ['band', 1], MIN, [0, 0, 255, 1], MAX, [255, 0, 0, 1]]
    ]
  });

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;
    mapInstance.current = new Map({
      target: mapRef.current,
      layers: [new TileLayer({ source: new OSM(), opacity: 0.6 })],
      view: new View({ center: BERLIN, zoom: 11 })
    });

    // Clear any cached requests on mount
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => {
          if (name.includes('raster')) {
            caches.delete(name);
          }
        });
      });
    }
  }, []);

  useEffect(() => {
    layerRef.current?.setStyle(buildStyle(threshold));
  }, [threshold]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;

    // click handler for pixel value
    const onClick = async (evt: any) => {
      if (!layerRef.current) return;
      const data = await layerRef.current.getData(evt.pixel);
      if (!data) return; // nodata or missed
      const value = data[0];
      if (Number.isFinite(value)) {
        const [lon, lat] = toLonLat(evt.coordinate);
        setPopup({ lon, lat, value });
      }
    };

    map.on('singleclick', onClick);
    return () => map.un('singleclick', onClick);
  }, []);

  // Fetch available files from server
  const fetchServerFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/list-backups`);
      const data: BackupResponse = await response.json();
      
      if (data.status === 'success' && data.files) {
        // Filter for .tif files only
        const tifFiles = data.files.filter(file => 
          file.name.toLowerCase().endsWith('.tif') || 
          file.name.toLowerCase().endsWith('.tiff')
        );
        setServerFiles(tifFiles);
      } else {
        console.warn('No files found or error:', data.message);
        setServerFiles([]);
      }
    } catch (error) {
      console.error('Error fetching server files:', error);
      setServerFiles([]);
    } finally {
      setLoading(false);
    }
  };

  // SIMPLE SOLUTION: Download and use like drag-and-drop
  const loadServerFile = async (filename: string) => {
    if (!filename) return;
    
    setLoading(true);
    setFileName(filename);
    setPopup(null);

    try {
      // Remove existing layer
      layerRef.current && mapInstance.current?.removeLayer(layerRef.current);

      console.log('Downloading file from server:', filename);

      // Download the file completely
      const response = await fetch(`${API_BASE_URL}/api/raster/simple/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to download: ${response.status}`);
      }
      
      const blob = await response.blob();
      console.log('Downloaded blob:', blob.size, 'bytes');
      
      // Create object URL - same as drag and drop
      const url = URL.createObjectURL(blob);
      
      // Use EXACTLY the same code as your working onDrop function
      const source = new GeoTIFF({
        sources: [{ url, nodata: -9999 }],
        interpolate: false,
        normalize: false
      });

      const layer = new WebGLTileLayer({
        source,
        opacity: 1,
        style: buildStyle(threshold)
      });

      mapInstance.current!.addLayer(layer);
      layerRef.current = layer;

      const view = await source.getView();
      view && mapInstance.current!.getView().fit(view.extent, {
        padding: [40, 40, 40, 40],
        maxZoom: 16
      });

      console.log('Success! File loaded using drag-and-drop method');

    } catch (error) {
      console.error('Error:', error);
      alert(`Failed to load ${filename}: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Original drag and drop handler
  const onDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setFileName(file.name);
    setPopup(null);

    layerRef.current && mapInstance.current?.removeLayer(layerRef.current);

    const source = new GeoTIFF({
      sources: [{ url, nodata: -9999 }],
      interpolate: false,
      normalize: false
    });

    const layer = new WebGLTileLayer({
      source,
      opacity: 1,
      style: buildStyle(threshold)
    });

    mapInstance.current!.addLayer(layer);
    layerRef.current = layer;

    const view = await source.getView();
    view && mapInstance.current!.getView().fit(view.extent, {
      padding: [40, 40, 40, 40],
      maxZoom: 16
    });
  }, [threshold]);

  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  return (
    <div onDrop={onDrop} onDragOver={onDragOver} style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ padding: '0.5rem 1rem', background: '#f5f5f5', zIndex: 10, borderBottom: '1px solid #ddd' }}>
        <h1 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem' }}>GeoTIFF Viewer (clickable)</h1>
        
        {fileName && (
          <p style={{ margin: '0 0 0.5rem 0', color: '#666' }}>
            Loaded: <strong>{fileName}</strong> | Range: {MIN} – {MAX} mm/yr
          </p>
        )}
        
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            Filter ±
            <input
              type="range"
              min="0"
              max={Math.max(Math.abs(MIN), Math.abs(MAX))}
              step="0.1"
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              style={{ width: '120px' }}
            />
            <span style={{ minWidth: '60px' }}>{threshold.toFixed(1)} mm/yr</span>
          </label>

          <button
            onClick={() => {
              setShowFileSelector(!showFileSelector);
              if (!showFileSelector && serverFiles.length === 0) {
                fetchServerFiles();
              }
            }}
            style={{
              padding: '0.5rem 1rem',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
            disabled={loading}
          >
            {loading ? 'Loading...' : (showFileSelector ? 'Hide' : 'Load from Server')}
          </button>
        </div>

        {showFileSelector && (
          <div style={{ 
            marginTop: '1rem', 
            padding: '1rem', 
            background: '#fff', 
            border: '1px solid #ddd', 
            borderRadius: '4px' 
          }}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h3 style={{ margin: 0, fontSize: '1rem' }}>Available Files on Server:</h3>
              <button
                onClick={fetchServerFiles}
                style={{
                  padding: '0.25rem 0.5rem',
                  background: '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '0.8rem'
                }}
                disabled={loading}
              >
                Refresh
              </button>
            </div>
            
            {loading ? (
              <p>Loading files...</p>
            ) : serverFiles.length > 0 ? (
              <div>
                <select
                  value={selectedServerFile}
                  onChange={(e) => setSelectedServerFile(e.target.value)}
                  style={{
                    padding: '0.5rem',
                    marginRight: '0.5rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    minWidth: '200px'
                  }}
                >
                  <option value="">Select a file...</option>
                  {serverFiles.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
                    </option>
                  ))}
                </select>
                
                <button
                  onClick={() => loadServerFile(selectedServerFile)}
                  disabled={!selectedServerFile || loading}
                  style={{
                    padding: '0.5rem 1rem',
                    background: selectedServerFile ? '#007bff' : '#ccc',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: selectedServerFile ? 'pointer' : 'not-allowed'
                  }}
                >
                  Load File
                </button>
                
                <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: '#666' }}>
                  Found {serverFiles.length} GeoTIFF file(s)
                </p>
              </div>
            ) : (
              <p style={{ color: '#666', fontStyle: 'italic' }}>
                No GeoTIFF files found in server directory. You can still drag and drop files from your local computer.
              </p>
            )}
          </div>
        )}
        
        {!fileName && (
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: '#888' }}>
            Drag and drop a GeoTIFF file here, or load from server using the button above
          </p>
        )}
      </header>

      <div ref={mapRef} style={{ flex: 1, position: 'relative' }}>
        {/* Loading overlay */}
        {loading && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(255, 255, 255, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            fontSize: '1.2rem'
          }}>
            Loading file...
          </div>
        )}

        {/* Popup for pixel values */}
        {popup && (
          <div
            style={{
              position: 'absolute',
              left: 20,
              top: 20,
              background: 'rgba(0,0,0,0.8)',
              color: '#fff',
              padding: '6px 10px',
              borderRadius: 4,
              pointerEvents: 'none',
              fontSize: 12,
              zIndex: 100
            }}
          >
            {popup.value.toFixed(2)} mm/yr<br />
            lon: {popup.lon.toFixed(4)}<br />
            lat: {popup.lat.toFixed(4)}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;