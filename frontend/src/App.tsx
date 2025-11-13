import React, { useState, useEffect, useRef, useCallback } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import WebGLTileLayer from 'ol/layer/WebGLTile';
import GeoTIFF from 'ol/source/GeoTIFF';
import { Vector as VectorSource } from 'ol/source';
import { Vector as VectorLayer } from 'ol/layer';
import Draw from 'ol/interaction/Draw';
import { toLonLat, transformExtent, fromLonLat } from 'ol/proj';
import { createBox } from 'ol/interaction/Draw';
import { Style, Stroke, Fill } from 'ol/style';
import 'ol/ol.css';
import './App.css';

// TRUE min/max scanner (no percentiles)
import { fromUrl as geotiffFromUrl } from 'geotiff';

// ========== Types ==========
interface ProcessingStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  started_at?: string;
  completed_at?: string;
  results?: ProcessingResults;
}

interface ProcessingResults {
  files: Record<string, string>;
  statistics: {
    total_files: number;
    sbas_files: number;
    ps_files: number;
    time_series_files: number;
    sbas_velocity_mean?: number;
    ps_velocity_mean?: number;
    sbas_velocity_range?: number[];
    ps_velocity_range?: number[];
  };
  processing_time: number;
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

interface BoundingBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

interface SentinelProduct {
  id: string;
  title: string;
  date: string;
  footprint: string;
  size: string;
  thumbnailUrl?: string;
  metadata: {
    mission: string;
    mode: string;
    orbitDirection: string;
    path: number;
    absoluteOrbit: number;
    startTime: string;
    stopTime: string;
    url: string;
    burstID: string;
    subswath: string;
    burstIndex: number;
    fullBurstID: string;
    polarization: string;
    frame?: number;
    burstIdentifier: string;
  };
  burstID?: string;
  subswath?: string;
  burstIndex?: number;
}

interface DownloadStatus {
  productId: string;
  status: string;
  message: string;
  progress?: number;
}

// ========== Config ==========
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ========== Component ==========
const App: React.FC = () => {
  // Processing state
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [layers, setLayers] = useState<MapLayer[]>([]);
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [debugInfo, setDebugInfo] = useState<string[]>([]);

  // GeoTIFF viewer state
  const [fileName, setFileName] = useState('');
  const [threshold, setThreshold] = useState(1);
  const [popup, setPopup] = useState<{ lon: number; lat: number; value: number } | null>(null);
  const [serverFiles, setServerFiles] = useState<BackupFile[]>([]);
  const [selectedServerFile, setSelectedServerFile] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [showFileSelector, setShowFileSelector] = useState(false);

  // Tabs / search / download
  const [activeTab, setActiveTab] = useState<'viewer' | 'search' | 'processing'>('viewer');
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
  const [drawInteraction, setDrawInteraction] = useState<Draw | null>(null);
  const [vectorSource] = useState(new VectorSource());
  const [searchResults, setSearchResults] = useState<SentinelProduct[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [availablePaths, setAvailablePaths] = useState<number[]>([]);
  const [availableSubswaths, setAvailableSubswaths] = useState<string[]>([]);
  const [downloadStatuses, setDownloadStatuses] = useState<Record<string, DownloadStatus>>({});
  const [selectedProducts, setSelectedProducts] = useState<string[]>([]);
  const [isDownloadingSelection, setIsDownloadingSelection] = useState(false);
  const [downloadedProducts, setDownloadedProducts] = useState<string[]>([]);
  const [asfUrl, setAsfUrl] = useState('');

  // Search form defaults
  const [searchParams, setSearchParams] = useState({
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    orbit: 'both',
    polarization: 'all',
    path: '',
    subswath: 'all',
    fullBurstID: ''
  });

  // Map refs
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const layerRef = useRef<WebGLTileLayer | null>(null);

  // Germany center
  const GERMANY_CENTER = fromLonLat([10.451526, 51.165691]);

  // Dynamic data range (true min/max; default until first raster is loaded)
  const [dataRange, setDataRange] = useState<{ min: number; max: number }>({ min: -30, max: 30 });

  // -------- Helpers --------
  const addDebugInfo = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const debugMessage = `${timestamp}: ${message}`;
    console.log(`[DEBUG] ${debugMessage}`);
    setDebugInfo(prev => [...prev.slice(-4), debugMessage]);
  };

  const apiFetch = async (url: string, options: RequestInit = {}) => {
    try {
      addDebugInfo(`Fetching: ${url}`);
      const response = await fetch(url, {
        ...options,
        mode: 'cors',
        credentials: 'omit',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          ...options.headers
        }
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

  // Build WebGL style using dynamic (true) range
  const buildStyle = (t: number) => ({
    color: [
      'case',
      ['any', ['==', ['band', 1], -9999], ['<=', ['abs', ['band', 1]], t]],
      [0, 0, 0, 0],
      ['interpolate', ['linear'], ['band', 1],
        dataRange.min, [0, 0, 255, 1],
        dataRange.max, [255, 0, 0, 1]
      ]
    ]
  });

  // -------- TRUE min/max scanner (block-wise; skips nodata & non-finite) --------
  async function computeTrueMinMax(url: string): Promise<{ min: number; max: number }> {
    const tiff = await geotiffFromUrl(url);
    const img = await tiff.getImage();

    const nodataRaw = (img as any).getGDALNoData?.() ?? (img as any).fileDirectory?.GDAL_NODATA;
    const noData = nodataRaw != null ? Number(nodataRaw) : undefined;

    const width = img.getWidth();
    const height = img.getHeight();

    const blockW = (img as any).getTileWidth?.() || (img as any).getBlockWidth?.() || 512;
    const blockH = (img as any).getTileHeight?.() || (img as any).getBlockHeight?.() || 512;

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let found = false;

    for (let y = 0; y < height; y += blockH) {
      const h = Math.min(blockH, height - y);
      for (let x = 0; x < width; x += blockW) {
        const w = Math.min(blockW, width - x);
        const ras = await img.readRasters({
          samples: [0],
          window: [x, y, x + w, y + h],
          interleave: false
        });
        const data = ras[0] as Iterable<number>;
        for (const v of data as any) {
          if (!Number.isFinite(v)) continue;
          if (noData !== undefined && v === noData) continue;
          if (v < min) min = v;
          if (v > max) max = v;
          found = true;
        }
      }
    }

    if (!found || !Number.isFinite(min) || !Number.isFinite(max) || min === max) {
      return { min: -30, max: 30 };
    }
    return { min, max };
  }

  // -------- Map init --------
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    addDebugInfo('Initializing OpenLayers map...');
    try {
      const vectorLayer = new VectorLayer({
        source: vectorSource,
        style: new Style({
          fill: new Fill({ color: 'rgba(255, 255, 255, 0.2)' }),
          stroke: new Stroke({ color: '#ffcc33', width: 2 })
        })
      });

      const map = new Map({
        target: mapRef.current,
        layers: [
          new TileLayer({ source: new OSM(), opacity: 0.6 }),
          vectorLayer
        ],
        view: new View({
          center: GERMANY_CENTER,
          zoom: 6,
          projection: 'EPSG:3857',
          minZoom: 4,
          maxZoom: 18
        })
      });

      mapInstanceRef.current = map;
      addDebugInfo('Map initialized successfully');

      // Click for pixel readout
      map.on('singleclick', async (evt) => {
        if (layerRef.current) {
          try {
            const data = await (layerRef.current as any).getData(evt.pixel);
            if (data) {
              const value = data[0];
              if (Number.isFinite(value)) {
                const [lon, lat] = toLonLat(evt.coordinate);
                setPopup({ lon, lat, value });
                addDebugInfo(`Pixel value: ${value.toFixed(2)} at ${lon.toFixed(6)}, ${lat.toFixed(6)}`);
              }
            }
          } catch (err) {
            console.warn('Could not get pixel data:', err);
          }
        }

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
  }, [vectorSource]);

  // Update style when threshold or dataRange changes
  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.setStyle(buildStyle(threshold));
    }
  }, [threshold, dataRange.min, dataRange.max]);

  // -------- Backend health + existing results --------
  useEffect(() => {
    const checkConnection = async () => {
      try {
        addDebugInfo(`Checking connection to: ${API_BASE_URL}`);
        const response = await apiFetch(`${API_BASE_URL}/health`);
        if (response && typeof response === 'object' && response.status === 'healthy') {
          setConnectionStatus('connected');
          setError(null);
          addDebugInfo('Backend connected successfully');
          await checkForExistingResults();
          await checkExistingSLCData();
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
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkExistingSLCData = async () => {
    try {
      const response = await apiFetch(`${API_BASE_URL}/scan-slc`);
      if (response.success && response.has_sufficient_data) {
        addDebugInfo(`Found ${response.scene_count} existing SLC scenes`);
        if (response.product_ids && response.product_ids.length > 0) {
          setDownloadedProducts(response.product_ids);
        } else {
          setDownloadedProducts(['existing_data']);
        }
      }
    } catch {
      addDebugInfo('No existing SLC data found');
    }
  };

  // -------- Draw bbox --------
  const enableDrawMode = () => {
    if (!mapInstanceRef.current) return;

    if (drawInteraction) mapInstanceRef.current.removeInteraction(drawInteraction);
    vectorSource.clear();
    setBoundingBox(null);

    const draw = new Draw({
      source: vectorSource,
      type: 'Circle',
      geometryFunction: createBox()
    });

    draw.on('drawend', (event) => {
      const feature = event.feature;
      const geometry = feature.getGeometry();
      const extent = geometry.getExtent();
      const wgs84Extent = transformExtent(extent, 'EPSG:3857', 'EPSG:4326');

      const formattedBBox = {
        west: Number(wgs84Extent[0].toFixed(6)),
        south: Number(wgs84Extent[1].toFixed(6)),
        east: Number(wgs84Extent[2].toFixed(6)),
        north: Number(wgs84Extent[3].toFixed(6))
      };

      setBoundingBox(formattedBBox);
      addDebugInfo(`Bounding box drawn: ${JSON.stringify(formattedBBox)}`);
      mapInstanceRef.current!.removeInteraction(draw);
      setDrawInteraction(null);
    });

    mapInstanceRef.current.addInteraction(draw);
    setDrawInteraction(draw);
  };

  const clearBoundingBox = () => {
    if (!mapInstanceRef.current) return;
    vectorSource.clear();
    setBoundingBox(null);
    setAsfUrl('');
    if (drawInteraction) {
      mapInstanceRef.current.removeInteraction(drawInteraction);
      setDrawInteraction(null);
    }
  };

  // -------- Search / Download --------
  const searchSentinelData = async () => {
    if (!boundingBox) {
      alert('Please draw a bounding box first');
      return;
    }
    setIsSearching(true);

    try {
      const response = await apiFetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        body: JSON.stringify({
          boundingBox: boundingBox,
          startDate: searchParams.startDate,
          endDate: searchParams.endDate,
          productType: 'SLC',
          orbitDirection: searchParams.orbit !== 'both' ? searchParams.orbit.toUpperCase() : null,
          polarization: searchParams.polarization !== 'all' ? searchParams.polarization.toUpperCase() : null,
          path: searchParams.path ? parseInt(searchParams.path) : null,
          subswath: searchParams.subswath !== 'all' ? searchParams.subswath : null,
          fullBurstID: searchParams.fullBurstID || null
        })
      });

      addDebugInfo(`Search completed: ${response.products?.length || 0} results`);
      setSearchResults(response.products || []);
      setAsfUrl(response.asfUrl || '');

      if (response.products && response.products.length > 0) {
        const paths = [...new Set(response.products.map((p: SentinelProduct) => p.metadata.path))].sort((a, b) => a - b);
        const subswaths = [...new Set(response.products.map((p: SentinelProduct) => p.metadata.subswath))].filter(Boolean);
        setAvailablePaths(paths);
        setAvailableSubswaths(subswaths);
      }
    } catch (error) {
      console.error('Search failed:', error);
      addDebugInfo(`Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      alert('Failed to search for Sentinel data. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  const downloadProduct = async (productId: string) => {
    try {
      setDownloadStatuses(prev => ({
        ...prev,
        [productId]: { productId, status: 'starting', message: 'Starting download...', progress: 0 }
      }));

      const response = await apiFetch(`${API_BASE_URL}/download/${productId}`, { method: 'POST' });

      setDownloadStatuses(prev => ({ ...prev, [productId]: response }));

      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await apiFetch(`${API_BASE_URL}/download/status/${productId}`);
          setDownloadStatuses(prev => ({ ...prev, [productId]: statusResponse }));

          if (statusResponse.status === 'completed' || statusResponse.status === 'failed') {
            clearInterval(pollInterval);
            if (statusResponse.status === 'completed') {
              setDownloadedProducts(prev => [...prev.filter(id => id !== productId), productId]);
            }
          }
        } catch {
          clearInterval(pollInterval);
        }
      }, 2000);
    } catch (error) {
      console.error('Error downloading product:', error);
      setDownloadStatuses(prev => ({
        ...prev,
        [productId]: {
          productId,
          status: 'failed',
          message: error instanceof Error ? error.message : 'Unknown error',
          progress: 0
        }
      }));
    }
  };

  const toggleProductSelection = (productId: string) => {
    setSelectedProducts(prev => prev.includes(productId) ? prev.filter(id => id !== productId) : [...prev, productId]);
  };

  // -------- Existing results (from backups) --------
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
        await loadExistingResults(dummyJobId);
      } else {
        addDebugInfo('No existing results found');
      }
    } catch (err) {
      addDebugInfo(`Could not check for existing results: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const loadExistingResults = async (jobIdVal: string) => {
    try {
      addDebugInfo('Loading existing results...');
      const layersData = await apiFetch(`${API_BASE_URL}/api/layers/${jobIdVal}`);
      if (layersData && layersData.length > 0) {
        addDebugInfo(`Loaded existing layers: ${layersData.length} layers`);
        setLayers(layersData);
      }
      await loadComparisonPlot(jobIdVal);
    } catch (err) {
      addDebugInfo(`Could not load existing results: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setJobId(null);
      setStatus(null);
    }
  };

  const loadComparisonPlot = async (jobIdVal: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/comparison/${jobIdVal}`, { mode: 'cors', credentials: 'omit' });
      if (response.ok) {
        setComparisonImage(`${API_BASE_URL}/api/comparison/${jobIdVal}`);
        addDebugInfo('Comparison plot loaded');
      }
    } catch {
      addDebugInfo('No comparison plot available');
    }
  };

  // -------- Start processing --------
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
        body: JSON.stringify({ analysis_type: 'complete' })
      });
      setJobId(result.job_id);
      setStatus(result);
    } catch (err) {
      console.error('Processing start failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setProcessing(false);
    }
  };

  // Poll status
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
      } catch {
        setError('Failed to check processing status');
      }
    };

    const interval = setInterval(pollStatus, 3000);
    return () => clearInterval(interval);
  }, [jobId, processing, connectionStatus]);

  const loadResults = async (jobIdVal: string) => {
    try {
      addDebugInfo(`Loading results for job ${jobIdVal}`);
      let layersData: MapLayer[] = [];
      let retries = 3;

      while (retries > 0) {
        try {
          layersData = await apiFetch(`${API_BASE_URL}/api/layers/${jobIdVal}`);
          break;
        } catch (err) {
          addDebugInfo(`Layer loading attempt failed (${4 - retries}/3): ${err instanceof Error ? err.message : 'Unknown error'}`);
          retries--;
          if (retries > 0) await new Promise(resolve => setTimeout(resolve, 2000));
          else throw err;
        }
      }

      addDebugInfo(`Loaded layers: ${layersData.length} layers`);
      setLayers(layersData);
      await loadComparisonPlot(jobIdVal);
    } catch (err) {
      addDebugInfo(`Error loading results: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setError(`Failed to load results: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // -------- Server file list & load --------
  const fetchServerFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/list-backups`);
      const data: BackupResponse = await response.json();

      if (data.status === 'success' && data.files) {
        const tifFiles = data.files.filter(file =>
          file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')
        );
        setServerFiles(tifFiles);
        addDebugInfo(`Found ${tifFiles.length} TIF files on server`);
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

  // Load a server GeoTIFF and fit to its extent (with TRUE min/max)
  const loadServerFile = async (filename: string) => {
    if (!filename) return;

    setLoading(true);
    setFileName(filename);
    setPopup(null);

    try {
      if (layerRef.current && mapInstanceRef.current) {
        mapInstanceRef.current.removeLayer(layerRef.current);
      }

      addDebugInfo(`Downloading file from server: ${filename}`);
      const response = await fetch(`${API_BASE_URL}/api/raster/simple/${filename}`);
      if (!response.ok) throw new Error(`Failed to download: ${response.status}`);

      const blob = await response.blob();
      addDebugInfo(`Downloaded blob: ${blob.size} bytes`);
      const url = URL.createObjectURL(blob);

      // TRUE min/max
      try {
        const range = await computeTrueMinMax(url);
        setDataRange(range);
        addDebugInfo(`True range (server file): [${range.min.toFixed(2)}, ${range.max.toFixed(2)}]`);
      } catch {
        addDebugInfo('Failed to compute true range; keeping previous range.');
      }

      const source = new GeoTIFF({
        sources: [{ url, nodata: -9999 }],
        interpolate: false,
        normalize: false
      });

      const layer = new WebGLTileLayer({ source, opacity: 1, style: buildStyle(threshold) });
      mapInstanceRef.current!.addLayer(layer);
      layerRef.current = layer;

      // Fit to extent (handles 4326 vs 3857)
      try {
        addDebugInfo('Waiting for source to load...');
        await new Promise((resolve, reject) => {
          let attempts = 0;
          const maxAttempts = 50;

          const checkReady = async () => {
            attempts++;
            try {
              const view: any = await (source as any).getView?.();
              if (view && view.extent && view.extent.length === 4) {
                const [minX, minY, maxX, maxY] = view.extent;
                if (minX !== maxX && minY !== maxY && isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
                  let extentToFit = view.extent as [number, number, number, number];
                  const looksGeographic = minX >= -180 && maxX <= 180 && minY >= -90 && maxY <= 90;
                  if (looksGeographic) {
                    extentToFit = transformExtent(extentToFit, 'EPSG:4326', 'EPSG:3857');
                    addDebugInfo(`Transformed extent to Web Mercator: [${extentToFit.map(x => x.toFixed(2)).join(', ')}]`);
                  }
                  mapInstanceRef.current!.getView().fit(extentToFit, { padding: [40, 40, 40, 40], maxZoom: 16, duration: 1000 });
                  resolve(view);
                  return;
                }
              }
              if (attempts >= maxAttempts) { reject(new Error('Source failed to load extent after 5 seconds')); return; }
              setTimeout(checkReady, 100);
            } catch (error) {
              if (attempts >= maxAttempts) reject(error);
              else setTimeout(checkReady, 100);
            }
          };
          checkReady();
        });
        addDebugInfo('Success! File loaded and extent fitted');
      } catch (extentError) {
        addDebugInfo(`Warning: Could not fit to extent: ${extentError}`);
        mapInstanceRef.current!.getView().animate({ center: GERMANY_CENTER, zoom: 6, duration: 1000 });
      }
    } catch (error) {
      console.error('Error:', error);
      addDebugInfo(`Failed to load ${filename}: ${error}`);
      alert(`Failed to load ${filename}: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Drag & drop — with TRUE min/max + robust fit
  const onDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setFileName(file.name);
    setPopup(null);

    // TRUE min/max first
    try {
      const range = await computeTrueMinMax(url);
      setDataRange(range);
      addDebugInfo(`True range (drag-drop): [${range.min.toFixed(2)}, ${range.max.toFixed(2)}]`);
    } catch {
      addDebugInfo('Failed to compute true range for dropped file; keeping previous range.');
    }

    if (layerRef.current && mapInstanceRef.current) {
      mapInstanceRef.current.removeLayer(layerRef.current);
    }

    const source = new GeoTIFF({
      sources: [{ url, nodata: -9999 }],
      interpolate: false,
      normalize: false
    });

    const layer = new WebGLTileLayer({ source, opacity: 1, style: buildStyle(threshold) });
    mapInstanceRef.current!.addLayer(layer);
    layerRef.current = layer;

    // Fit to extent (handles 4326 vs 3857)
    try {
      const view: any = await (source as any).getView?.();
      if (view && view.extent && view.extent.length === 4) {
        let extentToFit = view.extent as [number, number, number, number];
        const [minX, minY, maxX, maxY] = extentToFit;
        const looksGeographic = minX >= -180 && maxX <= 180 && minY >= -90 && maxY <= 90;
        if (looksGeographic) {
          extentToFit = transformExtent(extentToFit, 'EPSG:4326', 'EPSG:3857');
          addDebugInfo('Drag-drop extent detected in EPSG:4326; transformed to EPSG:3857.');
        }
        mapInstanceRef.current!.getView().fit(extentToFit, { padding: [40, 40, 40, 40], maxZoom: 16 });
        addDebugInfo(`Loaded file via drag-and-drop: ${file.name}`);
      } else {
        addDebugInfo('GeoTIFF view/extent not available; falling back to Germany view.');
        mapInstanceRef.current!.getView().animate({ center: GERMANY_CENTER, zoom: 6, duration: 800 });
      }
    } catch (err) {
      addDebugInfo(`Error reading GeoTIFF view; ${err instanceof Error ? err.message : String(err)}`);
      mapInstanceRef.current!.getView().animate({ center: GERMANY_CENTER, zoom: 6, duration: 800 });
    }
  }, [threshold]);

  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  // -------- UI subcomponents --------
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

  const ProgressBar: React.FC<{ progress: number }> = ({ progress }) => (
    <div className="progress-container">
      <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
      <span className="progress-text">{progress.toFixed(1)}%</span>
    </div>
  );

  const TabNavigation: React.FC = () => (
    <div className="tab-navigation">
      <button className={`tab-button ${activeTab === 'viewer' ? 'active' : ''}`} onClick={() => setActiveTab('viewer')}>GeoTIFF Viewer</button>
      <button className={`tab-button ${activeTab === 'search' ? 'active' : ''}`} onClick={() => setActiveTab('search')}>Data Search</button>
      <button className={`tab-button ${activeTab === 'processing' ? 'active' : ''}`} onClick={() => setActiveTab('processing')}>InSAR Processing</button>
    </div>
  );

  const GeoTIFFControls: React.FC = () => (
    <div className={`layer-controls ${activeTab === 'viewer' ? 'active' : 'hidden'}`}>
      <h3>GeoTIFF Viewer</h3>
      {fileName && (
        <div className="current-file-info">
          <strong>Loaded:</strong> {fileName}
          <br />
          <small>Range: {dataRange.min.toFixed(2)} – {dataRange.max.toFixed(2)} mm/yr</small>
        </div>
      )}

      <div className="threshold-control">
        <label>
          Filter ± {threshold.toFixed(1)} mm/yr
          <input
            type="range"
            min="0"
            max={Math.max(Math.abs(dataRange.min), Math.abs(dataRange.max))}
            step="0.1"
            value={threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            style={{ width: '100%', marginTop: '0.5rem' }}
          />
        </label>
      </div>

      <div className="server-file-controls">
        <button
          onClick={() => {
            setShowFileSelector(!showFileSelector);
            if (!showFileSelector && serverFiles.length === 0) fetchServerFiles();
          }}
          className="server-button"
          disabled={loading}
        >
          {loading ? 'Loading...' : (showFileSelector ? 'Hide Server Files' : 'Load from Server')}
        </button>

        {showFileSelector && (
          <div className="file-selector">
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
              <button onClick={fetchServerFiles} className="refresh-button" disabled={loading}>Refresh</button>
            </div>

            {loading ? (
              <p>Loading files...</p>
            ) : serverFiles.length > 0 ? (
              <div>
                <select
                  value={selectedServerFile}
                  onChange={(e) => setSelectedServerFile(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', marginBottom: '0.5rem', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="">Select a file...</option>
                  {serverFiles.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
                    </option>
                  ))}
                </select>

                <button onClick={() => loadServerFile(selectedServerFile)} disabled={!selectedServerFile || loading} className="load-button">
                  Load File
                </button>

                <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: '#666' }}>
                  Found {serverFiles.length} GeoTIFF file(s)
                </p>
              </div>
            ) : (
              <p style={{ color: '#666', fontStyle: 'italic' }}>No GeoTIFF files found in server directory.</p>
            )}
          </div>
        )}
      </div>

      {!fileName && <div className="drag-drop-hint"><small>You can also drag and drop GeoTIFF files onto the map</small></div>}
    </div>
  );

  const SearchControls: React.FC = () => (
    <div className={`layer-controls ${activeTab === 'search' ? 'active' : 'hidden'}`}>
      <h3>Sentinel-1 Data Search</h3>

      <div className="map-controls">
        <button onClick={enableDrawMode}>Draw Bounding Box</button>
        <button onClick={clearBoundingBox}>Clear</button>
      </div>

      {boundingBox && (
        <div className="bbox-info">
          <h4>Bounding Box</h4>
          <p>West: {boundingBox.west}°</p>
          <p>South: {boundingBox.south}°</p>
          <p>East: {boundingBox.east}°</p>
          <p>North: {boundingBox.north}°</p>
        </div>
      )}

      <div className="search-form">
        <div className="form-group">
          <label htmlFor="start-date">Start Date:</label>
          <input type="date" id="start-date" value={searchParams.startDate} onChange={(e) => setSearchParams(prev => ({ ...prev, startDate: e.target.value }))} />
        </div>
        <div className="form-group">
          <label htmlFor="end-date">End Date:</label>
          <input type="date" id="end-date" value={searchParams.endDate} onChange={(e) => setSearchParams(prev => ({ ...prev, endDate: e.target.value }))} />
        </div>
        <div className="form-group">
          <label htmlFor="orbit">Orbit Direction:</label>
          <select id="orbit" value={searchParams.orbit} onChange={(e) => setSearchParams(prev => ({ ...prev, orbit: e.target.value }))}>
            <option value="both">Both</option>
            <option value="ascending">Ascending</option>
            <option value="descending">Descending</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="polarization">Polarization:</label>
          <select id="polarization" value={searchParams.polarization} onChange={(e) => setSearchParams(prev => ({ ...prev, polarization: e.target.value }))}>
            <option value="all">All</option>
            <option value="vv">VV</option>
            <option value="vh">VH</option>
            <option value="hh">HH</option>
            <option value="hv">HV</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="path">Path (Relative Orbit):</label>
          <input type="number" id="path" placeholder="Optional - filter by path" min="1" max="175" value={searchParams.path} onChange={(e) => setSearchParams(prev => ({ ...prev, path: e.target.value }))} />
        </div>
        <div className="form-group">
          <label htmlFor="subswath">Subswath:</label>
          <select id="subswath" value={searchParams.subswath} onChange={(e) => setSearchParams(prev => ({ ...prev, subswath: e.target.value }))}>
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
            value={searchParams.fullBurstID}
            onChange={(e) => setSearchParams(prev => ({ ...prev, fullBurstID: e.target.value }))}
          />
        </div>
        <button onClick={searchSentinelData} disabled={!boundingBox || isSearching} className="search-button">
          {isSearching ? 'Searching...' : 'Search Available Burst Data'}
        </button>
      </div>

      {asfUrl && (
        <div className="asf-url-section">
          <h4>View in ASF Data Search</h4>
          <div className="asf-url-container">
            <p>View the same search parameters in the Alaska Satellite Facility data search tool:</p>
            <a href={asfUrl} target="_blank" rel="noopener noreferrer" className="asf-url-link">Open in ASF Data Search Tool</a>
          </div>
        </div>
      )}

      {availablePaths.length > 0 && (
        <div className="available-filters">
          <h4>Available Paths in Results</h4>
          <p>{availablePaths.join(', ')}</p>
        </div>
      )}
      {availableSubswaths.length > 0 && (
        <div className="available-filters">
          <h4>Available Subswaths in Results</h4>
          <p>{availableSubswaths.join(', ')}</p>
        </div>
      )}

      <h4>Search Results {searchResults.length > 0 && `(${searchResults.length})`}</h4>
      <div className="results-container">
        {isSearching ? (
          <p>Searching for available burst data...</p>
        ) : searchResults.length > 0 ? (
          <>
            <div className="selection-controls">
              <button onClick={() => setSelectedProducts(searchResults.map(p => p.id))} className="select-button">Select All</button>
              <button onClick={() => setSelectedProducts([])} className="select-button">Deselect All</button>
              <button
                onClick={() => {
                  selectedProducts.forEach(productId => downloadProduct(productId));
                  setIsDownloadingSelection(true);
                  setTimeout(() => setIsDownloadingSelection(false), 2000);
                }}
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
                    <p><strong>Path:</strong> {product.metadata.path}</p>
                    <p><strong>Subswath:</strong> {product.metadata.subswath}</p>
                    <p><strong>Burst ID:</strong> {product.metadata.burstID}</p>
                    <p><strong>Full Burst ID:</strong> {product.metadata.fullBurstID}</p>
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
    </div>
  );

  const ProcessingControls: React.FC = () => (
    <div className={`layer-controls ${activeTab === 'processing' ? 'active' : 'hidden'}`}>
      <h3>InSAR Processing</h3>

      <div className="processing-controls">
        <button className={`process-button ${processing ? 'processing' : ''}`} onClick={startProcessing} disabled={processing || connectionStatus !== 'connected'}>
          {processing ? 'Processing...' : 'Perform SBAS/PSI Analysis'}
        </button>

        {status && (
          <div className="status-display">
            <div className="status-info">
              <span className={`status-badge ${status.status}`}>{status.status.toUpperCase()}</span>
              <span className="status-message">{status.message}</span>
            </div>
            {processing && <ProgressBar progress={status.progress} />}
          </div>
        )}

        {error && <div className="error-message"><strong>Error:</strong> {error}</div>}
      </div>

      <div className="layer-list">
        <h4>Processing Results</h4>
        {layers.length === 0 ? (
          <p>No processing results available. {status?.status === 'completed' ? 'Processing completed but no files could be loaded.' : 'Run processing first.'}</p>
        ) : (
          <div className="layer-list">
            {layers.map((layer, index) => (
              <div key={index} className="layer-item">
                <div className="layer-info">
                  <div className="layer-name">{layer.name}</div>
                  <div className="layer-details"><small>Type: {layer.type} | Method: {layer.method}</small></div>
                  <div className="layer-range"><small>Range: {layer.color_range[0].toFixed(1)} to {layer.color_range[1].toFixed(1)} mm/year</small></div>
                  <div className="layer-path"><small>File: {layer.file_path}</small></div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {status?.results && (
        <div className="results-panel">
          <h4>Processing Statistics</h4>
          <div className="stats-grid">
            <div className="stat-item"><label>Processing Time:</label><span>{status.results.processing_time.toFixed(1)} minutes</span></div>
            <div className="stat-item"><label>Total Files:</label><span>{status.results.statistics.total_files}</span></div>
            <div className="stat-item"><label>SBAS Files:</label><span>{status.results.statistics.sbas_files}</span></div>
            <div className="stat-item"><label>PS Files:</label><span>{status.results.statistics.ps_files}</span></div>
            {status.results.statistics.sbas_velocity_mean && <div className="stat-item"><label>SBAS Avg Velocity:</label><span>{status.results.statistics.sbas_velocity_mean.toFixed(2)} mm/year</span></div>}
            {status.results.statistics.ps_velocity_mean && <div className="stat-item"><label>PS Avg Velocity:</label><span>{status.results.statistics.ps_velocity_mean.toFixed(2)} mm/year</span></div>}
          </div>

          {comparisonImage && (
            <div className="comparison-section">
              <h4>SBAS vs PS Comparison</h4>
              <img src={comparisonImage} alt="SBAS vs PS Comparison" className="comparison-plot" />
            </div>
          )}
        </div>
      )}
    </div>
  );

  const DebugPanel: React.FC = () => (
    <div className="debug-panel">
      <h4>Debug Info</h4>
      <div className="debug-messages">
        {debugInfo.map((info, index) => (<div key={index} className="debug-message">{info}</div>))}
      </div>
    </div>
  );

  const Legend: React.FC = () => (
    <div className="value-legend">
      <div className="legend-title">Velocity (mm/yr)</div>
      <div className="legend-gradient" />
      <div className="legend-scale">
        <span>{dataRange.min.toFixed(1)}</span>
        <span>0</span>
        <span>{dataRange.max.toFixed(1)}</span>
      </div>
      <div className="legend-notes">
        Transparent: |v| ≤ {threshold.toFixed(1)} or NoData
      </div>
    </div>
  );

  // -------- Render --------
  return (
    <div className="app" onDrop={onDrop} onDragOver={onDragOver}>
      <header className="app-header">
        <h1>Germany InSAR Analysis & GeoTIFF Viewer</h1>
        <p>SBAS and PS interferometric analysis with Sentinel-1 data search, download, and interactive GeoTIFF visualization</p>
        <ConnectionStatus />
      </header>

      <main className="app-main">
        <div className="control-panel">
          <TabNavigation />
          <GeoTIFFControls />
          <SearchControls />
          <ProcessingControls />
          <DebugPanel />
        </div>

        <div className="map-container">
          <div ref={mapRef} className="map" style={{ width: '100%', height: '100%' }} />

          {loading && (
            <div style={{
              position: 'absolute', inset: 0, background: 'rgba(255, 255, 255, 0.8)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              zIndex: 1000, fontSize: '1.2rem'
            }}>
              Loading file...
            </div>
          )}

          {popup && (
            <div
              style={{
                position: 'absolute', left: 20, top: 20,
                background: 'rgba(0,0,0,0.8)', color: '#fff', padding: '6px 10px',
                borderRadius: 4, pointerEvents: 'none', fontSize: 12, zIndex: 100
              }}
            >
              {popup.value.toFixed(2)} mm/yr<br />
              lon: {popup.lon.toFixed(4)}<br />
              lat: {popup.lat.toFixed(4)}
            </div>
          )}

          <div className="map-legend">
            <h4>Germany InSAR Map</h4>
            <div className="map-info">
              <small>Map Center: Germany</small>
              <small>Click map to see pixel values</small>
              <small>Drag & drop GeoTIFF files</small>
              {fileName && <small>Current: {fileName}</small>}
              {boundingBox && <small>Bounding box defined</small>}
            </div>
          </div>

          
          <Legend />
        </div>
      </main>
    </div>
  );
};

  export default App;
