#!/usr/bin/env python3
"""
Enhanced Master Thesis InSAR Processing Chain Comparison Tool
Now includes BBD CSV data comparison alongside PyGMTSAR vs Snap2StaMPS

NEW FEATURES:
- BBD CSV data loading and processing
- Time series filtering (Jan 2019 - June 2021)
- BBD vs Snap2StaMPS comparison
- BBD vs PyGMTSAR comparison
- Enhanced visualization with BBD data inclusion

Author: Simon Korfmacher
Purpose: Master Thesis Analysis (Enhanced)
"""
N_NEIGHBORS_PSI = 13
N_NEIGHBORS_SBAS = 10
N_NEIGHBORS_BBD = 8
MAX_DISTANCE_PSI = 25
MAX_DISTANCE_SBAS = 15
MAX_DISTANCE_BBD = 30
USE_BEST_FIT = True
USE_PERCENTILE_FILTER = True
PERCENTILE_RANGE = (2, 98)

BBD_START_DATE = "20190101"
BBD_END_DATE = "20210601"

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import distance_matrix
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
import glob
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Will use scipy distance matrix (may use more memory)")

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif'
})

class InSARComparison:
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.study_areas = []
        self.overall_results = {}
        self.setup_logging()
        
    def setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('insar_comparison_enhanced_thesis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def discover_study_areas(self):
        self.logger.info("Discovering study areas...")
        
        for subdir in self.results_dir.iterdir():
            if subdir.is_dir():
                tif_files = list(subdir.glob('*.tif'))
                csv_files = [f for f in subdir.glob('*.csv') if 'detailed_results' not in f.name]
                BBD_files = [f for f in subdir.glob('BBD*.csv')]
                
                if tif_files and csv_files:
                    area_info = {
                        'name': subdir.name,
                        'path': subdir,
                        'psi_files': [f for f in tif_files if any(x in f.name.lower() for x in ['psi', 'ps_'])],
                        'sbas_files': [f for f in tif_files if 'sbas' in f.name.lower()],
                        'csv_files': [f for f in csv_files if not f.name.startswith('BBD')],
                        'BBD_files': BBD_files
                    }
                    self.study_areas.append(area_info)
                    self.logger.info(f"Found study area: {subdir.name}")
                    self.logger.info(f"  PSI files: {[f.name for f in area_info['psi_files']]}")
                    self.logger.info(f"  SBAS files: {[f.name for f in area_info['sbas_files']]}")
                    self.logger.info(f"  CSV files: {[f.name for f in area_info['csv_files']]}")
                    self.logger.info(f"  BBD files: {[f.name for f in area_info['BBD_files']]}")
        
        self.logger.info(f"Total study areas found: {len(self.study_areas)}")
        return self.study_areas
    
    def load_BBD_data(self, BBD_files):
        combined_BBD = pd.DataFrame()
        
        for BBD_file in BBD_files:
            self.logger.info(f"Loading BBD data from: {BBD_file.name}")
            
            try:
                BBD_df = pd.read_csv(BBD_file)
                
                basic_cols = ['pid', 'latitude', 'longitude', 'mean_velocity']
                BBD_basic = BBD_df[basic_cols].copy()
                
                date_columns = [col for col in BBD_df.columns if col.isdigit() and len(col) == 8]
                date_columns.sort()
                
                self.logger.info(f"Found {len(date_columns)} date columns from {date_columns[0] if date_columns else 'N/A'} to {date_columns[-1] if date_columns else 'N/A'}")
                
                filtered_dates = []
                for date_col in date_columns:
                    if BBD_START_DATE <= date_col <= BBD_END_DATE:
                        filtered_dates.append(date_col)
                
                self.logger.info(f"Using {len(filtered_dates)} dates from {BBD_START_DATE} to {BBD_END_DATE}")
                self.logger.info(f"Date range: {filtered_dates[0] if filtered_dates else 'N/A'} to {filtered_dates[-1] if filtered_dates else 'N/A'}")
                
                if filtered_dates:
                    time_series_data = BBD_df[filtered_dates]
                    BBD_basic = pd.concat([BBD_basic, time_series_data], axis=1)
                
                BBD_basic = BBD_basic.rename(columns={
                    'latitude': 'Latitude',
                    'longitude': 'Longitude', 
                    'mean_velocity': 'Velocity_mm_year',
                    'pid': 'Point_ID'
                })
                
                BBD_basic['Source_File'] = BBD_file.stem
                BBD_basic['Data_Type'] = 'BBD'
                
                combined_BBD = pd.concat([combined_BBD, BBD_basic], ignore_index=True)
                
            except Exception as e:
                self.logger.error(f"Error loading BBD file {BBD_file.name}: {str(e)}")
                continue
        
        if len(combined_BBD) > 0:
            self.logger.info(f"Combined {len(combined_BBD)} BBD points from {len(BBD_files)} files")
            self.logger.info(f"BBD velocity range: {combined_BBD.Velocity_mm_year.min():.2f} to {combined_BBD.Velocity_mm_year.max():.2f} mm/year")
            self.logger.info(f"BBD velocity statistics: μ={combined_BBD.Velocity_mm_year.mean():.2f}, σ={combined_BBD.Velocity_mm_year.std():.2f} mm/year")
            
            if USE_PERCENTILE_FILTER:
                self.logger.info("Applying percentile filtering to BBD data...")
                p_low, p_high = PERCENTILE_RANGE
                lower_bound = np.percentile(combined_BBD.Velocity_mm_year, p_low)
                upper_bound = np.percentile(combined_BBD.Velocity_mm_year, p_high)
                
                before_filter = len(combined_BBD)
                combined_BBD = combined_BBD[
                    (combined_BBD.Velocity_mm_year >= lower_bound) & 
                    (combined_BBD.Velocity_mm_year <= upper_bound)
                ].reset_index(drop=True)
                after_filter = len(combined_BBD)
                
                self.logger.info(f"BBD filtering: {before_filter} -> {after_filter} points ({100*(before_filter-after_filter)/before_filter:.1f}% removed)")
                self.logger.info(f"Filtered BBD velocity range: {combined_BBD.Velocity_mm_year.min():.2f} to {combined_BBD.Velocity_mm_year.max():.2f} mm/year")
        
        return combined_BBD
    
    def load_snap2stamps_data(self, csv_files):
        combined_df = pd.DataFrame()
        
        for csv_file in csv_files:
            self.logger.info(f"Loading Snap2StaMPS data from: {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            if len(df.columns) == 4:
                df.columns = ['Point_ID', 'Longitude', 'Latitude', 'Velocity_mm_year']
            else:
                df = df.rename(columns={
                    df.columns[1]: 'Longitude',
                    df.columns[2]: 'Latitude', 
                    df.columns[3]: 'Velocity_mm_year'
                })
                df['Point_ID'] = range(len(df))
            
            df['Source_File'] = csv_file.stem
            df['Data_Type'] = 'Snap2StaMPS'
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            combined_df['Velocity_mm_year'] = combined_df['Velocity_mm_year'] 
        
        self.logger.info(f"Combined {len(combined_df)} Snap2StaMPS points from {len(csv_files)} files")
        self.logger.info(f"Velocity range: {combined_df.Velocity_mm_year.min():.2f} to {combined_df.Velocity_mm_year.max():.2f} mm/year")
        
        return combined_df
    
    def load_pygmtsar_data(self, tif_file, apply_sign_flip=False):
        self.logger.info(f"Loading PyGMTSAR data from: {tif_file.name}")
        
        with rasterio.open(tif_file) as src:
            velocity_data = src.read(1)
            transform = src.transform
            
            self.logger.info(f"Data type: {src.dtypes[0]}")
            self.logger.info(f"NoData value: {src.nodata}")
            self.logger.info(f"Raw data range: {velocity_data.min():.2f} to {velocity_data.max():.2f} mm/year")
            self.logger.info(f"Raw data shape: {velocity_data.shape}")
            
            height, width = velocity_data.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            lons = np.array(xs).flatten()
            lats = np.array(ys).flatten()
            velocities = velocity_data.flatten()
            
            if apply_sign_flip:
                velocities = -velocities
                self.logger.info("Applied sign flip to PyGMTSAR values")
            
            extreme_threshold = 1e10
            basic_valid_mask = (velocities != 0.0) & (~np.isnan(velocities)) & \
                            (~np.isinf(velocities)) & (np.abs(velocities) < extreme_threshold)

            if src.nodata is not None:
                basic_valid_mask = basic_valid_mask & (velocities != src.nodata) & (velocities != -src.nodata)

            basic_valid_velocities = velocities[basic_valid_mask]
            self.logger.info(f"Valid non-zero pixels: {len(basic_valid_velocities):,}")
            
            if USE_PERCENTILE_FILTER and len(basic_valid_velocities) > 100:
                p_low, p_high = PERCENTILE_RANGE
                lower_bound = np.percentile(basic_valid_velocities, p_low)
                upper_bound = np.percentile(basic_valid_velocities, p_high)
                
                self.logger.info(f"Percentile filtering ({p_low}th-{p_high}th percentile):")
                self.logger.info(f"  Lower bound: {lower_bound:.2f} mm/year")
                self.logger.info(f"  Upper bound: {upper_bound:.2f} mm/year")
                
                outliers_low = np.sum(basic_valid_velocities < lower_bound)
                outliers_high = np.sum(basic_valid_velocities > upper_bound)
                total_outliers = outliers_low + outliers_high
                
                self.logger.info(f"  Outliers below {lower_bound:.2f}: {outliers_low}")
                self.logger.info(f"  Outliers above {upper_bound:.2f}: {outliers_high}")
                self.logger.info(f"  Total outliers removed: {total_outliers} ({100*total_outliers/len(basic_valid_velocities):.2f}%)")
                
                percentile_mask = (velocities >= lower_bound) & (velocities <= upper_bound)
                final_valid_mask = basic_valid_mask & percentile_mask
                
            else:
                self.logger.info("Percentile filtering disabled or insufficient data")
                final_valid_mask = basic_valid_mask
            
            filtered_velocities = velocities[final_valid_mask]
            filtered_velocities = filtered_velocities 
            
            valid_df = pd.DataFrame({
                'Longitude': lons[final_valid_mask],
                'Latitude': lats[final_valid_mask],
                'Velocity_mm_year': filtered_velocities,
                'Data_Type': 'PyGMTSAR'
            })
            
            self.logger.info(f"Final dataset: {len(valid_df):,} valid PyGMTSAR pixels")
            self.logger.info(f"Final velocity statistics: μ={valid_df.Velocity_mm_year.mean():.2f}, σ={valid_df.Velocity_mm_year.std():.2f} mm/year")
            
            return valid_df, src.bounds
    
    def find_nearest_neighbors(self, reference_points, target_points, max_distance_m=15, n_neighbors=15):
        if USE_BEST_FIT:
            self.logger.info(f"Finding best-fit neighbors from {n_neighbors} candidates for robust comparison...")
        else:
            self.logger.info(f"Finding {n_neighbors} nearest neighbors for averaging...")
        
        memory_required_gb = (len(reference_points) * len(target_points) * 8) / (1024**3)
        self.logger.info(f"Estimated memory requirement: {memory_required_gb:.1f} GB")
        
        if memory_required_gb > 10 and HAS_SKLEARN:
            self.logger.info("Using KDTree for memory-efficient neighbor search")
            return self._find_neighbors_kdtree(reference_points, target_points, max_distance_m, n_neighbors)
        elif memory_required_gb > 20:
            self.logger.warning(f"Large memory requirement ({memory_required_gb:.1f} GB), using sampling approach")
            return self._find_neighbors_with_sampling(reference_points, target_points, max_distance_m)
        else:
            self.logger.info("Using distance matrix for nearest neighbor search")
            return self._find_neighbors_distance_matrix(reference_points, target_points, max_distance_m)
    
    def _find_best_fit_neighbor(self, reference_velocity, neighbor_velocities, neighbor_distances, neighbor_indices):
        if len(neighbor_velocities) == 1:
            return 0
            
        velocity_diffs = np.abs(neighbor_velocities - reference_velocity)
        
        normalized_vel_diffs = velocity_diffs / (velocity_diffs.max() + 1e-10)
        normalized_distances = neighbor_distances / (neighbor_distances.max() + 1e-10)
        
        combined_scores = 0.7 * normalized_vel_diffs + 0.3 * normalized_distances
        
        best_idx = np.argmin(combined_scores)
        return best_idx
    
    def _find_neighbors_kdtree(self, reference_points, target_points, max_distance_m, n_neighbors=12):
        if not HAS_SKLEARN:
            self.logger.error("KDTree method requested but sklearn not available")
            return self._find_neighbors_with_sampling(reference_points, target_points, max_distance_m)
        
        lat_mean = reference_points.Latitude.mean()
        m_per_deg_lon = 111320 * np.cos(np.radians(lat_mean))
        m_per_deg_lat = 111320
        
        ref_coords = np.column_stack([
            reference_points.Longitude * m_per_deg_lon,
            reference_points.Latitude * m_per_deg_lat
        ])
        
        target_coords = np.column_stack([
            target_points.Longitude * m_per_deg_lon,
            target_points.Latitude * m_per_deg_lat
        ])
        
        actual_n_neighbors = min(n_neighbors, len(target_points))
        self.logger.info(f"Using {actual_n_neighbors} neighbor candidates for best-fit selection")
        
        nbrs = NearestNeighbors(n_neighbors=actual_n_neighbors, algorithm='kd_tree').fit(target_coords)
        distances, indices = nbrs.kneighbors(ref_coords)
        
        matched_data_list = []
        valid_matches = 0
        
        for i, ref_point in reference_points.iterrows():
            point_distances = distances[i]
            point_indices = indices[i]
            
            valid_neighbor_mask = point_distances <= max_distance_m
            
            if not np.any(valid_neighbor_mask):
                continue
            
            valid_distances = point_distances[valid_neighbor_mask]
            valid_indices = point_indices[valid_neighbor_mask]
            
            neighbor_velocities = target_points.iloc[valid_indices]['Velocity_mm_year'].values
            neighbor_coords = target_points.iloc[valid_indices][['Longitude', 'Latitude']].values
            
            if USE_BEST_FIT:
                best_neighbor_idx = self._find_best_fit_neighbor(
                    ref_point['Velocity_mm_year'], 
                    neighbor_velocities, 
                    valid_distances, 
                    valid_indices
                )
                
                selected_velocity = neighbor_velocities[best_neighbor_idx]
                selected_longitude = neighbor_coords[best_neighbor_idx, 0]
                selected_latitude = neighbor_coords[best_neighbor_idx, 1]
                selected_distance = valid_distances[best_neighbor_idx]
                
                velocity_diff = abs(selected_velocity - ref_point['Velocity_mm_year'])
                n_candidates = len(valid_distances)
                
            else:
                if len(valid_distances) == 1:
                    selected_velocity = neighbor_velocities[0]
                    selected_longitude = neighbor_coords[0, 0]
                    selected_latitude = neighbor_coords[0, 1]
                    selected_distance = valid_distances[0]
                    velocity_diff = abs(selected_velocity - ref_point['Velocity_mm_year'])
                    n_candidates = 1
                else:
                    weights = 1.0 / (valid_distances + 1e-10)
                    weights = weights / weights.sum()
                    
                    selected_velocity = np.average(neighbor_velocities, weights=weights)
                    selected_longitude = np.average(neighbor_coords[:, 0], weights=weights)
                    selected_latitude = np.average(neighbor_coords[:, 1], weights=weights)
                    selected_distance = np.average(valid_distances, weights=weights)
                    velocity_diff = abs(selected_velocity - ref_point['Velocity_mm_year'])
                    n_candidates = len(valid_distances)
            
            matched_data_list.append({
                'Point_ID': ref_point['Point_ID'],
                'Longitude_Ref': ref_point['Longitude'],
                'Latitude_Ref': ref_point['Latitude'],
                'Velocity_Ref': ref_point['Velocity_mm_year'],
                'Longitude_Target': selected_longitude,
                'Latitude_Target': selected_latitude,
                'Velocity_Target': selected_velocity,
                'Distance_m': selected_distance,
                'N_Candidates': n_candidates,
                'Velocity_Difference': velocity_diff,
                'Neighbor_Velocity_Std': np.std(neighbor_velocities) if len(neighbor_velocities) > 1 else 0.0,
                'Source_File': ref_point.get('Source_File', 'unknown'),
                'Ref_Data_Type': ref_point.get('Data_Type', 'unknown'),
                'Target_Data_Type': target_points.iloc[valid_indices[best_neighbor_idx if USE_BEST_FIT else 0]].get('Data_Type', 'unknown') if len(valid_indices) > 0 else 'unknown'
            })
            
            valid_matches += 1
        
        if not matched_data_list:
            self.logger.warning("No valid matches found within distance threshold")
            return pd.DataFrame()
        
        matched_data = pd.DataFrame(matched_data_list)
        
        method_str = "best-fit selection" if USE_BEST_FIT else "averaging"
        self.logger.info(f"Found {valid_matches} matches within {max_distance_m}m using {method_str}")
        self.logger.info(f"Mean distance: {matched_data.Distance_m.mean():.1f}m")
        self.logger.info(f"Mean candidates considered: {matched_data.N_Candidates.mean():.1f}")
        self.logger.info(f"Mean velocity difference: {matched_data.Velocity_Difference.mean():.2f} mm/year")
        
        return matched_data
    
    def _find_neighbors_with_sampling(self, reference_points, target_points, max_distance_m):
        self.logger.info("Using sampling approach to reduce memory usage")
        
        max_target_points = 50000
        if len(target_points) > max_target_points:
            self.logger.info(f"Sampling {max_target_points} from {len(target_points)} target points")
            target_sample = target_points.sample(n=max_target_points, random_state=42)
        else:
            target_sample = target_points
        
        return self._find_neighbors_distance_matrix(reference_points, target_sample, max_distance_m)
    
    def _find_neighbors_distance_matrix(self, reference_points, target_points, max_distance_m):
        lat_mean = reference_points.Latitude.mean()
        m_per_deg_lon = 111320 * np.cos(np.radians(lat_mean))
        m_per_deg_lat = 111320
        
        ref_coords = np.column_stack([
            reference_points.Longitude * m_per_deg_lon,
            reference_points.Latitude * m_per_deg_lat
        ])
        
        target_coords = np.column_stack([
            target_points.Longitude * m_per_deg_lon,
            target_points.Latitude * m_per_deg_lat
        ])
        
        distances = distance_matrix(ref_coords, target_coords)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = np.min(distances, axis=1)
        
        valid_matches = nearest_distances <= max_distance_m
        
        self.logger.info(f"Found {np.sum(valid_matches)} matches within {max_distance_m}m")
        if np.sum(valid_matches) > 0:
            self.logger.info(f"Mean distance: {nearest_distances[valid_matches].mean():.1f}m")
        
        ref_valid = reference_points[valid_matches].reset_index(drop=True)
        target_valid = target_points.iloc[nearest_indices[valid_matches]].reset_index(drop=True)
        
        velocity_diffs = abs(ref_valid.Velocity_mm_year - target_valid.Velocity_mm_year)
        
        matched_data = pd.DataFrame({
            'Point_ID': ref_valid.Point_ID,
            'Longitude_Ref': ref_valid.Longitude,
            'Latitude_Ref': ref_valid.Latitude,
            'Velocity_Ref': ref_valid.Velocity_mm_year,
            'Longitude_Target': target_valid.Longitude,
            'Latitude_Target': target_valid.Latitude,
            'Velocity_Target': target_valid.Velocity_mm_year,
            'Distance_m': nearest_distances[valid_matches],
            'N_Candidates': 1,
            'Velocity_Difference': velocity_diffs,
            'Source_File': ref_valid.get('Source_File', 'unknown'),
            'Ref_Data_Type': ref_valid.get('Data_Type', 'unknown'),
            'Target_Data_Type': target_valid.get('Data_Type', 'unknown')
        })
        
        return matched_data
    
    def calculate_statistics(self, matched_data):
        print(f"Columns available: {matched_data.columns.tolist()}")
        print(f"Sample velocity values - Ref: {matched_data.Velocity_Ref.head()}")
        print(f"Sample velocity values - Target: {matched_data.Velocity_Target.head()}")
        ref_vel = matched_data.Velocity_Ref
        target_vel = matched_data.Velocity_Target
        differences = target_vel - ref_vel
        
        correlation, p_value = stats.pearsonr(ref_vel, target_vel)
        
        slope, intercept, r_value, p_reg, std_err = stats.linregress(ref_vel, target_vel)
        
        stats_dict = {
            'n_points': len(matched_data),
            'mean_ref': ref_vel.mean(),
            'std_ref': ref_vel.std(),
            'mean_target': target_vel.mean(),
            'std_target': target_vel.std(),
            'mean_difference': differences.mean(),
            'std_difference': differences.std(),
            'rmse': np.sqrt(np.mean(differences**2)),
            'mae': np.mean(np.abs(differences)),
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'r_squared': r_value**2,
            'regression_slope': slope,
            'regression_intercept': intercept,
            'regression_pvalue': p_reg,
            'min_difference': differences.min(),
            'max_difference': differences.max(),
            'median_difference': differences.median(),
            'q25_difference': differences.quantile(0.25),
            'q75_difference': differences.quantile(0.75),
            'bias': differences.mean(),
            'bias_percentage': (differences.mean() / ref_vel.mean()) * 100 if ref_vel.mean() != 0 else 0,
            'mean_velocity_diff': matched_data.Velocity_Difference.mean() if 'Velocity_Difference' in matched_data.columns else np.nan,
            'mean_candidates': matched_data.N_Candidates.mean() if 'N_Candidates' in matched_data.columns else np.nan
        }
        
        return stats_dict, differences
    
    def create_enhanced_comparison_plots(self, area_name, comparison_name, matched_data, differences, stats_dict, output_dir=None):
        
        if output_dir is None:
            output_dir = Path('.')
        
        AGREEMENT_THRESHOLD = 5.0
        within_threshold_mask = np.abs(differences) <= AGREEMENT_THRESHOLD
        agreement_count = within_threshold_mask.sum()
        total_count = len(differences)
        agreement_percentage = (agreement_count / total_count) * 100
        
        stats_dict['agreement_threshold_mmyr'] = AGREEMENT_THRESHOLD
        stats_dict['points_within_threshold'] = agreement_count
        stats_dict['points_outside_threshold'] = total_count - agreement_count
        stats_dict['agreement_rate_percentage'] = agreement_percentage
        
        self.logger.info(f"Agreement Quality Analysis:")
        self.logger.info(f"  • Points within ±{AGREEMENT_THRESHOLD} mm/year: {agreement_count}/{total_count} ({agreement_percentage:.1f}%)")
        
        colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C',
            'accent': '#28B463',
            'neutral': '#566573',
            'BBD': '#8E44AD',
            'snap': '#F39C12',
            'pygmtsar': '#17A2B8',
            'good_agreement': '#27AE60',
            'poor_agreement': '#E74C3C'
        }
        
        ref_type = matched_data.iloc[0]['Ref_Data_Type'] if 'Ref_Data_Type' in matched_data.columns else 'Reference'
        target_type = matched_data.iloc[0]['Target_Data_Type'] if 'Target_Data_Type' in matched_data.columns else 'Target'
        
        if 'BBD' in ref_type:
            ref_color = colors['BBD']
        elif 'Snap' in ref_type:
            ref_color = colors['snap']
        else:
            ref_color = colors['primary']
            
        if 'BBD' in target_type:
            target_color = colors['BBD']
        elif 'PyGMTSAR' in target_type:
            target_color = colors['pygmtsar']
        else:
            target_color = colors['secondary']
        
        combined_velocities = np.concatenate([
            matched_data.Velocity_Ref.values,
            matched_data.Velocity_Target.values
        ])
        vmin_global = np.percentile(combined_velocities, 2)
        vmax_global = np.percentile(combined_velocities, 98)
        
        method_str = "Best-Fit Selection" if USE_BEST_FIT else "Multi-Neighbor Averaging"
        filter_str = " + Percentile Filtering" if USE_PERCENTILE_FILTER else ""
        
        fig1 = plt.figure(figsize=(18, 12))
        
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(matched_data.Velocity_Ref, matched_data.Velocity_Target, 
                            alpha=0.7, s=80, c=colors['primary'], edgecolors='white', linewidth=1)
        
        x_range = np.linspace(matched_data.Velocity_Ref.min(), matched_data.Velocity_Ref.max(), 100)
        y_pred = stats_dict['regression_slope'] * x_range + stats_dict['regression_intercept']
        ax1.plot(x_range, y_pred, color=colors['secondary'], linewidth=2.5, 
                label=f'Regression: y = {stats_dict["regression_slope"]:.3f}x + {stats_dict["regression_intercept"]:.3f}')
        
        min_val = min(matched_data.Velocity_Ref.min(), matched_data.Velocity_Target.min())
        max_val = max(matched_data.Velocity_Ref.max(), matched_data.Velocity_Target.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5, alpha=0.8, label='1:1 line')
        
        ax1.set_xlabel(f'{ref_type} Velocity (mm/year)', fontweight='bold', fontsize=13)
        ax1.set_ylabel(f'{target_type} Velocity (mm/year)', fontweight='bold', fontsize=13)
        ax1.set_title(f'{area_name}: {comparison_name}\n{method_str}{filter_str}\n$R^2$ = {stats_dict["r_squared"]:.3f}, RMSE = {stats_dict["rmse"]:.2f} mm/year', 
                    fontweight='bold', fontsize=14)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.tick_params(labelsize=11)
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(matched_data.Velocity_Ref, differences, alpha=0.7, s=80, 
                c=colors['accent'], edgecolors='white', linewidth=1)
        ax2.axhline(0, color=colors['secondary'], linestyle='--', linewidth=2.5, alpha=0.8)
        ax2.axhline(differences.mean(), color=colors['neutral'], linestyle=':', linewidth=2.5, 
                label=f'Mean bias: {differences.mean():.2f} mm/year')
        ax2.set_xlabel(f'{ref_type} Velocity (mm/year)', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Residuals (mm/year)', fontweight='bold', fontsize=13)
        ax2.set_title('Residual Analysis', fontweight='bold', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=11)
        
        ax3 = plt.subplot(2, 3, 3)
        n, bins, patches = ax3.hist(differences, bins=30, alpha=0.8, color=colors['primary'], 
                                    edgecolor='white', linewidth=1.5)
        ax3.axvline(differences.mean(), color=colors['secondary'], linestyle='--', linewidth=2.5,
                label=f'Mean: {differences.mean():.2f} mm/year')
        ax3.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        ax3.axvline(AGREEMENT_THRESHOLD, color=colors['good_agreement'], linestyle=':', linewidth=2.5,
                label=f'±{AGREEMENT_THRESHOLD} mm/year threshold')
        ax3.axvline(-AGREEMENT_THRESHOLD, color=colors['good_agreement'], linestyle=':', linewidth=2.5)
        ax3.set_xlabel('Velocity Difference (mm/year)', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Frequency', fontweight='bold', fontsize=13)
        ax3.set_title('Distribution of Differences', fontweight='bold', fontsize=14)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(labelsize=11)
        
        ax4 = plt.subplot(2, 3, 4)
        stats.probplot(differences, dist="norm", plot=ax4)
        ax4.set_title('Normality Test (Q-Q Plot)', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=11)
        line = ax4.get_lines()[0]
        line.set_linewidth(2.5)
        line.set_color(colors['secondary'])
        
        ax5 = plt.subplot(2, 3, 5)
        box_data = [matched_data.Velocity_Ref, matched_data.Velocity_Target]
        bp = ax5.boxplot(box_data, labels=[ref_type, target_type], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(ref_color)
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(target_color)
        bp['boxes'][1].set_alpha(0.8)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element in bp:
                plt.setp(bp[element], linewidth=2)
        ax5.set_ylabel('Velocity (mm/year)', fontweight='bold', fontsize=13)
        ax5.set_title('Method Comparison', fontweight='bold', fontsize=14)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.tick_params(labelsize=11)
        
        ax6 = plt.subplot(2, 3, 6)
        pie_data = [agreement_count, total_count - agreement_count]
        pie_labels = [f'Within ±{AGREEMENT_THRESHOLD} mm/yr\n({agreement_percentage:.1f}%)',
                    f'Outside threshold\n({100-agreement_percentage:.1f}%)']
        pie_colors = [colors['good_agreement'], colors['poor_agreement']]
        
        wedges, texts, autotexts = ax6.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                            autopct='%d pts', startangle=90, 
                                            textprops={'fontweight': 'bold', 'fontsize': 12})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
        ax6.set_title('Agreement Rate Distribution', fontweight='bold', fontsize=14, pad=20)
        
        plt.tight_layout(pad=3.0)
        
        method_suffix = "bestfit" if USE_BEST_FIT else "averaging"
        filter_suffix = "_filtered" if USE_PERCENTILE_FILTER else ""
        comparison_clean = comparison_name.replace(' ', '_').replace('vs', 'vs')
        
        plot_filename1 = output_dir / f'{area_name}_{comparison_clean}_{method_suffix}{filter_suffix}_stats.pdf'
        plt.savefig(plot_filename1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / f'{area_name}_{comparison_clean}_{method_suffix}{filter_suffix}_stats.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        self.logger.info(f"Saved statistical plots: {plot_filename1}")
        plt.close()
        
        fig2 = plt.figure(figsize=(18, 12))
        
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(matched_data.Longitude_Ref, matched_data.Latitude_Ref, 
                            c=matched_data.Velocity_Ref, s=100, cmap='turbo', 
                            vmin=vmin_global, vmax=vmax_global,
                            edgecolors='black', linewidth=0.6)
        cbar = plt.colorbar(scatter, ax=ax1, label='Velocity (mm/year)')
        cbar.ax.tick_params(labelsize=11)
        ax1.set_xlabel('Longitude', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Latitude', fontweight='bold', fontsize=13)
        ax1.set_title(f'{ref_type} Velocity Field', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=11)
        
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(matched_data.Longitude_Target, matched_data.Latitude_Target, 
                            c=matched_data.Velocity_Target, s=100, cmap='turbo',
                            vmin=vmin_global, vmax=vmax_global,
                            edgecolors='black', linewidth=0.6)
        cbar = plt.colorbar(scatter, ax=ax2, label='Velocity (mm/year)')
        cbar.ax.tick_params(labelsize=11)
        ax2.set_xlabel('Longitude', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Latitude', fontweight='bold', fontsize=13)
        ax2.set_title(f'{target_type} Velocity Field', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=11)
        
        ax3 = plt.subplot(2, 3, 3)
        max_abs_diff = np.percentile(np.abs(differences), 98)
        scatter = ax3.scatter(matched_data.Longitude_Ref, matched_data.Latitude_Ref, 
                            c=differences, s=100, cmap='RdBu_r', 
                            vmin=-max_abs_diff, vmax=max_abs_diff,
                            edgecolors='black', linewidth=0.6)
        cbar = plt.colorbar(scatter, ax=ax3, label='Difference (mm/year)')
        cbar.ax.tick_params(labelsize=11)
        ax3.set_xlabel('Longitude', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Latitude', fontweight='bold', fontsize=13)
        ax3.set_title('Spatial Residuals (All Points)', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=11)
        
        ax4 = plt.subplot(2, 3, 4)
        high_agreement_data = matched_data[within_threshold_mask]
        high_agreement_diffs = differences[within_threshold_mask]
        
        if len(high_agreement_data) > 0:
            scatter = ax4.scatter(high_agreement_data.Longitude_Ref, high_agreement_data.Latitude_Ref, 
                                c=high_agreement_diffs, s=100, cmap='RdYlGn_r', 
                                vmin=-AGREEMENT_THRESHOLD, vmax=AGREEMENT_THRESHOLD,
                                edgecolors='darkgreen', linewidth=1.0, alpha=0.9)
            cbar = plt.colorbar(scatter, ax=ax4, label='Difference (mm/year)')
            cbar.ax.tick_params(labelsize=11)
            ax4.set_title(f'High-Agreement Points Map\n({agreement_count} pts, {agreement_percentage:.1f}% of total)', 
                        fontweight='bold', fontsize=14, color=colors['good_agreement'])
            
            ax4.text(0.02, 0.98, f'|Δ| ≤ {AGREEMENT_THRESHOLD} mm/yr', 
                    transform=ax4.transAxes, fontsize=12, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen', linewidth=2))
        else:
            ax4.text(0.5, 0.5, 'No points within\nagreement threshold', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14, fontweight='bold')
            ax4.set_title(f'High-Agreement Points Map\n(0 pts, 0.0% of total)', fontweight='bold', fontsize=14)
        
        ax4.set_xlabel('Longitude', fontweight='bold', fontsize=13)
        ax4.set_ylabel('Latitude', fontweight='bold', fontsize=13)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=11)
        
        ax5 = plt.subplot(2, 3, 5)
        if 'Distance_m' in matched_data.columns:
            scatter_colors = np.where(within_threshold_mask, colors['good_agreement'], colors['poor_agreement'])
            ax5.scatter(matched_data.Distance_m, np.abs(differences),
                    alpha=0.7, s=80, c=scatter_colors, edgecolors='white', linewidth=1)
            ax5.axhline(AGREEMENT_THRESHOLD, color='black', linestyle='--', linewidth=2.5, 
                    label=f'Agreement threshold ({AGREEMENT_THRESHOLD} mm/yr)')
            ax5.set_xlabel('Distance (m)', fontweight='bold', fontsize=13)
            ax5.set_ylabel('|Velocity Difference| (mm/year)', fontweight='bold', fontsize=13)
            ax5.set_title('Distance vs Absolute Difference', fontweight='bold', fontsize=14)
            ax5.legend(fontsize=11)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(labelsize=11)
        
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        selection_info = ""
        if USE_BEST_FIT and 'N_Candidates' in matched_data.columns:
            selection_info = f"""Best-Fit Selection Method:
    - Mean candidates: {matched_data.N_Candidates.mean():.1f}
    - Mean velocity difference: {matched_data.Velocity_Difference.mean():.2f} mm/year
    - Selection: 70% velocity + 30% distance
    """
        
        filtering_info = ""
        if USE_PERCENTILE_FILTER:
            filtering_info = f"""
    Percentile Outlier Filtering:
    - Range: {PERCENTILE_RANGE[0]}th - {PERCENTILE_RANGE[1]}th percentile
    - Removes extreme outliers
    """
        
        stats_text = f"""{area_name.upper()} - {comparison_name.upper()}

    Dataset Information:
    - Matched points: {stats_dict['n_points']:,}
    - Mean distance: {matched_data.Distance_m.mean():.1f} m
    - Velocity range: [{vmin_global:.2f}, {vmax_global:.2f}] mm/year

    {selection_info}{filtering_info}
    Velocity Statistics (mm/year):
    - {ref_type}: {stats_dict['mean_ref']:.2f} ± {stats_dict['std_ref']:.2f}
    - {target_type}: {stats_dict['mean_target']:.2f} ± {stats_dict['std_target']:.2f}

    Agreement Metrics:
    - Correlation (R): {stats_dict['correlation']:.3f}
    - R-squared: {stats_dict['r_squared']:.3f}
    - RMSE: {stats_dict['rmse']:.2f} mm/year
    - MAE: {stats_dict['mae']:.2f} mm/year
    - Mean bias: {stats_dict['bias']:.2f} mm/year
    - Bias percentage: {stats_dict['bias_percentage']:.1f}%

    AGREEMENT QUALITY ANALYSIS:
    - Agreement threshold: ±{AGREEMENT_THRESHOLD} mm/year
    - High-agreement points: {agreement_count}/{total_count} ({agreement_percentage:.1f}%)
    - Agreement rate: {agreement_percentage:.1f}% ✓
    - Points outside threshold: {total_count - agreement_count} ({100-agreement_percentage:.1f}%)

    Statistical Tests:
    - Correlation p-value: {stats_dict['correlation_pvalue']:.2e}
    - Regression slope: {stats_dict['regression_slope']:.3f}
    - Regression p-value: {stats_dict['regression_pvalue']:.2e}

    Quality Assessment:
    - Strong correlation: {'✓' if stats_dict['correlation'] > 0.7 else '✗'}
    - Low bias (<2 mm/yr): {'✓' if abs(stats_dict['bias']) < 2 else '✗'}
    - Good agreement: {'✓' if stats_dict['rmse'] < 5 else '✗'}
    - High agreement rate: {'✓' if agreement_percentage > 70 else '✗'}
    """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.0', facecolor='lightgray', alpha=0.9, edgecolor='black', linewidth=2))
        
        plt.tight_layout(pad=3.0)
        
        plot_filename2 = output_dir / f'{area_name}_{comparison_clean}_{method_suffix}{filter_suffix}_spatial.pdf'
        plt.savefig(plot_filename2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / f'{area_name}_{comparison_clean}_{method_suffix}{filter_suffix}_spatial.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        self.logger.info(f"Saved spatial plots: {plot_filename2}")
        plt.close()
        
        return plot_filename1, plot_filename2
    
    def create_multi_dataset_comparison(self, area_name, snap_data, BBD_data, pygmtsar_data, output_dir):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        colors = {
            'snap': '#F39C12',
            'BBD': '#8E44AD',
            'pygmtsar': '#17A2B8'
        }
        
        ax = axes[0, 0]
        scatter = ax.scatter(snap_data.Longitude, snap_data.Latitude, 
                           c=snap_data.Velocity_mm_year, s=40, cmap='turbo', 
                           edgecolors='black', linewidth=0.3, alpha=0.8)
        ax.set_title('Snap2StaMPS Velocity Field', fontweight='bold', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax, label='Velocity (mm/year)')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        if len(BBD_data) > 0:
            scatter = ax.scatter(BBD_data.Longitude, BBD_data.Latitude, 
                               c=BBD_data.Velocity_mm_year, s=40, cmap='turbo', 
                               edgecolors='black', linewidth=0.3, alpha=0.8)
            ax.set_title('BBD Velocity Field', fontweight='bold', fontsize=14)
            plt.colorbar(scatter, ax=ax, label='Velocity (mm/year)')
        else:
            ax.text(0.5, 0.5, 'No BBD Data\nAvailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('BBD Velocity Field', fontweight='bold', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        if len(pygmtsar_data) > 10000:
            pygmtsar_sample = pygmtsar_data.sample(n=10000, random_state=42)
        else:
            pygmtsar_sample = pygmtsar_data
        
        scatter = ax.scatter(pygmtsar_sample.Longitude, pygmtsar_sample.Latitude, 
                           c=pygmtsar_sample.Velocity_mm_year, s=20, cmap='turbo', 
                           edgecolors='none', alpha=0.7)
        ax.set_title('PyGMTSAR Velocity Field', fontweight='bold', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax, label='Velocity (mm/year)')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.hist(snap_data.Velocity_mm_year, bins=30, alpha=0.7, color=colors['snap'], 
               label=f'Snap2StaMPS (n={len(snap_data):,})', density=True)
        if len(BBD_data) > 0:
            ax.hist(BBD_data.Velocity_mm_year, bins=30, alpha=0.7, color=colors['BBD'], 
                   label=f'BBD (n={len(BBD_data):,})', density=True)
        ax.hist(pygmtsar_data.Velocity_mm_year, bins=30, alpha=0.7, color=colors['pygmtsar'], 
               label=f'PyGMTSAR (n={len(pygmtsar_data):,})', density=True)
        ax.set_xlabel('Velocity (mm/year)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Velocity Distribution Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        datasets = ['Snap2StaMPS', 'PyGMTSAR']
        means = [snap_data.Velocity_mm_year.mean(), pygmtsar_data.Velocity_mm_year.mean()]
        stds = [snap_data.Velocity_mm_year.std(), pygmtsar_data.Velocity_mm_year.std()]
        
        if len(BBD_data) > 0:
            datasets.append('BBD')
            means.append(BBD_data.Velocity_mm_year.mean())
            stds.append(BBD_data.Velocity_mm_year.std())
        
        x_pos = np.arange(len(datasets))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=[colors['snap'], colors['pygmtsar'], colors['BBD']][:len(datasets)],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Mean Velocity ± Std (mm/year)', fontweight='bold')
        ax.set_title('Statistical Comparison', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(datasets)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                   f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        {area_name.upper()} - DATASET OVERVIEW
        
        Dataset Statistics:
        
        Snap2StaMPS:
        • Points: {len(snap_data):,}
        • Mean velocity: {snap_data.Velocity_mm_year.mean():.2f} mm/year
        • Std deviation: {snap_data.Velocity_mm_year.std():.2f} mm/year
        • Range: {snap_data.Velocity_mm_year.min():.1f} to {snap_data.Velocity_mm_year.max():.1f}
        
        PyGMTSAR:
        • Points: {len(pygmtsar_data):,}
        • Mean velocity: {pygmtsar_data.Velocity_mm_year.mean():.2f} mm/year
        • Std deviation: {pygmtsar_data.Velocity_mm_year.std():.2f} mm/year
        • Range: {pygmtsar_data.Velocity_mm_year.min():.1f} to {pygmtsar_data.Velocity_mm_year.max():.1f}
        """
        
        if len(BBD_data) > 0:
            summary_text += f"""
        BBD:
        • Points: {len(BBD_data):,}
        • Mean velocity: {BBD_data.Velocity_mm_year.mean():.2f} mm/year
        • Std deviation: {BBD_data.Velocity_mm_year.std():.2f} mm/year
        • Range: {BBD_data.Velocity_mm_year.min():.1f} to {BBD_data.Velocity_mm_year.max():.1f}
        • Time period: {BBD_START_DATE} to {BBD_END_DATE}
        """
        else:
            summary_text += """
        BBD:
        • No BBD data available for this area
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        plot_filename = output_dir / f'{area_name}_multi_dataset_overview.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / f'{area_name}_multi_dataset_overview.pdf', bbox_inches='tight', facecolor='white')
        
        self.logger.info(f"Saved multi-dataset comparison: {plot_filename}")
        return plot_filename
    
    def process_single_area(self, area_info):
        area_name = area_info['name']
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PROCESSING STUDY AREA: {area_name.upper()} (ENHANCED WITH BBD)")
        self.logger.info(f"{'='*60}")
        
        area_log_file = area_info['path'] / f'{area_name}_enhanced_analysis.log'
        area_handler = logging.FileHandler(area_log_file)
        area_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(area_handler)

        method_suffix = "bestfit" if USE_BEST_FIT else "averaging"
        filter_suffix = "_filtered" if USE_PERCENTILE_FILTER else ""
        method_str = "best-fit" if USE_BEST_FIT else "averaging"  
        filter_str = " + filtering" if USE_PERCENTILE_FILTER else ""
        
        try:
            snap_data = self.load_snap2stamps_data(area_info['csv_files'])
            BBD_data = self.load_BBD_data(area_info['BBD_files']) if area_info['BBD_files'] else pd.DataFrame()
            
            results = {}
            
            pygmtsar_data = pd.DataFrame()
            if area_info['psi_files']:
                psi_file = area_info['psi_files'][0]
                pygmtsar_data, _ = self.load_pygmtsar_data(psi_file, apply_sign_flip=False)
            
            if len(pygmtsar_data) > 0:
                self.create_multi_dataset_comparison(area_name, snap_data, BBD_data, pygmtsar_data, area_info['path'])
            
            if len(BBD_data) > 0:
                self.logger.info("Processing BBD vs Snap2StaMPS comparison")
                
                matched_BBD_snap = self.find_nearest_neighbors(snap_data, BBD_data, 
                                                             max_distance_m=MAX_DISTANCE_BBD, 
                                                             n_neighbors=N_NEIGHBORS_BBD)
                
                if len(matched_BBD_snap) >= 5:
                    stats_BBD_snap, differences_BBD_snap = self.calculate_statistics(matched_BBD_snap)
                    
                    plot_file_BBD_snap = self.create_enhanced_comparison_plots(
                        area_name, "BBD vs Snap2StaMPS", matched_BBD_snap, differences_BBD_snap, stats_BBD_snap,
                        area_info['path']
                    )
                    
                    detailed_results_BBD_snap = matched_BBD_snap.copy()
                    detailed_results_BBD_snap['Difference'] = differences_BBD_snap
                    method_suffix = "bestfit" if USE_BEST_FIT else "averaging"
                    filter_suffix = "_filtered" if USE_PERCENTILE_FILTER else ""
                    detailed_results_BBD_snap.to_csv(
                        area_info['path'] / f'{area_name}_BBD_vs_Snap2StaMPS_detailed_results_{method_suffix}{filter_suffix}.csv', 
                        index=False
                    )
                    
                    results['BBD_vs_Snap2StaMPS'] = {
                        'statistics': stats_BBD_snap,
                        'matched_points': len(matched_BBD_snap),
                        'plot_file': str(plot_file_BBD_snap),
                        'comparison_type': 'BBD_vs_Snap2StaMPS',
                        'method': 'best_fit_selection' if USE_BEST_FIT else 'multi_neighbor_averaging',
                        'filtering': 'percentile' if USE_PERCENTILE_FILTER else 'none'
                    }
                    
                    method_str = "best-fit" if USE_BEST_FIT else "averaging"
                    filter_str = " + filtering" if USE_PERCENTILE_FILTER else ""
                    self.logger.info(f"Completed BBD vs Snap2StaMPS ({method_str}{filter_str}): R²={stats_BBD_snap['r_squared']:.3f}, RMSE={stats_BBD_snap['rmse']:.2f}")
                else:
                    self.logger.warning(f"Too few matches for BBD vs Snap2StaMPS comparison ({len(matched_BBD_snap)} points)")
            
            if len(BBD_data) > 0 and len(pygmtsar_data) > 0:
                self.logger.info("Processing BBD vs PyGMTSAR comparison")
                
                matched_BBD_pygmtsar = self.find_nearest_neighbors(BBD_data, pygmtsar_data, 
                                                                 max_distance_m=MAX_DISTANCE_BBD, 
                                                                 n_neighbors=N_NEIGHBORS_BBD)
                
                if len(matched_BBD_pygmtsar) >= 5:
                    stats_BBD_pygmtsar, differences_BBD_pygmtsar = self.calculate_statistics(matched_BBD_pygmtsar)
                    
                    plot_file_BBD_pygmtsar = self.create_enhanced_comparison_plots(
                        area_name, "BBD vs PyGMTSAR", matched_BBD_pygmtsar, differences_BBD_pygmtsar, stats_BBD_pygmtsar,
                        area_info['path']
                    )
                    
                    detailed_results_BBD_pygmtsar = matched_BBD_pygmtsar.copy()
                    detailed_results_BBD_pygmtsar['Difference'] = differences_BBD_pygmtsar
                    detailed_results_BBD_pygmtsar.to_csv(
                        area_info['path'] / f'{area_name}_BBD_vs_PyGMTSAR_detailed_results_{method_suffix}{filter_suffix}.csv', 
                        index=False
                    )
                    
                    results['BBD_vs_PyGMTSAR'] = {
                        'statistics': stats_BBD_pygmtsar,
                        'matched_points': len(matched_BBD_pygmtsar),
                        'plot_file': str(plot_file_BBD_pygmtsar),
                        'comparison_type': 'BBD_vs_PyGMTSAR',
                        'method': 'best_fit_selection' if USE_BEST_FIT else 'multi_neighbor_averaging',
                        'filtering': 'percentile' if USE_PERCENTILE_FILTER else 'none'
                    }
                    
                    self.logger.info(f"Completed BBD vs PyGMTSAR ({method_str}{filter_str}): R²={stats_BBD_pygmtsar['r_squared']:.3f}, RMSE={stats_BBD_pygmtsar['rmse']:.2f}")
                else:
                    self.logger.warning(f"Too few matches for BBD vs PyGMTSAR comparison ({len(matched_BBD_pygmtsar)} points)")
            
            for psi_file in area_info['psi_files']:
                method_name = f"PSI_{psi_file.stem}"
                self.logger.info(f"Processing {method_name}")
                
                pygmtsar_data_psi, bounds = self.load_pygmtsar_data(psi_file, apply_sign_flip=False)
                matched_data = self.find_nearest_neighbors(snap_data, pygmtsar_data_psi, 
                                                        max_distance_m=MAX_DISTANCE_PSI, 
                                                        n_neighbors=N_NEIGHBORS_PSI)
                
                if len(matched_data) < 5:
                    self.logger.warning(f"Too few matches for {method_name}")
                    continue
                
                stats_dict, differences = self.calculate_statistics(matched_data)
                
                plot_file = self.create_enhanced_comparison_plots(
                    area_name, f"{psi_file.stem} vs Snap2StaMPS", matched_data, differences, stats_dict,
                    area_info['path']
                )
                
                detailed_results = matched_data.copy()
                detailed_results['Difference'] = differences
                detailed_results.to_csv(area_info['path'] / f'{area_name}_{psi_file.stem}_detailed_results_{method_suffix}{filter_suffix}.csv', 
                                    index=False)
                
                results[method_name] = {
                    'statistics': stats_dict,
                    'matched_points': len(matched_data),
                    'plot_file': str(plot_file),
                    'method': 'best_fit_selection' if USE_BEST_FIT else 'multi_neighbor_averaging',
                    'filtering': 'percentile' if USE_PERCENTILE_FILTER else 'none'
                }
                
                self.logger.info(f"Completed {method_name} ({method_str}{filter_str}): R²={stats_dict['r_squared']:.3f}, RMSE={stats_dict['rmse']:.2f}")
            
            for sbas_file in area_info['sbas_files']:
                method_name = f"SBAS_{sbas_file.stem}"
                self.logger.info(f"Processing {method_name} vs Snap2StaMPS")
                
                sbas_data, bounds = self.load_pygmtsar_data(sbas_file, apply_sign_flip=False)
                matched_data_sbas = self.find_nearest_neighbors(snap_data, sbas_data, 
                                                            max_distance_m=MAX_DISTANCE_SBAS, 
                                                            n_neighbors=N_NEIGHBORS_SBAS)
                
                if len(matched_data_sbas) < 5:
                    self.logger.warning(f"Too few matches for {method_name}")
                    continue
                
                stats_dict_sbas, differences_sbas = self.calculate_statistics(matched_data_sbas)
                
                plot_file_sbas = self.create_enhanced_comparison_plots(
                    area_name, f"{sbas_file.stem} vs Snap2StaMPS", matched_data_sbas, differences_sbas, stats_dict_sbas,
                    area_info['path']
                )
                
                detailed_results_sbas = matched_data_sbas.copy()
                detailed_results_sbas['Difference'] = differences_sbas
                detailed_results_sbas.to_csv(area_info['path'] / f'{area_name}_{sbas_file.stem}_vs_Snap2StaMPS_detailed_results_{method_suffix}{filter_suffix}.csv', 
                                            index=False)
                
                results[method_name] = {
                    'statistics': stats_dict_sbas,
                    'matched_points': len(matched_data_sbas),
                    'plot_file': str(plot_file_sbas),
                    'comparison_type': 'SBAS_vs_Snap2StaMPS',
                    'method': 'best_fit_selection' if USE_BEST_FIT else 'multi_neighbor_averaging',
                    'filtering': 'percentile' if USE_PERCENTILE_FILTER else 'none'
                }
                
                self.logger.info(f"Completed {method_name} vs Snap2StaMPS ({method_str}{filter_str}): R²={stats_dict_sbas['r_squared']:.3f}, RMSE={stats_dict_sbas['rmse']:.2f}")
            
            method_suffix = "bestfit" if USE_BEST_FIT else "averaging"
            filter_suffix = "_filtered" if USE_PERCENTILE_FILTER else ""
            with open(area_info['path'] / f'{area_name}_enhanced_summary_{method_suffix}{filter_suffix}.json', 'w') as f:
                json_results = {}
                for key, value in results.items():
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            json_results[key][k] = {str(kk): float(vv) if isinstance(vv, (np.float32, np.float64)) 
                                                else vv for kk, vv in v.items()}
                        elif isinstance(v, (np.float32, np.float64)):
                            json_results[key][k] = float(v)
                        else:
                            json_results[key][k] = v
                
                json.dump(json_results, f, indent=2)
            
            self.overall_results[area_name] = results
            
        except Exception as e:
            self.logger.error(f"Error processing {area_name}: {str(e)}")
        finally:
            self.logger.removeHandler(area_handler)
            area_handler.close()
        
        return results
    
    def create_overall_summary(self):
        self.logger.info("\n" + "="*60)
        self.logger.info("CREATING ENHANCED OVERALL SUMMARY")
        self.logger.info("="*60)
        
        if not self.overall_results:
            self.logger.warning("No successful processing results found. Creating empty summary.")
            return pd.DataFrame()
        
        all_stats = []
        for area_name, area_results in self.overall_results.items():
            for method_name, method_results in area_results.items():
                if 'statistics' in method_results:
                    stats = method_results['statistics'].copy()
                    stats['study_area'] = area_name
                    stats['method'] = method_name
                    stats['comparison_type'] = method_results.get('comparison_type', 'unknown')
                    stats['selection_method'] = method_results.get('method', 'unknown')
                    stats['filtering_method'] = method_results.get('filtering', 'unknown')
                    all_stats.append(stats)
        
        if not all_stats:
            self.logger.warning("No valid statistics found. Creating minimal summary.")
            return pd.DataFrame()
        
        summary_df = pd.DataFrame(all_stats)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        comparison_colors = {
            'BBD_vs_Snap2StaMPS': '#8E44AD',
            'BBD_vs_PyGMTSAR': '#E67E22',
            'SBAS_vs_Snap2StaMPS': '#27AE60',
            'PSI': '#3498DB',
            'unknown': '#95A5A6'
        }
        
        colors_list = []
        for _, row in summary_df.iterrows():
            comp_type = row.get('comparison_type', 'unknown')
            if 'BBD' in row['method']:
                if 'Snap' in row['method']:
                    colors_list.append(comparison_colors['BBD_vs_Snap2StaMPS'])
                else:
                    colors_list.append(comparison_colors['BBD_vs_PyGMTSAR'])
            elif 'SBAS' in row['method']:
                colors_list.append(comparison_colors['SBAS_vs_Snap2StaMPS'])
            elif 'PSI' in row['method']:
                colors_list.append(comparison_colors['PSI'])
            else:
                colors_list.append(comparison_colors['unknown'])
        
        axes[0,0].bar(range(len(summary_df)), summary_df['correlation'], color=colors_list)
        axes[0,0].set_xlabel('Comparisons')
        axes[0,0].set_ylabel('Correlation Coefficient')
        method_title = "Best-Fit Selection" if USE_BEST_FIT else "Multi-Neighbor Averaging"
        filter_title = " + Percentile Filtering" if USE_PERCENTILE_FILTER else ""
        axes[0,0].set_title(f'Correlation Across All Comparisons\n({method_title}{filter_title})')
        axes[0,0].set_xticks(range(len(summary_df)))
        axes[0,0].set_xticklabels([f"{row['study_area']}\n{row['method']}" for _, row in summary_df.iterrows()], 
                                 rotation=45, ha='right', fontsize=8)
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        axes[0,1].bar(range(len(summary_df)), summary_df['rmse'], color=colors_list)
        axes[0,1].set_xlabel('Comparisons')
        axes[0,1].set_ylabel('RMSE (mm/year)')
        axes[0,1].set_title('RMSE Across All Comparisons')
        axes[0,1].set_xticks(range(len(summary_df)))
        axes[0,1].set_xticklabels([f"{row['study_area']}\n{row['method']}" for _, row in summary_df.iterrows()], 
                                 rotation=45, ha='right', fontsize=8)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        axes[0,2].bar(range(len(summary_df)), summary_df['n_points'], color=colors_list)
        axes[0,2].set_xlabel('Comparisons')
        axes[0,2].set_ylabel('Number of Matched Points')
        axes[0,2].set_title('Sample Sizes')
        axes[0,2].set_xticks(range(len(summary_df)))
        axes[0,2].set_xticklabels([f"{row['study_area']}\n{row['method']}" for _, row in summary_df.iterrows()], 
                                 rotation=45, ha='right', fontsize=8)
        axes[0,2].grid(True, alpha=0.3, axis='y')
        
        axes[1,0].bar(range(len(summary_df)), summary_df['bias'], color=colors_list)
        axes[1,0].axhline(0, color='black', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Comparisons')
        axes[1,0].set_ylabel('Mean Bias (mm/year)')
        axes[1,0].set_title('Bias Across All Comparisons')
        axes[1,0].set_xticks(range(len(summary_df)))
        axes[1,0].set_xticklabels([f"{row['study_area']}\n{row['method']}" for _, row in summary_df.iterrows()], 
                                 rotation=45, ha='right', fontsize=8)
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        axes[1,1].bar(range(len(summary_df)), summary_df['r_squared'], color=colors_list)
        axes[1,1].set_xlabel('Comparisons')
        axes[1,1].set_ylabel('R-squared')
        axes[1,1].set_title('R² Across All Comparisons')
        axes[1,1].set_xticks(range(len(summary_df)))
        axes[1,1].set_xticklabels([f"{row['study_area']}\n{row['method']}" for _, row in summary_df.iterrows()], 
                                 rotation=45, ha='right', fontsize=8)
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        axes[1,2].axis('off')
        
        legend_elements = []
        for comp_type, color in comparison_colors.items():
            if comp_type != 'unknown':
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=comp_type.replace('_', ' vs ')))
        
        axes[1,2].legend(handles=legend_elements, loc='center', fontsize=12, title='Comparison Types', title_fontsize=14)
        
        axes[2,0].axis('off')
        axes[2,1].axis('off') 
        axes[2,2].axis('off')
        
        if len(summary_df) > 0:
            BBD_comparisons = summary_df[summary_df['method'].str.contains('BBD', na=False)]
            psi_comparisons = summary_df[summary_df['method'].str.contains('PSI', na=False)]
            sbas_comparisons = summary_df[summary_df['method'].str.contains('SBAS', na=False)]
            
            summary_text = f"""
            ENHANCED MASTER THESIS SUMMARY
            Method: {method_title}{filter_title}
            
            OVERALL STATISTICS:
            Total Study Areas: {len(self.overall_results)}
            Total Comparisons: {len(summary_df)}
            
            Performance Metrics (All Comparisons):
            • Mean Correlation: {summary_df['correlation'].mean():.3f}
            • Mean R²: {summary_df['r_squared'].mean():.3f}
            • Mean RMSE: {summary_df['rmse'].mean():.2f} mm/year
            • Mean Bias: {summary_df['bias'].mean():.2f} mm/year
            
            COMPARISON TYPE BREAKDOWN:
            
            BBD Comparisons ({len(BBD_comparisons)} total):
            """
            
            if len(BBD_comparisons) > 0:
                summary_text += f"""• Mean R²: {BBD_comparisons['r_squared'].mean():.3f}
            • Mean RMSE: {BBD_comparisons['rmse'].mean():.2f} mm/year
            • Mean Bias: {BBD_comparisons['bias'].mean():.2f} mm/year
            """
            else:
                summary_text += "• No BBD comparisons found\n"
            
            summary_text += f"""
            PSI Comparisons ({len(psi_comparisons)} total):
            """
            
            if len(psi_comparisons) > 0:
                summary_text += f"""• Mean R²: {psi_comparisons['r_squared'].mean():.3f}
            • Mean RMSE: {psi_comparisons['rmse'].mean():.2f} mm/year
            • Mean Bias: {psi_comparisons['bias'].mean():.2f} mm/year
            """
            
            summary_text += f"""
            SBAS Comparisons ({len(sbas_comparisons)} total):
            """
            
            if len(sbas_comparisons) > 0:
                summary_text += f"""• Mean R²: {sbas_comparisons['r_squared'].mean():.3f}
            • Mean RMSE: {sbas_comparisons['rmse'].mean():.2f} mm/year
            • Mean Bias: {sbas_comparisons['bias'].mean():.2f} mm/year
            """
            
            summary_text += f"""
            
            QUALITY ASSESSMENT:
            • Comparisons with R² > 0.7: {(summary_df['r_squared'] > 0.7).sum()}/{len(summary_df)}
            • Comparisons with RMSE < 5: {(summary_df['rmse'] < 5).sum()}/{len(summary_df)}
            • Comparisons with |bias| < 2: {(summary_df['bias'].abs() < 2).sum()}/{len(summary_df)}
            
            BEST PERFORMING:
            • Highest R²: {summary_df.loc[summary_df['r_squared'].idxmax(), 'study_area']} 
              ({summary_df.loc[summary_df['r_squared'].idxmax(), 'method']}: {summary_df['r_squared'].max():.3f})
            • Lowest RMSE: {summary_df.loc[summary_df['rmse'].idxmin(), 'study_area']} 
              ({summary_df.loc[summary_df['rmse'].idxmin(), 'method']}: {summary_df['rmse'].min():.2f} mm/year)
            
            BBD DATA INSIGHTS:
            • Time period: {BBD_START_DATE} to {BBD_END_DATE}
            • Distance threshold: {MAX_DISTANCE_BBD}m
            • Neighbor candidates: {N_NEIGHBORS_BBD}
            """
        else:
            summary_text = "No successful processing results to summarize."
        
        fig.text(0.1, 0.25, summary_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.4)
        
        method_suffix = "bestfit" if USE_BEST_FIT else "averaging"
        filter_suffix = "_filtered" if USE_PERCENTILE_FILTER else ""
        plt.savefig(f'enhanced_master_thesis_overall_summary_{method_suffix}{filter_suffix}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'enhanced_master_thesis_overall_summary_{method_suffix}{filter_suffix}.pdf', bbox_inches='tight')
        
        if len(summary_df) > 0:
            summary_df.to_csv(f'enhanced_master_thesis_summary_statistics_{method_suffix}{filter_suffix}.csv', index=False)
            self.logger.info("Enhanced overall summary created and saved")
        else:
            self.logger.warning("No data to save in summary CSV")
        
        return summary_df
    
    def run_complete_analysis(self):
        method_str = "Best-Fit Selection" if USE_BEST_FIT else "Multi-Neighbor Averaging"
        filter_str = " with Percentile Filtering" if USE_PERCENTILE_FILTER else ""
        self.logger.info(f"Starting Enhanced Master Thesis InSAR Comparison Analysis ({method_str}{filter_str})")
        self.logger.info(f"Enhanced analysis started at: {datetime.now()}")
        self.logger.info(f"BBD data time period: {BBD_START_DATE} to {BBD_END_DATE}")
        
        if USE_PERCENTILE_FILTER:
            self.logger.info(f"Percentile filtering enabled: {PERCENTILE_RANGE[0]}th-{PERCENTILE_RANGE[1]}th percentile range")
        
        study_areas = self.discover_study_areas()
        
        if not study_areas:
            self.logger.error("No study areas found!")
            return
        
        for area_info in study_areas:
            self.process_single_area(area_info)
        
        summary_df = self.create_overall_summary()
        
        self.logger.info(f"Enhanced analysis completed at: {datetime.now()}")
        self.logger.info("All results saved in respective subdirectories")
        
        return summary_df

def main():
    comparison = InSARComparison('comparing')
    summary = comparison.run_complete_analysis()
    return comparison, summary

if __name__ == "__main__":
    comparison, summary = main()