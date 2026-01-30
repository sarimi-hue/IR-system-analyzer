import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import sys
import os
from scipy.stats import chi2, probplot, f #f for Hotelling's T2 test
from datetime import datetime

# for potentially complex data processing, important?
sys.setrecursionlimit(2000)

def save_2d_ellipsoid_report(df, features, md_threshold, median_vec, covariance_matrix, filename, output_folder):
#3 elliptic figures in one page start here ^^
    
    pairs = [(features[0], features[1]), (features[0], features[2]), (features[1], features[2])]
    fig, axes = plt.subplots(3, 1, figsize=(8, 18))
    
    for i, (f1, f2) in enumerate(pairs):
        ax = axes[i]
        sns.scatterplot(data=df, x=f1, y=f2, hue='is_abnormal', palette={False: 'blue', True: 'red'}, ax=ax, alpha=0.5, s=30)
        
        # Calculate ellipse parameters from covariance
        idx1, idx2 = features.index(f1), features.index(f2)
        
        # Ensure we are using numpy arrays for indexing
        if isinstance(covariance_matrix, pd.DataFrame):
            sub_cov = covariance_matrix.iloc[[idx1, idx2], [idx1, idx2]].values
        else:
            sub_cov = covariance_matrix[np.ix_([idx1, idx2], [idx1, idx2])]
            
        sub_pos = median_vec.iloc[[idx1, idx2]].values if isinstance(median_vec, pd.Series) else median_vec[[idx1, idx2]]
        
        vals, vecs = np.linalg.eigh(sub_cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
        # Width and height are based on the MD threshold scaled by eigenvalues
        width, height = 2 * md_threshold * np.sqrt(np.maximum(vals, 0))
        
        ellipse = patches.Ellipse(xy=sub_pos, width=width, height=height, angle=theta, 
                                  edgecolor='green', fc='none', lw=2, linestyle='--', label='MD Boundary')
        ax.add_patch(ellipse)
        ax.set_title(f'2D Slice: {f1} vs {f2}', fontweight='bold')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"2D_ellipsoid_{filename}.png"), dpi=150)
    plt.close()

def save_3d_ellipsoid_report(df, features, md_threshold, median_vec, covariance_matrix, filename, output_folder):
#3d md ellips is here^^
    
    if len(features) < 3: return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = df['is_abnormal'].map({False: 'blue', True: 'red'})
    ax.scatter(df[features[0]], df[features[1]], df[features[2]], c=colors, alpha=0.4, s=15)
    
    # Calculate 3D Ellipsoid mesh
    cov_vals = covariance_matrix.values if isinstance(covariance_matrix, pd.DataFrame) else covariance_matrix
    vals, vecs = np.linalg.eigh(cov_vals)
    radii = md_threshold * np.sqrt(np.maximum(vals, 0))
    
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    sphere_pts = np.stack([(np.cos(u)*np.sin(v)).flatten(), (np.sin(u)*np.sin(v)).flatten(), np.cos(v).flatten()])
    ellipsoid_pts = (vecs @ (radii[:, None] * sphere_pts))
    
    center = median_vec.values if isinstance(median_vec, pd.Series) else median_vec
    ax.plot_wireframe(ellipsoid_pts[0,:].reshape(u.shape)+center[0], 
                      ellipsoid_pts[1,:].reshape(u.shape)+center[1], 
                      ellipsoid_pts[2,:].reshape(u.shape)+center[2], color='green', alpha=0.1)
    
    ax.set_title("Full 3D Mahalanobis Ellipsoid", fontweight='bold')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    
    plt.savefig(os.path.join(output_folder, f"3D_ellipsoid_{filename}.png"), dpi=150)
    plt.close()
    
class MTS:
#DO NOT FORGET! use median, first screening using chi2 then final threhold using quantile (empirical quantile)
    
    def __init__(self, df, features, status_col='Actual_Status', normal_status='OK', 
                 ir_thresholds=None):
        self.df = df.copy()
        self.features = features
        self.status_col = status_col
        self.normal_status = normal_status
        self.ir_thresholds = ir_thresholds if ir_thresholds is not None else {}
        self.md_threshold = None 
        
        self.raw_median = self.df[self.features].median()
        self.raw_std = self.df[self.features].std()
        self.raw_mean = self.df[self.features].mean()
        
        self.median_3sigma = None
        self.std_3sigma = None
        self.condition_number = np.nan 

        if self.status_col not in self.df.columns:
            raise KeyError(f"The status column '{self.status_col}' was not found in the DataFrame. Ensure the filtering step was run.")
        
        initial_normal_data = self.df[self.df[self.status_col] == self.normal_status]
        
        if initial_normal_data.empty:
            raise ValueError(f"No data points meet the criteria for the 'Normal' group ('{self.normal_status}') to build the Mahalanobis Space.")
        
        # Confirmation: 3sigma stats are derived from the data *after* filtering
        self.median_3sigma = initial_normal_data[self.features].median()
        self.std_3sigma = initial_normal_data[self.features].std()
        self.mean_3sigma = initial_normal_data[self.features].mean()
        self.normal_data = initial_normal_data[self.features] 
        
        # FIX: Store both the Series for iteration and the NumPy array for calculation
        self.normal_median_series = self.normal_data.median()
        self.normal_median_values = self.normal_median_series.values # The NumPy array for MD calculation
        # END FIX

        try:
            if len(self.features) == 1:
                cov_matrix = np.atleast_2d(self.normal_data.cov().values)
                self.inv_cov = np.linalg.inv(cov_matrix)
            else:
                cov_matrix = self.normal_data.cov().values
                self.inv_cov = np.linalg.inv(cov_matrix)
            
            # NEW STATISTICAL OUTPUT: Condition Number
            self.condition_number = np.linalg.cond(cov_matrix)

        except np.linalg.LinAlgError:
            print("Warning: covariance matrix is singular. Using pseudo-inverse.")
            if len(self.features) == 1:
                cov_matrix = np.atleast_2d(self.normal_data.cov().values)
                self.inv_cov = np.linalg.pinv(cov_matrix)
            else:
                cov_matrix = self.normal_data.cov().values
                self.inv_cov = np.linalg.pinv(cov_matrix)
            
            self.condition_number = np.linalg.cond(cov_matrix)
                
    def calculate_md_all(self):
    #MD after filtering
        if 'Mahalanobis_Distance' in self.df.columns: return
        
        if self.normal_data.empty or np.isnan(self.normal_median_values).any() or np.isnan(self.inv_cov).any() or not self.inv_cov.any():
            self.df['Mahalanobis_Distance'] = np.nan
            return

        data_to_evaluate = self.df[self.features].values
        # Use the NumPy array for the calculation
        normal_median_expanded = self.normal_median_values.reshape(1, -1) 
        data_minus_median = data_to_evaluate - normal_median_expanded
        
        self.df['Mahalanobis_Distance'] = np.sqrt(np.einsum('ij,jk,ik->i', data_minus_median, self.inv_cov, data_minus_median))

    def detect_abnormal(self, md_quantile=0.99):
    #empirical quantile threshold
        self.calculate_md_all()
        result_df = self.df.copy()
        
        md_ok_data = result_df[result_df[self.status_col] == self.normal_status]['Mahalanobis_Distance'].dropna()

        if md_ok_data.empty:
            result_df['is_abnormal'] = (result_df[self.status_col] != self.normal_status) 
            result_df['abnormal_type'] = 'N/A'
            return result_df
        
        self.md_threshold = md_ok_data.quantile(md_quantile)
        print(f"\nPrimary MD Threshold (Empirical {md_quantile*100:.2f}th Quantile): {self.md_threshold:.4f}")
        
        md_is_abnormal = result_df['Mahalanobis_Distance'] > self.md_threshold
        initial_ng_mask = (result_df[self.status_col] != self.normal_status)
        
        # Define the initial MTS NG flag: Initial Screening NG OR MD Outlier
        result_df['is_abnormal'] = initial_ng_mask | md_is_abnormal
        result_df['abnormal_type'] = 'OK' 
        
        # Diagnosis/Type Assignment Logic (Using the clean 3sigma stats)
        is_outside_3sigma = pd.Series(False, index=result_df.index)
        for feature in self.features:
            if feature in result_df.columns and self.std_3sigma[feature] > 0:
                upper_bound = self.median_3sigma[feature] + 3 * self.std_3sigma[feature]
                lower_bound = self.median_3sigma[feature] - 3 * self.std_3sigma[feature]
                is_outside_3sigma = is_outside_3sigma | ((result_df[feature] > upper_bound) | (result_df[feature] < lower_bound))

        # Store the raw count of outside 3 sigma for the summary table 
        result_df['is_outside_3sigma_raw'] = is_outside_3sigma

        is_below_ir_threshold = pd.Series(False, index=result_df.index)
        for feature, threshold_val in self.ir_thresholds.items():
            if feature in self.features:
                is_below_ir_threshold = is_below_ir_threshold | (result_df[feature] < threshold_val)

        # Important for correct counting
        result_df.loc[is_below_ir_threshold, 'abnormal_type'] = 'initial_ir_threshold_ng'
        
        # Check for 3-sigma *excluding* points already flagged as IR NG
        current_ok_or_3sigma = (result_df['abnormal_type'] == 'OK') | (result_df['abnormal_type'] == '3_Sigma_Reject')
        result_df.loc[is_outside_3sigma & current_ok_or_3sigma, 'abnormal_type'] = '3_Sigma_Reject'
        
        # Check for MD outlier *excluding* points already flagged as IR/3-sigma NG
        current_ok = (result_df['abnormal_type'] == 'OK')
        result_df.loc[md_is_abnormal & current_ok, 'abnormal_type'] = 'MD_Reject'
        
        # Final result for any remaining initial screened NG (should be few/none if logic is complete)
        result_df.loc[initial_ng_mask & (result_df['abnormal_type'] == 'OK'), 'abnormal_type'] = 'initial_screened_ng'
        
        print(f"Found {result_df['is_abnormal'].sum()} total abnormal points based on all criteria before conservative case removal.")
        return result_df
    
    def export_points_only_table(self, abnormal_df):
        if abnormal_df.empty or self.median_3sigma is None:
            return pd.DataFrame()
            
        df = abnormal_df.copy()
        # Initialize list with the new 'Sample No.' column
        output_cols = ['original_row_number']
        
        for feature in self.features:
            status_col_name = f'Status_{feature}'
            
            df[status_col_name] = df.apply(
                lambda row: 'NG' if 'NG' in self._get_feature_diagnosis(row, feature, self.median_3sigma, self.std_3sigma) else 'OK',
                axis=1
            )
            
            output_cols.extend([feature, status_col_name])
        
        # Select columns and rename 'original_row_number' to 'Sample no.'
        final_table = df[output_cols].copy()
        final_table.rename(columns={'original_row_number': 'Sample no.'}, inplace=True)
        
        return final_table
    
    def calculate_type1_type2_errors(self, result_df):
    #calculate typeI/II
        
        if result_df.empty:
             return {
                'N_Total': 0, 'N_Actual_Normal': 0, 'N_Actual_Abnormal': 0,
                'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'Type_I_Error_Rate': 0.0, 'Type_II_Error_Rate': 0.0
            }

        actual_abnormal = (result_df[self.status_col] != self.normal_status)
        actual_normal = (result_df[self.status_col] == self.normal_status)
        
        # Note: The 'is_abnormal' mask used here reflects the initial rule (Screening NG OR MD NG).
        predicted_abnormal = result_df['is_abnormal'] 
        
        TP = (actual_abnormal & predicted_abnormal).sum()
        TN = (actual_normal & ~predicted_abnormal).sum()
        FP = (actual_normal & predicted_abnormal).sum()
        FN = (actual_abnormal & ~predicted_abnormal).sum()

        N_actual_normal = actual_normal.sum()
        N_actual_abnormal = actual_abnormal.sum()
        
        type_i_error_rate = FP / N_actual_normal if N_actual_normal > 0 else 0.0
        type_ii_error_rate = FN / N_actual_abnormal if N_actual_abnormal > 0 else 0.0
        
        result_df['classification_type'] = 'TN (Correct Normal)'
        result_df.loc[actual_abnormal & predicted_abnormal, 'classification_type'] = 'TP (Correct Abnormal)'
        result_df.loc[actual_normal & predicted_abnormal, 'classification_type'] = 'FP (Type I Error)'
        result_df.loc[actual_abnormal & ~predicted_abnormal, 'classification_type'] = 'FN (Type II Error)'

        return {
            'N_Total': len(result_df), 'N_Actual_Normal': N_actual_normal, 'N_Actual_Abnormal': N_actual_abnormal,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'Type_I_Error_Rate': type_i_error_rate,
            'Type_II_Error_Rate': type_ii_error_rate
        }

    def _get_feature_diagnosis(self, row, feature, median_df, std_df):
            value = row[feature]
            median = median_df[feature]
            std = std_df[feature]

            # 1. IR Threshold Check
            if feature in self.ir_thresholds and value < self.ir_thresholds[feature]:
                return 'NG', f'{feature} is NG (below threshold: {self.ir_thresholds[feature]:.2e})'
            
            # 2. Sigma Check
            if std > 0:
                upper_3sigma = median + 3 * std
                lower_3sigma = median - 3 * std
                if value > upper_3sigma:
                    return 'NG', f'{feature} is NG (outside +3σ)'
                elif value < lower_3sigma:
                    return 'NG', f'{feature} is NG (outside -3σ)'
            elif value != median:
                return 'NG', f'{feature} is NG (deviates significantly with zero std)'
                
            return 'OK', f'{feature} is OK'
        
    def diagnose_abnormal_features(self, abnormal_df):
            if abnormal_df.empty or self.median_3sigma is None or self.std_3sigma is None:
                return pd.DataFrame()
                
            median_df = self.median_3sigma
            std_df = self.std_3sigma
            df = abnormal_df.copy()
            
            # We will build a list of the new column names to maintain order
            ir_status_cols = []
            diagnosis_cols = []

            for feature in self.features:
                # We use a unique temporary name internally to prevent Pandas 
                # from merging columns with the same name during calculation
                temp_status_name = f'temp_stat_{feature}'
                diag_col_name = f'Diagnosis_{feature}'
                
                results = df.apply(
                    lambda row: self._get_feature_diagnosis(row, feature, median_df, std_df),
                    axis=1
                )
                
                df[temp_status_name] = [r[0] for r in results]
                df[diag_col_name] = [r[1] for r in results]
                
                # Record the order: Feature value, then its Status
                ir_status_cols.extend([feature, temp_status_name])
                diagnosis_cols.append(diag_col_name)
            
            # Define Metadata columns
            base_info = ['original_row_number', 'Mahalanobis_Distance', 'abnormal_type']
            
            # Final selection and Reordering
            final_df = df[base_info + ir_status_cols + diagnosis_cols].copy()
            
            # Rename the temporary status columns to exactly "Status"
            # This works because we are renaming by index/position
            rename_dict = {f'temp_stat_{f}': 'Status' for f in self.features}
            final_df.rename(columns=rename_dict, inplace=True)
            
            return final_df

    # PLOT_RESULTS FUNCTION (6x3 layout with log plots)
    def plot_results(self, result_df, secondary_threshold_percentile=0.99, save_filename=None):
        if self.md_threshold is None:
            print("Mahalanobis Distance threshold not set. Cannot plot with thresholds.")
            return

        # 1. PREPARE LOG DATA AND STATS
        log_features = [f'ln({f})' for f in self.features]

        # Create temporary log dataframes/series for plotting
        log_df = result_df.copy()
        log_normal_data = self.normal_data.copy()

        for feature in self.features:
            log_feature = f'ln({feature})'
            # Log transform all data (handling non-positive if they exist, though unlikely for IR)
            log_df[log_feature] = np.log(log_df[feature].replace(0, np.nan).dropna())
            log_normal_data[log_feature] = np.log(self.normal_data[feature].replace(0, np.nan).dropna())

        # Calculate Log Stats and Thresholds
        log_median_ok = log_normal_data[log_features].median()
        log_std_ok = log_normal_data[log_features].std()
        log_raw_median = log_df[log_features].median()
        log_raw_std = log_df[log_features].std()
        
        # Transform IR thresholds to log scale
        log_ir_thresholds = {
            f'ln({f})': np.log(v) for f, v in self.ir_thresholds.items() 
            if v > 0 # Log is only valid for positive thresholds
        }
        
        # 2. SETUP FIGURE
        dof = len(self.features)
        secondary_md_threshold = np.sqrt(chi2.ppf(secondary_threshold_percentile, dof))
        
        # 6 Rows x 3 Columns (R1-R2: Linear, R3-R4: Log, R5: MD Scatter, R6: Error Viz)
        fig, axes = plt.subplots(6, 3, figsize=(18, 24)) 
        
        # Data Filtering (Ensure these are correctly defined)
        # Note: result_df already contains the MD and flags, so we use it for filtering
        normal_points = result_df[result_df['is_abnormal'] == False]
        abnormal_points = result_df[result_df['is_abnormal'] == True]
        
        multivariate_anomalies = abnormal_points[abnormal_points['abnormal_type'] == 'MD_Reject']
        outside_3sigma = abnormal_points[abnormal_points['abnormal_type'] == '3_Sigma_Reject']
        ir_threshold_ng = abnormal_points[abnormal_points['abnormal_type'] == 'initial_ir_threshold_ng']
        initial_screened_ng = abnormal_points[abnormal_points['abnormal_type'] == 'initial_screened_ng']
        
        tn_points = result_df[result_df['classification_type'] == 'TN (Correct Normal)']
        tp_points = result_df[result_df['classification_type'] == 'TP (Correct Abnormal)']
        fp_points = result_df[result_df['classification_type'] == 'FP (Type I Error)']
        fn_points = result_df[result_df['classification_type'] == 'FN (Type II Error)']

        y_max_overall = result_df['Mahalanobis_Distance'].max() * 1.05
        y_max_normal = self.md_threshold * 1.5

        # Helper for MD Scatter Plot (Row 5 & 6)
        def draw_md_scatter(ax, title, is_error_plot=False, set_ylim=None, threshold_color='green'):
            ax.scatter(normal_points.index, normal_points['Mahalanobis_Distance'], color='blue', label='Normal Samples', alpha=0.6, s=15)
            ax.scatter(multivariate_anomalies.index, multivariate_anomalies['Mahalanobis_Distance'], color='yellow', label='MD Reject', edgecolors='black', s=15)
            ax.scatter(outside_3sigma.index, outside_3sigma['Mahalanobis_Distance'], color='red', label='3 Sigma Reject', edgecolors='black', s=15)
            ax.scatter(ir_threshold_ng.index, ir_threshold_ng['Mahalanobis_Distance'], color='purple', marker='x', s=15, label='IR Threshold NG', linewidth=1.0)
            ax.scatter(initial_screened_ng.index, initial_screened_ng['Mahalanobis_Distance'], color='gray', marker='D', s=15, label='Initial Screened NG', linewidth=1.0)
            ax.axhline(y=self.md_threshold, color=threshold_color, linestyle='--', label=f'Empirical MD Threshold ({self.md_threshold:.2f})')
            if not is_error_plot:
                ax.axhline(y=secondary_md_threshold, color='orange', linestyle=':', label=f'Chi-Squared MD Threshold ({secondary_md_threshold:.2f})')
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Mahalanobis Distance')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(fontsize='x-small', loc='upper right' if set_ylim else 'upper left')
            if set_ylim: ax.set_ylim(*set_ylim)

        # Helper function for Distribution Plots (Rows R_start and R_start+1)
        def plot_distributions(feature_set, data_df_all, data_df_ok, median_ok_stats, std_ok_stats, raw_median_stats, raw_std_stats, ir_thresholds_stats, R_start, scale_label):
            
            for i, feature in enumerate(feature_set):
                
                # Use the original feature name for the axis title if it's a log feature
                original_feature_name = feature.replace('ln(', '').replace(')', '')
                # X-axis label includes scale information
                feature_label = rf'{original_feature_name} ($\\Omega$)' if scale_label == 'Linear Scale' else f'$\\ln$({original_feature_name})'
                
                # Check for data existence
                if data_df_ok.empty or median_ok_stats is None or std_ok_stats is None:
                    axes[R_start, i].set_title(f'No Normal Data for {feature_label}', fontsize=9)
                    axes[R_start + 1, i].set_title(f'No Normal Data for {feature_label}', fontsize=9)
                    axes[R_start, i].set_ylabel('Number of Samples') 
                    axes[R_start + 1, i].set_ylabel('Number of Samples') 
                    continue

                # CALCULATE TIGHT LIMITS
                median_ok = median_ok_stats[feature]
                std_ok = std_ok_stats[feature]
                ir_threshold_val = ir_thresholds_stats.get(feature, None)
                
                # 1. Initial limits based on 4-sigma of the *clean* data (the MTS reference)
                if std_ok <= 1e-9: # Handle near-zero variance
                    X_RANGE = 1e-7 if scale_label == 'Linear Scale' else 0.5 # Small range for linear, 0.5 for log
                    x_min_limit = median_ok - X_RANGE
                    x_max_limit = median_ok + X_RANGE
                    zero_std_flag = True
                else:
                    x_min_limit = median_ok - 4 * std_ok 
                    x_max_limit = median_ok + 4 * std_ok
                    zero_std_flag = False
                
                # 2. Incorporate 4-sigma of the *raw* data to ensure it's visible 
                raw_median = raw_median_stats[feature]
                raw_std = raw_std_stats[feature]
                raw_4sigma_min = raw_median - 4 * raw_std
                raw_4sigma_max = raw_median + 4 * raw_std
                
                # Use the min and max of the combined 4-sigma ranges to set the overall boundary
                x_min_limit = min(x_min_limit, raw_4sigma_min)
                x_max_limit = max(x_max_limit, raw_4sigma_max)
                    
                # 3. Final Check: Ensure the IR specification is also included if it's even further out
                if ir_threshold_val is not None:
                     if ir_threshold_val < x_min_limit:
                         # Adjust the minimum limit to comfortably include the IR threshold
                         x_min_limit = ir_threshold_val * 0.95 if ir_threshold_val > 0 and scale_label == 'Linear Scale' else ir_threshold_val - 0.1 # Adjust slightly differently for log
                     elif ir_threshold_val > x_max_limit:
                          x_max_limit = ir_threshold_val * 1.05 if ir_threshold_val > 0 and scale_label == 'Linear Scale' else ir_threshold_val + 0.1

                bins_setting = 20 if original_feature_name == 'IR2' else None 
                
                # Outliers relative to the *final* determined plot limits
                raw_data_for_feature = data_df_all[feature].dropna()
                left_outliers = raw_data_for_feature[raw_data_for_feature < x_min_limit]
                right_outliers = raw_data_for_feature[raw_data_for_feature > x_max_limit]       

                # Row R_start + 1: Cleaned "OK" Data Distribution (Plot first to get Y_max)
                ax_row2 = axes[R_start + 1, i]
                hist_kwargs_r2 = {'data': data_df_ok, 'x': feature, 'ax': ax_row2, 'color': 'blue', 'label': f'Natural (OK) Data Distribution ({scale_label})', 'kde': True, 'stat': 'count'}
                if bins_setting is not None: hist_kwargs_r2['bins'] = bins_setting
                
                if zero_std_flag:
                    # Manually plot for zero variance data with clearer annotation
                    ax_row2.axvline(median_ok, color='blue', linestyle='-', linewidth=10, alpha=0.3, label='Zero Variance Data')
                    ax_row2.text(median_ok, len(data_df_ok) * 0.5, f'Data Constant\nValue: {median_ok:.2e}', 
                                 horizontalalignment='center', color='darkblue', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                    ax_row2.set_ylim(0, len(data_df_ok) * 1.1)
                    ax_row2.set_xlim(x_min_limit, x_max_limit) # Apply limits here too
                else:
                    sns.histplot(**hist_kwargs_r2)
                    
                if not zero_std_flag: # Only add lines if there's actual variance to show them relative to
                    # Reference Lines for Row 2 (Only 3-Sigma and IR Spec)
                    ax_row2.axvline(median_ok, color='darkblue', linestyle='--', label='Median (OK)')
                    ax_row2.axvline(median_ok + 3 * std_ok, color='green', linestyle='--', label='+3 Sigma (OK)')
                    ax_row2.axvline(median_ok - 3 * std_ok, color='green', linestyle='--', label='-3 Sigma (OK)')
                
                # IR Spec lines 
                if ir_threshold_val is not None: 
                    ax_row2.axvline(ir_threshold_val, color='purple', linestyle='-.', label=f'IR Threshold') # Remove value to avoid clutter
                
                ax_row2.set_title(f'Cleaned "OK" Distribution of {original_feature_name} ({scale_label})', fontsize=9)
                ax_row2.set_ylabel('Number of Samples') 
                ax_row2.set_xlabel(feature_label) 
                ax_row2.legend(fontsize='small') 
                if not zero_std_flag: ax_row2.set_xlim(x_min_limit, x_max_limit)

                # Row R_start: Raw Data Distribution (Plot second)
                ax_row1 = axes[R_start, i]
                
                if zero_std_flag:
                    # Manually plot for zero variance data with clearer annotation
                    ax_row1.axvline(median_ok, color='blue', linestyle='-', linewidth=10, alpha=0.3, label='Zero Variance Data')
                    ax_row1.text(median_ok, len(data_df_all) * 0.5, f'Data Constant\nValue: {median_ok:.2e}', 
                                 horizontalalignment='center', color='darkblue', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                    ax_row1.set_ylim(0, len(data_df_all) * 1.1)
                else:
                    # Use fill=False to contrast Raw (outline) and Cleaned (filled)
                    hist_kwargs_r1 = {
                        'data': data_df_all, 'x': feature, 'ax': ax_row1, 'color': 'black', 
                        'label': f'Raw Data Distribution ({scale_label})', 'kde': True, 'stat': 'count', 
                        'line_kws': {'linestyle': '-', 'color': 'black'}, 'fill': False 
                    }
                    if bins_setting is not None: hist_kwargs_r1['bins'] = bins_setting
                    
                    sns.histplot(**hist_kwargs_r1)
                    
                    # Plot Raw reference lines
                    median_val, std_val = raw_median_stats[feature], raw_std_stats[feature]
                    ax_row1.axvline(median_val, color='darkgray', linestyle=':', label='Median (Raw)')
                    
                    # Include 3 sigma lines if they fall within the 4-sigma OK zoom window
                    if (median_val + 3 * std_val) > x_min_limit and (median_val - 3 * std_val) < x_max_limit:
                         ax_row1.axvline(median_val + 3 * std_val, color='red', linestyle='--', label='+3 Sigma (Raw)')
                         ax_row1.axvline(median_val - 3 * std_val, color='red', linestyle='--', label='-3 Sigma (Raw)')
                    
                # IR Spec lines (show)
                if ir_threshold_val is not None: ax_row1.axvline(ir_threshold_val, color='purple', linestyle='-.', label=f'IR Threshold')
                
                ax_row1.set_title(f'Raw Data Distribution of {original_feature_name} ({scale_label})', fontsize=9)
                ax_row1.set_ylabel('Number of Samples') 
                ax_row1.set_xlabel(feature_label) 
                ax_row1.legend(fontsize='small') 
                
                # Apply X-Axis Limit only (Keeping flexible Y-axis)
                ax_row1.set_xlim(x_min_limit, x_max_limit) 
                
                # Add arrows for outliers (applied to all features now)
                if not zero_std_flag:
                    # Get the dynamically set Y-max for annotation positioning
                    y_max_for_anno = ax_row1.get_ylim()[1] 
                    
                    if not left_outliers.empty: 
                       ax_row1.annotate(f'<{len(left_outliers)} pts', 
                                        xy=(x_min_limit, y_max_for_anno * 0.9), 
                                        xytext=(x_min_limit + (x_max_limit-x_min_limit)*0.05, y_max_for_anno * 0.8),
                                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                        fontsize=8, ha='left', color='red')
                    if not right_outliers.empty: 
                        ax_row1.annotate(f'{len(right_outliers)} pts>', 
                                        xy=(x_max_limit, y_max_for_anno * 0.9), 
                                        xytext=(x_max_limit - (x_max_limit-x_min_limit)*0.05, y_max_for_anno * 0.8),
                                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                        fontsize=8, ha='right', color='red')
        
        # 3. PLOT LINEAR DISTRIBUTIONS (Rows 1 and 2)
        plot_distributions(
            self.features, 
            result_df, 
            self.normal_data, 
            self.median_3sigma, 
            self.std_3sigma, 
            self.raw_median, 
            self.raw_std, 
            self.ir_thresholds, 
            R_start=0, 
            scale_label='Linear Scale'
        )
        
        # 4. PLOT LOG DISTRIBUTIONS (Rows 3 and 4)
        plot_distributions(
            log_features, 
            log_df, # Log-transformed data for raw distribution
            log_normal_data, # Log-transformed data for cleaned distribution
            log_median_ok, 
            log_std_ok, 
            log_raw_median, 
            log_raw_std, 
            log_ir_thresholds, # Log-transformed thresholds
            R_start=2, # Starting row index is 2 (3rd row)
            scale_label='Log Scale'
        )


        # 5. Row 5 (axes[4, :]): Mahalanobis Distance Plots (Empirical Quantile Analysis)
        
        # R5C1: Overall View
        draw_md_scatter(axes[4, 0], 'MD Empirical Quantile Analysis: 1. Overall View', set_ylim=(0, y_max_overall))

        # R5C2: Zoom Near Normal Zone
        draw_md_scatter(axes[4, 1], f'MD Empirical Quantile Analysis: 2. Zoom Near Normal (Y-Limit: 0 to {y_max_normal:.2f})', set_ylim=(0, y_max_normal))

        # R5C3: Zoom Around Thresholds 
        min_t = min(self.md_threshold, secondary_md_threshold)
        max_t = max(self.md_threshold, secondary_md_threshold)
        ax3_ylim_min = max(0, min_t * 0.9)
        ax3_ylim_max = max_t * 1.1 
        draw_md_scatter(axes[4, 2], 'MD Empirical Quantile Analysis: 3. Zoom Around Thresholds', set_ylim=(ax3_ylim_min, ax3_ylim_max))

        # 6. Row 6 (axes[5, :]): Type I/II Error Visualization
        def draw_error_scatter(ax, title, set_ylim=None):
            # Highlight points by classification type
            ax.scatter(tn_points.index, tn_points['Mahalanobis_Distance'], color='blue', label='TN (Correct Normal)', alpha=0.6, s=15)
            ax.scatter(tp_points.index, tp_points['Mahalanobis_Distance'], color='red', label='TP (Correct Abnormal)', alpha=0.9, marker='X', edgecolors='black', s=70)
            ax.scatter(fp_points.index, fp_points['Mahalanobis_Distance'], color='green', label='FP (Type I Error)', s=70, marker='o', linewidth=2)
            ax.scatter(fn_points.index, fn_points['Mahalanobis_Distance'], color='purple', label='FN (Type II Error)', s=70, marker='x')
            ax.axhline(y=self.md_threshold, color='black', linestyle='--', label=f'Classification Threshold ({self.md_threshold:.2f})')
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Mahalanobis Distance')
            ax.legend(fontsize='x-small', loc='upper left')
            ax.grid(True)
            if set_ylim: ax.set_ylim(*set_ylim)

        # R6C1: Error Overall View
        draw_error_scatter(axes[5, 0], 'Type I/II Error: 1. Overall View', set_ylim=(0, y_max_overall))
        
        # R6C2: Error Zoom Near Normal (close to threshold)
        draw_error_scatter(axes[5, 1], 'Type I/II Error: 2. Zoom Near Normal', set_ylim=(0, y_max_normal))

        # R6C3: Error Zoom Near Thresholds
        draw_error_scatter(axes[5, 2], 'Type I/II Error: 3. Zoom Near Thresholds', set_ylim=(ax3_ylim_min, ax3_ylim_max))
        
        plt.tight_layout()
        
        if save_filename:
            try:
                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(save_filename), exist_ok=True)
                plt.savefig(save_filename, dpi=300) 
                print(f"\nFigure successfully saved as '{save_filename}'")
            except Exception as e:
                print(f"\nWarning: Could not save figure to file {save_filename}. Error: {e}. Showing plot instead.")
                plt.show()
        else:
            plt.show()

        def plot_md_distribution_and_oval(self, result_df, save_filename=None):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # PANEL 1: MD Distribution (Zoomed)
            sns.histplot(data=result_df, x='Mahalanobis_Distance', hue='is_abnormal', 
                         bins=100, ax=ax1, palette={False: 'blue', True: 'red'}, alpha=0.6)
            ax1.axvline(self.md_threshold, color='green', linestyle='--', label=f'Threshold: {self.md_threshold:.2f}')
            ax1.set_xlim(0, self.md_threshold * 3) # Zoom to see the normal distribution
            ax1.set_title('MD Distribution (Zoomed to Threshold Area)', fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.legend()

            # PANEL 2: Mahalanobis Oval (Zoomed)
            if len(self.features) >= 2:
                feat1, feat2 = self.features[0], self.features[1]
                sns.scatterplot(data=result_df, x=feat1, y=feat2, hue='is_abnormal', 
                                palette={False: 'blue', True: 'red'}, ax=ax2, alpha=0.5, s=30)
                
                cov = self.normal_data[[feat1, feat2]].cov().values
                pos = self.median_3sigma[[feat1, feat2]].values
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * self.md_threshold * np.sqrt(np.maximum(vals, 0))
                
                ellipse = patches.Ellipse(xy=pos, width=width, height=height, angle=theta,
                                          edgecolor='green', fc='none', lw=2, linestyle='--',
                                          label=f'99% MD Oval')
                ax2.add_patch(ellipse)
                
                # Auto-Zoom logic to ensure oval is visible regardless of outlier distance
                padding = 1.5
                ax2.set_xlim(pos[0] - width * padding, pos[0] + width * padding)
                ax2.set_ylim(pos[1] - height * padding, pos[1] + height * padding)
                
                ax2.set_title(f'Mahalanobis Oval (Zoomed on Normal Space)', fontweight='bold')
                ax2.set_xlabel(rf'{feat1} ($\Omega$)')
                ax2.set_ylabel(rf'{feat2} ($\Omega$)')
                ax2.legend()
            
            plt.tight_layout()
            if save_filename:
                plt.savefig(save_filename, dpi=300)
                print(f"Distribution plot saved to: {save_filename}")

            def plot_md_distribution_and_oval(self, result_df, save_filename=None):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                sns.histplot(data=result_df, x='Mahalanobis_Distance', hue='is_abnormal', bins=100, ax=ax1)
                ax1.axvline(self.md_threshold, color='green', linestyle='--')
                if len(self.features) >= 2:
                    f1, f2 = self.features[0], self.features[1]
                    sns.scatterplot(data=result_df, x=f1, y=f2, hue='is_abnormal', ax=ax2)
                plt.tight_layout()
                if save_filename: plt.savefig(save_filename)

# FOR Q-Q PLOT
def plot_chi2_qq_plot(mts_analyzer, result_df, save_filename):
    
    # 1. Select the MDs of the clean 'OK' data
    md_ok_data = result_df[result_df[mts_analyzer.status_col] == mts_analyzer.normal_status]['Mahalanobis_Distance'].dropna()
    
    if md_ok_data.empty:
        print("Warning: Cannot generate Q-Q plot. No 'OK' data points found.")
        return

    # 2. Convert MD to MD^2 (as Chi-Squared distribution models MD^2)
    mds_sq_ok = md_ok_data.values**2
    dof = len(mts_analyzer.features)

    # 3. Create the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use probplot from scipy.stats for Q-Q plot against theoretical chi2
    probplot(
        mds_sq_ok, 
        dist=chi2, 
        sparams=(dof,), # Degrees of freedom parameter for chi2
        plot=ax, 
        fit=True 
    )

    ax.set_title(f'Q-Q Plot: Mahalanobis Distance Squared ($MD^2$) vs. $\\chi^2_{{{dof}}}$', fontsize=12)
    ax.set_xlabel('Theoretical Chi-Squared Quantiles', fontsize=10)
    ax.set_ylabel('Empirical $MD^2$ Quantiles', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 4. Save the figure
    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=300) 
        print(f"\nSuccessfully saved Chi-Squared Q-Q plot as '{save_filename}'")
    except Exception as e:
        print(f"\nWarning: Could not save Q-Q plot to file {save_filename}. Error: {e}.")
        plt.show() 
    plt.close(fig)


# FOR TWO-SAMPLE HOTELLING'S T2
def calculate_hotellings_t2(test_df_raw, benchmark_df_raw, features, ir_thresholds, chi2_alpha=0.001, significance_level=0.05):
#Performs Two-Sample Hotelling's T2 test to compare the mean two batches.
    #Steps:
    #1. Filter both batches to get the clean 'OK' samples.
    #2. Calculate means and sample sizes for both clean batches.
    #3. Calculate the pooled covariance matrix (Sp).
    #4. Calculate the T2 statistic.
    #5. Convert T2 to the F-statistic and find the P-value. 
    #Returns: A dictionary with T2 results or None if data is insufficient.
    
    # 1. Filter both datasets using the same criteria
    # The filtering process modifies the DF in place to add the 'Actual_Status' column
    
    # Filter Test Data (Batch B)
    test_df_filtered, _, _, _ = unsupervised_filter_and_label(test_df_raw, features, ir_thresholds, chi2_alpha)
    test_data_ok = test_df_filtered[test_df_filtered['Actual_Status'] == 'OK'][features].copy()
    
    # Filter Benchmark Data (Batch A)
    benchmark_df_filtered, _, _, _ = unsupervised_filter_and_label(benchmark_df_raw, features, ir_thresholds, chi2_alpha)
    benchmark_data_ok = benchmark_df_filtered[benchmark_df_filtered['Actual_Status'] == 'OK'][features].copy()
    
    N_A = len(benchmark_data_ok)
    N_B = len(test_data_ok)
    P = len(features)
    
    if N_A <= P or N_B <= P:
        print(f"\n--- Hotelling's T2 Test Skipped ---")
        print(f"Warning: Insufficient clean 'OK' samples (N_A={N_A}, N_B={N_B}) or not enough degrees of freedom (P={P}). Requires N > P.")
        return None

    # 2. Calculate means and covariance matrices
    mean_A = benchmark_data_ok.mean().values
    mean_B = test_data_ok.mean().values
    
    cov_A = benchmark_data_ok.cov().values
    cov_B = benchmark_data_ok.cov().values # Using benchmark cov as a proxy for both if data is assumed similar
    
    # Check for singularity before pooling
    if np.linalg.det(cov_A) < 1e-12 or np.linalg.det(cov_B) < 1e-12:
        print("Warning: Covariance matrix of one or both batches is singular. Cannot perform Two-Sample T2 test.")
        return None

    # 3. Calculate Pooled Covariance Matrix (Sp)
    # Sp = [ (NA - 1) * CovA + (NB - 1) * CovB ] / (NA + NB - 2)
    df_pooled = N_A + N_B - 2
    
    try:
        Sp = ( (N_A - 1) * cov_A + (N_B - 1) * cov_B ) / df_pooled
        inv_Sp = np.linalg.inv(Sp)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        print("Warning: Pooled covariance matrix is singular. Using pseudo-inverse.")
        Sp = ( (N_A - 1) * cov_A + (N_B - 1) * cov_B ) / df_pooled
        inv_Sp = np.linalg.pinv(Sp)

    # 4. Calculate T2 Statistic
    # T2 = [ NA * NB / (NA + NB) ] * (mean_A - mean_B)T * Sp^-1 * (mean_A - mean_B)
    
    mean_diff = mean_A - mean_B
    T2 = (N_A * N_B / (N_A + N_B)) * np.dot(np.dot(mean_diff.T, inv_Sp), mean_diff)

    # 5. Convert T2 to the F-statistic and find the P-value
    # F = [ (NA + NB - P - 1) / (P * (NA + NB - 2)) ] * T2
    
    f_stat_numerator = (N_A + N_B - P - 1)
    f_stat_denominator = (P * (N_A + NB - 2))
    
    # Check for valid degrees of freedom
    if f_stat_denominator == 0 or f_stat_numerator <= 0:
        print("Warning: Degrees of freedom invalid for F-statistic calculation.")
        return None
        
    F_statistic = (f_stat_numerator / f_stat_denominator) * T2
    
    df1 = P                  # Numerator degrees of freedom
    df2 = N_A + N_B - P - 1  # Denominator degrees of freedom

    # Calculate P-value (Survival function: 1 - CDF)
    p_value = f.sf(F_statistic, df1, df2)
    
    result = {
        'Test_Status': 'SUCCESS',
        'N_Benchmark': N_A,
        'N_Test': N_B,
        'Degrees_of_Freedom_1': df1,
        'Degrees_of_Freedom_2': df2,
        'T2_Statistic': T2,
        'F_Statistic': F_statistic,
        'P_Value': p_value,
        'Significance_Level': significance_level,
        'Mean_Shift_Detected': 'YES' if p_value < significance_level else 'NO'
    }
    
    print(f"\n--- Hotelling's T2 Test Results (Test Mean vs. Benchmark Mean) ---")
    print(f"P-value: {p_value:.6f}")
    print(f"Mean Shift Detected (P < {significance_level})?: {result['Mean_Shift_Detected']}")
    
    return result


def unsupervised_filter_and_label(df, features, ir_thresholds, chi2_alpha=0.001):
    
######important step!!!!!!#####
    #Stage 1 Filtering (REVISED 3-Stage):
    #1. IR check
    #2. Chi-Squared Multivariate Outlier check (on IR-Passed group)
    #3. 3-sigma check (on IR & Chi2-Passed group)

    df = df.copy()
    df['Actual_Status'] = 'OK'
    # Initialize new p-value column
    df['Chi2_MD_P_Value'] = np.nan 
    
    # 1. IR Threshold Check
    ir_fail_mask = pd.Series(False, index=df.index)
    for feature in ir_thresholds:
        if feature in features:
            # Check for IR failure on all samples
            ir_fail_mask = ir_fail_mask | (df[feature] < ir_thresholds[feature])
            
    # Set IR failures as NG immediately
    df.loc[ir_fail_mask, 'Actual_Status'] = 'NG'
    ir_fail_indices = df[ir_fail_mask].index.tolist()
    
    # The IR Passed group
    ir_passed_df = df[~ir_fail_mask].copy()

    # 2. Chi-Squared Multivariate Outlier Check (on IR-Passed Group)
    chi2_fail_indices = []
    
    if not ir_passed_df.empty:
        ir_passed_features = ir_passed_df[features]
        
        # Check if we have enough data points to compute 
        if len(ir_passed_features) > len(features):
            # Calculate MD for the IR-Passed group relative to its own space
            ir_median = ir_passed_features.median().values
            
            # Calculate Inverse Covariance (using pseudo-inverse for stability)
            try:
                cov_matrix = ir_passed_features.cov().values
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                cov_matrix = ir_passed_features.cov().values
                inv_cov = np.linalg.pinv(cov_matrix)
            
            # Calculate MD^2 (Hotelling's T^2 for single-sample)
            data_minus_median = ir_passed_features.values - ir_median.reshape(1, -1)
            mds_sq = np.einsum('ij,jk,ik->i', data_minus_median, inv_cov, data_minus_median)
            
            # Hotelling's T2 / Chi-Squared P-VALUE CALCULATION
            dof = len(features)
            # Use the survival function (1 - CDF) to get the p-value
            p_values = chi2.sf(mds_sq, dof)
            
            # Store the p-values in the original DataFrame
            df.loc[ir_passed_df.index, 'Chi2_MD_P_Value'] = p_values
            
            # Chi-Squared Threshold for multivariate outliers (99% confidence)
            chi2_threshold = chi2.ppf(1 - chi2_alpha, dof) 

            # Filtering Mask
            chi2_fail_mask = mds_sq > chi2_threshold # equivalent to p_values < chi2_alpha
            
            # Flag Chi2 failures as NG
            chi2_fail_indices = ir_passed_df.index[chi2_fail_mask].tolist()
            df.loc[chi2_fail_indices, 'Actual_Status'] = 'NG'
        else:
            print("Warning: Insufficient data points in IR-Passed group to compute stable covariance matrix for Chi2 filter. Skipping this step.")
        
        # The IR and Chi2 Passed group (which are now only those still 'OK')
        ir_chi2_passed_df = df[df['Actual_Status'] == 'OK'].copy()
    else:
        ir_chi2_passed_df = pd.DataFrame()

    # 3 Z-Gate (+/- 4 Sigma) Filter
    if not ir_chi2_passed_df.empty:
        # Calculate stats for the Z-Gate from the current survivors
        z_gate_median = ir_chi2_passed_df[features].median()
        z_gate_std = ir_chi2_passed_df[features].std()
        
        # Identify points outside +/- 4 Sigma
        z_gate_fail_mask = pd.Series(False, index=ir_chi2_passed_df.index)
        for feature in features:
            if z_gate_std[feature] > 0:
                z_vals = (ir_chi2_passed_df[feature] - z_gate_median[feature]) / z_gate_std[feature]
                z_gate_fail_mask = z_gate_fail_mask | (z_vals.abs() > 4)
        
        # Label these extreme outliers as NG in the main dataframe
        z_gate_fail_indices = ir_chi2_passed_df.index[z_gate_fail_mask].tolist()
        df.loc[z_gate_fail_indices, 'Actual_Status'] = 'NG'
        
        # Update the passed group so the 3-Sigma check uses the cleanest possible data
        ir_chi2_passed_df = df[df['Actual_Status'] == 'OK'].copy()

    # 4. 3-Sigma Univariate Outlier Check (on IR + Chi2 Passed Group)
    sigma_fail_indices = []
    
    if not ir_chi2_passed_df.empty:
        # Calculate 3-sigma stats on the IR/Chi2 Passed Group
        passed_median = ir_chi2_passed_df[features].median()
        passed_std = ir_chi2_passed_df[features].std()
        
        for feature in features:
            median, std = passed_median[feature], passed_std[feature]
            
            if std == 0:
                # If std is 0, any deviation from median is an outlier
                fail_mask = (ir_chi2_passed_df[feature] != median)
            else:
                upper_bound, lower_bound = median + 3 * std, median - 3 * std
                fail_mask = (ir_chi2_passed_df[feature] > upper_bound) | (ir_chi2_passed_df[feature] < lower_bound)
            
            # Collect indices of 3-sigma failures within the IR/Chi2 Passed group
            sigma_fail_indices.extend(ir_chi2_passed_df[fail_mask].index.tolist())
            
        # 3. Flag the 3-sigma failures
        sigma_fail_indices = list(set(sigma_fail_indices))
        df.loc[sigma_fail_indices, 'Actual_Status'] = 'NG'
        
    # Return the four sets of indices (Original DF, IR, Chi2, Sigma)
    return df, ir_fail_indices, chi2_fail_indices, sigma_fail_indices

def main():
    parser = argparse.ArgumentParser(description="Run two-stage, unsupervised MTS analysis with filtering and labeling.")
    parser.add_argument('--file', type=str, required=True, help='Input Normalised CSV file')
    parser.add_argument('--benchmark_file', type=str, default=None, help='Path to benchmark CSV.')
    parser.add_argument('--ir2_threshold', type=float, required=True, help='IR2 threshold.')
    parser.add_argument('--ir3_threshold', type=float, required=True, help='IR3 threshold.')
    parser.add_argument('--ir4_threshold', type=float, required=True, help='IR4 threshold.')
    parser.add_argument('--md_quantile', type=float, default=0.99, help='Quantile for MTS.')
    parser.add_argument('--plot_t2', type=float, default=0.99, help='Percentile for Threshold.')
    parser.add_argument('--target_alpha', type=float, default=0.05, help='Target Type I Error.')
    parser.add_argument('--target_beta', type=float, default=0.1, help='Target Type II Error.')
    parser.add_argument('--output_dir', type=str, default='.', help='Folder to save results')
    
    args = parser.parse_args()
    
    # Set output folder to current directory (or what is passed in --output_dir)
    output_folder = args.output_dir
    
    # Commented out the auto-generation of the "sorting result" subfolder
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # PATHING (check all outputs wanted are included)
    input_file = args.file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    quantile_str = f"{int(args.md_quantile * 100)}"
    METHOD_NAME = "MTS with Hotelling"
    current_date = datetime.now().strftime('%Y%m%d')
    
    # --- COMMENTED OUT: Logic that forces a separate folder ---
    # output_folder = f"sorting result_{base_name}_{METHOD_NAME}"
    # os.makedirs(output_folder, exist_ok=True)
    # -----------------------------------------------------------
    
    # Setting output_folder to current directory to ensure files save in script location
    output_folder = "."

    output_filtered_file = os.path.join(output_folder, f'filtered_{METHOD_NAME}_{quantile_str}_{base_name}.csv')
    output_summary_file = os.path.join(output_folder, f'summary_results_{METHOD_NAME}_{quantile_str}_{base_name}.csv')
    output_plot_file = os.path.join(output_folder, f'summary-MTS_figures_{base_name}_{METHOD_NAME}_{quantile_str}.png')
    output_qq_plot_file = os.path.join(output_folder, f'q-q_plot_{METHOD_NAME}_{quantile_str}_{base_name}.png')
    output_params_file = os.path.join(output_folder, f'parameter_{quantile_str}_{base_name}.csv')
    output_all_flagged_file = os.path.join(output_folder, f'Initial_flagged_points_{quantile_str}_{base_name}.csv')
    output_conservative_passed_file = os.path.join(output_folder, f'Acceptable_by_MD_{quantile_str}_{base_name}.csv')

    electrical_features = ['IR2', 'IR3', 'IR4']
    ir_thresholds = {'IR2': args.ir2_threshold, 'IR3': args.ir3_threshold, 'IR4': args.ir4_threshold}
# 1. Load data and clean headers
    raw_df = pd.read_csv(input_file)
    raw_df.columns = [str(c).strip() for c in raw_df.columns] # Remove hidden spaces

    # 2. SMART COLUMN MAPPING (This is the critical part)
    # This finds the actual names used in your CSV
    mapping = {}
    for col in raw_df.columns:
        col_up = col.upper()
        # Look for the numeric value column, ignoring 'status' or 'flag' columns
        if 'IR2' in col_up and 'STATUS' not in col_up: mapping['IR2'] = col
        if 'IR3' in col_up and 'STATUS' not in col_up: mapping['IR3'] = col
        if 'IR4' in col_up and 'STATUS' not in col_up: mapping['IR4'] = col

    # 3. Check if we found the columns
    if not all(k in mapping for k in ['IR2', 'IR3', 'IR4']):
        print(f"❌ Error: Could not find columns for IR2, IR3, or IR4.")
        print(f"Columns found in file: {list(raw_df.columns)}")
        sys.exit(1)

    # 4. Use the ACTUAL column names found
    electrical_features = [mapping['IR2'], mapping['IR3'], mapping['IR4']]

    # 5. Define thresholds using the mapped names (Fixes the KeyError)
    ir_thresholds = {
        mapping['IR2']: float(args.ir2_threshold),
        mapping['IR3']: float(args.ir3_threshold),
        mapping['IR4']: float(args.ir4_threshold)
    }

    # 6. Clean numeric data using the mapped names
    for col in electrical_features:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)

    # Now your original line 1057 will work:
    filtered_df, ir_fail_indices, chi2_fail_indices, sigma_fail_indices = unsupervised_filter_and_label(
        raw_df, electrical_features, ir_thresholds
    )

    # Stage 2: MTS
    mts_analyzer = MTS(filtered_df, features=electrical_features, status_col='Actual_Status', 
                       normal_status='OK', ir_thresholds=ir_thresholds)
    result_df = mts_analyzer.detect_abnormal(md_quantile=args.md_quantile)
    error_summary = mts_analyzer.calculate_type1_type2_errors(result_df)

    # ADDED: EXPORT ALL MD VALUES
    md_output_file = os.path.join(output_folder, f'MD_{base_name}.csv')
    md_export_df = result_df[['original_row_number', 'Mahalanobis_Distance']].copy()
    md_export_df.rename(columns={'original_row_number': 'data no', 'Mahalanobis_Distance': 'MD'}, inplace=True)
    md_export_df.to_csv(md_output_file, index=False)
    print(f"Full Mahalanobis Distances exported to: {md_output_file}")

    # HOTELLING'S T2 for comparing with benchmark~
    t2_results = None 
    
    if args.benchmark_file:
        try:
            benchmark_df = pd.read_csv(args.benchmark_file)
            benchmark_df.columns = benchmark_df.columns.str.strip()
            t2_results = calculate_hotellings_t2(raw_df, benchmark_df, electrical_features, ir_thresholds)
        except Exception as e:
            print(f"Warning: Hotelling's T2 test failed. Error: {e}")

    # classification on why they are rejected/abnormal
    is_screened_ng = (result_df[mts_analyzer.status_col] != mts_analyzer.normal_status)
    is_md_ok = (result_df['Mahalanobis_Distance'] <= mts_analyzer.md_threshold)
    is_ir_fail = (result_df['abnormal_type'] == 'initial_ir_threshold_ng')
    initial_ng_but_md_ok = result_df[is_screened_ng & is_md_ok & ~is_ir_fail].copy()
    
    corrected_is_abnormal = result_df['is_abnormal'].copy()
    corrected_is_abnormal.loc[initial_ng_but_md_ok.index] = False 
    total_flagged_sample = corrected_is_abnormal.sum()
    
    final_ng_mask = corrected_is_abnormal 
    final_ng_breakdown = result_df[final_ng_mask]['abnormal_type'].value_counts()

    output_ng_only_file = os.path.join(output_folder, f'NG_points_only_{base_name}_{quantile_str}.csv')
    ng_only_df = result_df[final_ng_mask].copy()
    if not ng_only_df.empty:
            ng_diagnosed = mts_analyzer.diagnose_abnormal_features(ng_only_df)
            ng_diagnosed.to_csv(output_ng_only_file, index=False)
            
            output_mts_ng_file = os.path.join(output_folder, f'MTS_NG_{base_name}.csv')
            mts_ng_table = mts_analyzer.export_points_only_table(ng_only_df)
            mts_ng_table.to_csv(output_mts_ng_file, index=False)
            print(f"Simplified MTS NG table exported to: {output_mts_ng_file}")
    else:
        pd.DataFrame(columns=result_df.columns).to_csv(output_ng_only_file, index=False)
        print("No NG points found to export.")
     
    ng_ir_reject = final_ng_breakdown.get('initial_ir_threshold_ng', 0)
    ng_md_reject = final_ng_breakdown.get('MD_Reject', 0)
    ng_3sigma_reject = final_ng_breakdown.get('3_Sigma_Reject', 0) 

    total_samples = len(result_df)
    total_ng_rate = total_flagged_sample / total_samples if total_samples > 0 else 0
    
    NEON_YELLOW = "\033[93m"
    RESET = "\033[0m"
    total_ng_rate_str = f"{total_ng_rate:.2%}"
    
    summary_list = [
        {'Metric': 'N_total', 'Value': total_samples},
        {'Metric': 'N_actual normal', 'Value': error_summary['N_Actual_Normal']},
        {'Metric': 'N_actual abnormal', 'Value': error_summary['N_Actual_Abnormal']},
        {'Metric': f'Type I error rate (alpha = {args.target_alpha})', 'Value': f"{error_summary['Type_I_Error_Rate']:.5f}"},
        {'Metric': f'Type II error rate (beta = {args.target_beta})', 'Value': f"{error_summary['Type_II_Error_Rate']:.5f}"},
        {'Metric': 'Total NG', 'Value': total_flagged_sample},
        {'Metric': 'Total NG rate', 'Value': total_ng_rate_str},
        {'Metric': '---------------------', 'Value': ''},
        {'Metric': 'NG type IR threshold reject', 'Value': ng_ir_reject},
        {'Metric': 'NG type MD reject', 'Value': ng_md_reject},
        {'Metric': 'NG type 3 sigma reject', 'Value': ng_3sigma_reject},
        {'Metric': '-------------------------', 'Value': ''},
        {'Metric': 'Mahalanobis Distance Threshold', 'Value': f"{mts_analyzer.md_threshold:.4f}"}
    ]

    print(f"\n{NEON_YELLOW}Metric: Total NG rate | Value: {total_ng_rate_str}{RESET}\n")
    
    summary_list.append({'Metric': '--- Hotelling’s T2 Comparison ---', 'Value': ''})
    
    if t2_results:
        summary_list.extend([
            {'Metric': 'N Benchmark (OK)', 'Value': t2_results['N_Benchmark']},
            {'Metric': 'N Test (OK)', 'Value': t2_results['N_Test']},
            {'Metric': 'T2 Statistic', 'Value': f"{t2_results['T2_Statistic']:.4f}"},
            {'Metric': 'F Statistic', 'Value': f"{t2_results['F_Statistic']:.4f}"},
            {'Metric': 'P-Value', 'Value': f"{t2_results['P_Value']:.6f}"},
            {'Metric': 'Mean Shift Detected', 'Value': t2_results['Mean_Shift_Detected']}
        ])
    else:
        summary_list.append({'Metric': 'Status', 'Value': 'No benchmark was provided for comparison.'})

    pd.DataFrame(summary_list).to_csv(output_summary_file, index=False)

    mts_analyzer.plot_results(result_df, secondary_threshold_percentile=args.plot_t2, save_filename=output_plot_file)
    plot_chi2_qq_plot(mts_analyzer, result_df, save_filename=output_qq_plot_file)

    stats_data = []
    for f in electrical_features:
        for dataset_type, med, s, avg in [
            ('Raw', mts_analyzer.raw_median[f], mts_analyzer.raw_std[f], mts_analyzer.raw_mean[f]), 
            ('Filtered', mts_analyzer.median_3sigma[f], mts_analyzer.std_3sigma[f], mts_analyzer.mean_3sigma[f])
        ]:
            stats_data.append({
                'Feature': f, 
                'Dataset': dataset_type, 
                'Mean': avg,
                'Median': med, 
                'Sigma': s,
                '-4 Sigma': med - 4*s, 
                '-3 Sigma': med - 3*s, 
                '-2 Sigma': med - 2*s, 
                '-1 Sigma': med - 1*s,
                '+1 Sigma': med + 1*s, 
                '+2 Sigma': med + 2*s, 
                '+3 Sigma': med + 3*s, 
                '+4 Sigma': med + 4*s
            })
            
    stats_path = os.path.join(output_folder, f'feature_statistics_{base_name}_{quantile_str}.csv')
    pd.DataFrame(stats_data).to_csv(stats_path, index=False)

    save_2d_ellipsoid_report(result_df, electrical_features, mts_analyzer.md_threshold, 
                         mts_analyzer.median_3sigma, result_df[electrical_features].cov(), 
                         base_name, output_folder)

    save_3d_ellipsoid_report(result_df, electrical_features, mts_analyzer.md_threshold, 
                         mts_analyzer.median_3sigma, result_df[electrical_features].cov(), 
                         base_name, output_folder)
    
    print(f"Analysis complete. Statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()
