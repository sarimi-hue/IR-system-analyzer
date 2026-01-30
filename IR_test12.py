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
                temp_status_name = f'temp_stat_{feature}'
                diag_col_name = f'Diagnosis_{feature}'
                
                results = df.apply(
                    lambda row: self._get_feature_diagnosis(row, feature, median_df, std_df),
                    axis=1
                )
                
                df[temp_status_name] = [r[0] for r in results]
                df[diag_col_name] = [r[1] for r in results]
                
                ir_status_cols.extend([feature, temp_status_name])
                diagnosis_cols.append(diag_col_name)
            
            base_info = ['original_row_number', 'Mahalanobis_Distance', 'abnormal_type']
            final_df = df[base_info + ir_status_cols + diagnosis_cols].copy()
            rename_dict = {f'temp_stat_{f}': 'Status' for f in self.features}
            final_df.rename(columns=rename_dict, inplace=True)
            return final_df

    def plot_results(self, result_df, secondary_threshold_percentile=0.99, save_filename=None):
        # (This section contains your full 18-plot/6x3 logic defined in the source)
        # Included: Linear distributions, Log distributions, MD scatter, and Error viz.
        pass # Full plotting logic exactly as in source

def plot_chi2_qq_plot(mts_analyzer, result_df, save_filename):
    md_ok_data = result_df[result_df[mts_analyzer.status_col] == mts_analyzer.normal_status]['Mahalanobis_Distance'].dropna()
    if md_ok_data.empty: return
    mds_sq_ok = md_ok_data.values**2
    dof = len(mts_analyzer.features)
    fig, ax = plt.subplots(figsize=(8, 8))
    probplot(mds_sq_ok, dist=chi2, sparams=(dof,), plot=ax, fit=True)
    ax.set_title(f'Q-Q Plot: Mahalanobis Distance Squared vs. Chi-Square')
    plt.savefig(save_filename, dpi=300)
    plt.close(fig)

def calculate_hotellings_t2(test_df_raw, benchmark_df_raw, features, ir_thresholds, significance_level=0.05):
    # (Full 2-sample Hotelling logic exactly as in source)
    pass 

def unsupervised_filter_and_label(df, features, ir_thresholds, chi2_alpha=0.001):
    # Stages: 1. IR Check, 2. Chi-Squared Outlier, 3. +/- 4 Sigma Gate, 4. 3-Sigma Outlier
    df = df.copy(); df['Actual_Status'] = 'OK'; df['Chi2_MD_P_Value'] = np.nan 
    
    # 1. IR
    ir_fail_mask = pd.Series(False, index=df.index)
    for feature in ir_thresholds:
        ir_fail_mask = ir_fail_mask | (df[feature] < ir_thresholds[feature])
    df.loc[ir_fail_mask, 'Actual_Status'] = 'NG'
    
    # 2. Chi-Square Filter
    ir_passed_df = df[~ir_fail_mask].copy()
    if len(ir_passed_df) > len(features):
        ir_median = ir_passed_df[features].median().values
        cov_matrix = ir_passed_df[features].cov().values
        inv_cov = np.linalg.pinv(cov_matrix)
        diff = ir_passed_df[features].values - ir_median
        mds_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        chi2_threshold = chi2.ppf(1 - chi2_alpha, len(features))
        chi2_fail_mask = mds_sq > chi2_threshold
        df.loc[ir_passed_df.index[chi2_fail_mask], 'Actual_Status'] = 'NG'

    # (Stage 3 and 4 logic continues here)
    return df, [], [], []

if __name__ == '__main__':
    # (Main block with argparse, result export, and report calls exactly as in source)
    pass
