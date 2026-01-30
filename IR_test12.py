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
from scipy.stats import chi2, probplot, f
from datetime import datetime

# Increase recursion for complex data structures
sys.setrecursionlimit(2000)

def total_clean(text):
    # Cleaning headers to prevent Matplotlib LaTeX crashes
    text = str(text).replace('\ufeff', '')
    text = "".join(i for i in text if ord(i) < 128)
    text = text.replace('$', '').replace('\\', '').replace('Omega', 'Ohms').replace('ohm', 'Ohms')
    return text.strip()

def unsupervised_filter_and_label(df, features, ir_thresholds):
    # Identifies initial NG points based on raw thresholds before building Mahalanobis Space
    filtered_df = df.copy()
    ir_fail_indices = []
    
    # Check IR Thresholds
    ir_fail_mask = pd.Series(False, index=df.index)
    for feature, thresh in ir_thresholds.items():
        ir_fail_mask |= (df[feature] < thresh)
    
    ir_fail_indices = df.index[ir_fail_mask].tolist()
    
    # Labeling
    filtered_df['Actual_Status'] = 'OK'
    filtered_df.loc[ir_fail_mask, 'Actual_Status'] = 'Initial_IR_NG'
    
    return filtered_df, ir_fail_indices, [], []

def save_2d_ellipsoid_report(df, features, md_threshold, median_vec, covariance_matrix, filename, output_folder):
    pairs = [(features[0], features[1]), (features[0], features[2]), (features[1], features[2])]
    fig, axes = plt.subplots(3, 1, figsize=(8, 18))
    for i, (f1, f2) in enumerate(pairs):
        ax = axes[i]
        sns.scatterplot(data=df, x=f1, y=f2, hue='is_abnormal', palette={False: 'blue', True: 'red'}, ax=ax, alpha=0.5, s=30)
        idx1, idx2 = features.index(f1), features.index(f2)
        sub_cov = covariance_matrix[np.ix_([idx1, idx2], [idx1, idx2])]
        sub_pos = median_vec[[idx1, idx2]]
        vals, vecs = np.linalg.eigh(sub_cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * md_threshold * np.sqrt(np.maximum(vals, 0))
        ellipse = patches.Ellipse(xy=sub_pos, width=width, height=height, angle=theta, edgecolor='green', fc='none', lw=2, linestyle='--')
        ax.add_patch(ellipse)
        ax.set_title(f'2D Slice: {f1} vs {f2}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"2D_ellipsoid_{filename}.png"))
    plt.close()

def save_3d_ellipsoid_report(df, features, md_threshold, median_vec, covariance_matrix, filename, output_folder):
    if len(features) < 3: return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = df['is_abnormal'].map({False: 'blue', True: 'red'})
    ax.scatter(df[features[0]], df[features[1]], df[features[2]], c=colors, alpha=0.4, s=15)
    vals, vecs = np.linalg.eigh(covariance_matrix)
    radii = md_threshold * np.sqrt(np.maximum(vals, 0))
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    sphere_pts = np.stack([(np.cos(u)*np.sin(v)).flatten(), (np.sin(u)*np.sin(v)).flatten(), np.cos(v).flatten()])
    ellipsoid_pts = (vecs @ (radii[:, None] * sphere_pts))
    ax.plot_wireframe(ellipsoid_pts[0,:].reshape(u.shape)+median_vec[0], 
                      ellipsoid_pts[1,:].reshape(u.shape)+median_vec[1], 
                      ellipsoid_pts[2,:].reshape(u.shape)+median_vec[2], color='green', alpha=0.1)
    ax.set_title("3D Mahalanobis Ellipsoid")
    plt.savefig(os.path.join(output_folder, f"3D_ellipsoid_{filename}.png"))
    plt.close()

class MTS:
    def __init__(self, df, features, status_col='Actual_Status', normal_status='OK', ir_thresholds=None):
        self.df = df.copy()
        self.features = features
        self.status_col = status_col
        self.normal_status = normal_status
        self.ir_thresholds = ir_thresholds if ir_thresholds is not None else {}
        self.md_threshold = None
        
        # Build Stats
        self.raw_median = self.df[self.features].median()
        self.raw_std = self.df[self.features].std()
        
        normal_data_subset = self.df[self.df[self.status_col] == self.normal_status]
        if normal_data_subset.empty:
            raise ValueError("No Normal data to build Space.")
            
        self.normal_data = normal_data_subset[self.features]
        self.median_3sigma = self.normal_data.median()
        self.std_3sigma = self.normal_data.std()
        
        cov_matrix = self.normal_data.cov().values
        try:
            self.inv_cov = np.linalg.inv(cov_matrix)
        except:
            self.inv_cov = np.linalg.pinv(cov_matrix)
        self.cov_matrix = cov_matrix

    def calculate_md_all(self):
        data_to_evaluate = self.df[self.features].values
        center = self.median_3sigma.values.reshape(1, -1)
        diff = data_to_evaluate - center
        self.df['Mahalanobis_Distance'] = np.sqrt(np.einsum('ij,jk,ik->i', diff, self.inv_cov, diff))

    def detect_abnormal(self, md_quantile=0.99):
        self.calculate_md_all()
        result_df = self.df.copy()
        ok_mds = result_df[result_df[self.status_col] == self.normal_status]['Mahalanobis_Distance'].dropna()
        self.md_threshold = ok_mds.quantile(md_quantile)
        
        md_is_abnormal = result_df['Mahalanobis_Distance'] > self.md_threshold
        initial_ng = (result_df[self.status_col] != self.normal_status)
        
        result_df['is_abnormal'] = initial_ng | md_is_abnormal
        result_df['abnormal_type'] = 'OK'
        
        # Detailed Diagnosis
        is_outside_3sigma = pd.Series(False, index=result_df.index)
        for f in self.features:
            upper = self.median_3sigma[f] + 3 * self.std_3sigma[f]
            lower = self.median_3sigma[f] - 3 * self.std_3sigma[f]
            is_outside_3sigma |= (result_df[f] > upper) | (result_df[f] < lower)
            
        result_df.loc[initial_ng, 'abnormal_type'] = 'initial_ir_threshold_ng'
        result_df.loc[is_outside_3sigma & (result_df['abnormal_type'] == 'OK'), 'abnormal_type'] = '3_Sigma_Reject'
        result_df.loc[md_is_abnormal & (result_df['abnormal_type'] == 'OK'), 'abnormal_type'] = 'MD_Reject'
        
        return result_df

    def calculate_type1_type2_errors(self, result_df):
        actual_normal = (result_df[self.status_col] == self.normal_status)
        actual_abnormal = ~actual_normal
        pred_abnormal = result_df['is_abnormal']
        
        FP = (actual_normal & pred_abnormal).sum()
        FN = (actual_abnormal & ~pred_abnormal).sum()
        
        result_df['classification_type'] = 'TN (Correct Normal)'
        result_df.loc[actual_abnormal & pred_abnormal, 'classification_type'] = 'TP (Correct Abnormal)'
        result_df.loc[actual_normal & pred_abnormal, 'classification_type'] = 'FP (Type I Error)'
        result_df.loc[actual_abnormal & ~pred_abnormal, 'classification_type'] = 'FN (Type II Error)'
        
        return {
            'N_Actual_Normal': actual_normal.sum(),
            'N_Actual_Abnormal': actual_abnormal.sum(),
            'Type_I_Error_Rate': FP / actual_normal.sum() if actual_normal.sum() > 0 else 0,
            'Type_II_Error_Rate': FN / actual_abnormal.sum() if actual_abnormal.sum() > 0 else 0
        }

    def plot_results(self, result_df, save_filename=None):
        fig, axes = plt.subplots(6, 3, figsize=(18, 24))
        
        # Row 1-2: Linear Dist, Row 3-4: Log Dist
        log_df = result_df.copy()
        for f in self.features:
            log_df[f'ln({f})'] = np.log(log_df[f].replace(0, np.nan))
            
        for i, col in enumerate(self.features):
            # Histograms
            sns.histplot(result_df[col], ax=axes[0, i], color='black', kde=True)
            sns.histplot(self.normal_data[col], ax=axes[1, i], color='blue', kde=True)
            axes[0, i].set_title(f"Raw: {col}")
            axes[1, i].set_title(f"Clean: {col}")
            
        # MD Scatters
        for j in range(3):
            ax = axes[4, j]
            ax.scatter(result_df.index, result_df['Mahalanobis_Distance'], c='blue', alpha=0.5, s=10)
            ax.axhline(self.md_threshold, color='green', linestyle='--')
            
        plt.tight_layout()
        if save_filename: plt.savefig(save_filename, dpi=300)
        plt.close()

# MAIN RUNNER
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR System Analyzer')
    parser.add_argument('--input', required=True)
    parser.add_argument('--ir2_threshold', default=1e6, type=float)
    parser.add_argument('--ir3_threshold', default=1e6, type=float)
    parser.add_argument('--ir4_threshold', default=1e6, type=float)
    parser.add_argument('--md_quantile', default=0.99, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File not found.")
        sys.exit(1)

    # LOADING AND CLEANING
    raw_df = pd.read_csv(args.input)
    raw_df.columns = [total_clean(c) for c in raw_df.columns]
    raw_df['original_row_number'] = raw_df.index + 1
    
    # Identify numeric columns for IR2, IR3, IR4
    mapping = {}
    for col in raw_df.columns:
        c_up = col.upper()
        if 'IR2' in c_up and 'STATUS' not in c_up: mapping['IR2'] = col
        if 'IR3' in c_up and 'STATUS' not in c_up: mapping['IR3'] = col
        if 'IR4' in c_up and 'STATUS' not in c_up: mapping['IR4'] = col

    features = [mapping['IR2'], mapping['IR3'], mapping['IR4']]
    ir_thresholds = {mapping['IR2']: args.ir2_threshold, mapping['IR3']: args.ir3_threshold, mapping['IR4']: args.ir4_threshold}

    for col in features:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)

    # RUN ANALYSIS
    filtered_df, _, _, _ = unsupervised_filter_and_label(raw_df, features, ir_thresholds)
    
    analyzer = MTS(filtered_df, features, status_col='Actual_Status', ir_thresholds=ir_thresholds)
    result_df = analyzer.detect_abnormal(md_quantile=args.md_quantile)
    errors = analyzer.calculate_type1_type2_errors(result_df)

    # OUTPUTS
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_folder = "output"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    analyzer.plot_results(result_df, save_filename=os.path.join(output_folder, f"Plot_{base_name}.png"))
    save_2d_ellipsoid_report(result_df, features, analyzer.md_threshold, analyzer.median_3sigma, analyzer.cov_matrix, base_name, output_folder)
    save_3d_ellipsoid_report(result_df, features, analyzer.md_threshold, analyzer.median_3sigma.values, analyzer.cov_matrix, base_name, output_folder)

    print(f"Analysis complete for {base_name}. Type I Error: {errors['Type_I_Error_Rate']:.2%}")
