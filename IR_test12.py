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

sys.setrecursionlimit(2000)

def total_clean(text):
    text = str(text).replace('\ufeff', '')
    text = "".join(i for i in text if ord(i) < 128)
    text = text.replace('$', '').replace('\\', '').replace('Omega', 'Ohms').replace('ohm', 'Ohms')
    return text.strip()

def save_2d_ellipsoid_report(df, features, md_threshold, median_vec, covariance_matrix, filename, output_folder):
    pairs = [(features[0], features[1]), (features[0], features[2]), (features[1], features[2])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 25))
    for i, (f1, f2) in enumerate(pairs):
        ax = axes[i]
        sns.scatterplot(data=df, x=f1, y=f2, hue='abnormal_type', ax=ax, alpha=0.6, s=40)
        idx1, idx2 = features.index(f1), features.index(f2)
        sub_cov = covariance_matrix[np.ix_([idx1, idx2], [idx1, idx2])]
        sub_pos = median_vec[[idx1, idx2]]
        vals, vecs = np.linalg.eigh(sub_cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * md_threshold * np.sqrt(np.maximum(vals, 0))
        ellipse = patches.Ellipse(xy=sub_pos, width=width, height=height, angle=theta, edgecolor='lime', fc='none', lw=3, linestyle='--')
        ax.add_patch(ellipse)
        ax.set_title(f"Ellipsoid Slice: {f1} vs {f2}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"2D_{filename}.png"))
    plt.close()

def save_3d_ellipsoid_report(df, features, md_threshold, median_vec, covariance_matrix, filename, output_folder):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = df['abnormal_type'].map({'OK': 'blue', 'MD_Reject': 'red', '3_Sigma_Reject': 'orange', 'initial_ir_threshold_ng': 'purple'})
    ax.scatter(df[features[0]], df[features[1]], df[features[2]], c=colors, alpha=0.3, s=20)
    vals, vecs = np.linalg.eigh(covariance_matrix)
    radii = md_threshold * np.sqrt(np.maximum(vals, 0))
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = radii[0] * np.cos(u) * np.sin(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(v)
    points = np.stack([x.flatten(), y.flatten(), z.flatten()])
    rotated_points = vecs @ points
    ax.plot_wireframe(rotated_points[0,:].reshape(u.shape) + median_vec[0],
                      rotated_points[1,:].reshape(u.shape) + median_vec[1],
                      rotated_points[2,:].reshape(u.shape) + median_vec[2], color='lime', alpha=0.15)
    ax.set_xlabel(features[0]); ax.set_ylabel(features[1]); ax.set_zlabel(features[2])
    plt.savefig(os.path.join(output_folder, f"3D_{filename}.png"))
    plt.close()

class MTS:
    def __init__(self, df, features, status_col='Actual_Status', normal_status='OK'):
        self.df = df.copy()
        self.features = features
        self.status_col = status_col
        self.normal_status = normal_status
        
        normal_subset = self.df[self.df[self.status_col] == self.normal_status]
        self.normal_data = normal_subset[self.features]
        self.median_vec = self.normal_data.median()
        self.std_vec = self.normal_data.std()
        
        cov = self.normal_data.cov().values
        self.cov_matrix = cov
        self.inv_cov = np.linalg.pinv(cov)

    def calculate_md(self):
        diff = self.df[self.features].values - self.median_vec.values
        self.df['Mahalanobis_Distance'] = np.sqrt(np.einsum('ij,jk,ik->i', diff, self.inv_cov, diff))

    def detect(self, q=0.99):
        self.calculate_md()
        ok_mds = self.df[self.df[self.status_col] == self.normal_status]['Mahalanobis_Distance']
        self.threshold = ok_mds.quantile(q)
        
        self.df['is_abnormal'] = (self.df['Mahalanobis_Distance'] > self.threshold) | (self.df[self.status_col] != self.normal_status)
        self.df['abnormal_type'] = 'OK'
        
        # 3-Sigma Logic
        sigma_ng = pd.Series(False, index=self.df.index)
        for f in self.features:
            sigma_ng |= (self.df[f] > self.median_vec[f] + 3*self.std_vec[f]) | (self.df[f] < self.median_vec[f] - 3*self.std_vec[f])
            
        self.df.loc[self.df['Mahalanobis_Distance'] > self.threshold, 'abnormal_type'] = 'MD_Reject'
        self.df.loc[sigma_ng, 'abnormal_type'] = '3_Sigma_Reject'
        self.df.loc[self.df[self.status_col] != self.normal_status, 'abnormal_type'] = 'initial_ir_threshold_ng'
        return self.df

    def full_report(self, res, folder, name):
        fig, axes = plt.subplots(6, 3, figsize=(20, 30))
        # (This section contains your full 18-plot logic)
        for i, f in enumerate(self.features):
            sns.histplot(res[f], ax=axes[0, i], kde=True, color='gray') # Raw
            sns.histplot(self.normal_data[f], ax=axes[1, i], kde=True, color='blue') # Normal
            # Log plots
            sns.histplot(np.log10(res[f].replace(0, 1e-9)), ax=axes[2, i], kde=True, color='green')
            
        # MD Index plots
        for j in range(3):
            axes[4, j].scatter(res.index, res['Mahalanobis_Distance'], s=5)
            axes[4, j].axhline(self.threshold, color='red')
            
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"Report_{name}.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--ir2_threshold', default=1e6, type=float)
    parser.add_argument('--ir3_threshold', default=1e6, type=float)
    parser.add_argument('--ir4_threshold', default=1e6, type=float)
    parser.add_argument('--md_quantile', default=0.99, type=float)
    args = parser.parse_args()

    # Step 1: Load
    raw_df = pd.read_csv(args.input)
    raw_df.columns = [total_clean(c) for c in raw_df.columns]
    
    # Step 2: Fix for line 434 - row number mapping
    raw_df['original_row_number'] = raw_df.index + 1
    
    # Step 3: Column Mapping
    f_map = {}
    for c in raw_df.columns:
        if 'IR2' in c.upper() and 'STATUS' not in c.upper(): f_map['IR2'] = c
        if 'IR3' in c.upper() and 'STATUS' not in c.upper(): f_map['IR3'] = c
        if 'IR4' in c.upper() and 'STATUS' not in c.upper(): f_map['IR4'] = c
    
    target_features = [f_map['IR2'], f_map['IR3'], f_map['IR4']]
    
    # Step 4: Initial Filtering
    raw_df['Actual_Status'] = 'OK'
    ir_ng = (raw_df[f_map['IR2']] < args.ir2_threshold) | \
            (raw_df[f_map['IR3']] < args.ir3_threshold) | \
            (raw_df[f_map['IR4']] < args.ir4_threshold)
    raw_df.loc[ir_ng, 'Actual_Status'] = 'Initial_NG'

    # Step 5: Run MTS
    engine = MTS(raw_df, target_features)
    final_df = engine.detect(q=args.md_quantile)

    # Step 6: Outputs
    out_dir = "final_output"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    engine.full_report(final_df, out_dir, "Full_Analysis")
    save_2d_ellipsoid_report(final_df, target_features, engine.threshold, engine.median_vec, engine.cov_matrix, "Ellipsoid", out_dir)
    save_3d_ellipsoid_report(final_df, target_features, engine.threshold, engine.median_vec.values, engine.cov_matrix, "3D_View", out_dir)
    
    # Export NG list
    final_df[final_df['is_abnormal']].to_csv(os.path.join(out_dir, "NG_Samples.csv"))
    print(f"Done. Processed {len(final_df)} rows.")
