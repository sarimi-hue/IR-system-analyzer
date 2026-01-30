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

# Prevent recursion depth errors for large datasets
sys.setrecursionlimit(2000)

def total_clean(text):
    # Removes BOM and bad characters that make plots crash
    text = str(text).replace('\ufeff', '')
    text = "".join(i for i in text if ord(i) < 128)
    text = text.replace('$', '').replace('\\', '').replace('Omega', 'Ohms').replace('ohm', 'Ohms')
    return text.strip()

def unsupervised_filter_and_label(df, features, thresholds):
    # This logic identifies the 'Initial NG' points before the MD analysis starts
    temp_df = df.copy()
    ir_fail = pd.Series(False, index=temp_df.index)
    for f, t in thresholds.items():
        ir_fail |= (temp_df[f] < t)
    
    temp_df['Actual_Status'] = 'OK'
    temp_df.loc[ir_fail, 'Actual_Status'] = 'Initial_IR_NG'
    return temp_df

class MTS:
    def __init__(self, df, features, status_col='Actual_Status', normal_status='OK', ir_thresholds=None):
        self.df = df.copy()
        self.features = features
        self.status_col = status_col
        self.normal_status = normal_status
        self.ir_thresholds = ir_thresholds if ir_thresholds is not None else {}
        
        # Calculate base stats
        self.raw_median = self.df[self.features].median()
        self.raw_std = self.df[self.features].std()
        
        # Isolate the 'Normal' group to build the Mahalanobis Space
        normal_subset = self.df[self.df[self.status_col] == self.normal_status]
        if normal_subset.empty:
            print("Error: No normal data found to build the space.")
            sys.exit(1)
            
        self.normal_data = normal_subset[self.features]
        self.median_3sigma = self.normal_data.median()
        self.std_3sigma = self.normal_data.std()
        
        # Covariance and Matrix Inversion
        cov_matrix = self.normal_data.cov().values
        try:
            self.inv_cov = np.linalg.inv(cov_matrix)
        except:
            self.inv_cov = np.linalg.pinv(cov_matrix)

    def calculate_md(self):
        # The heart of the MTS: calculating the distance for every point
        data_values = self.df[self.features].values
        center = self.median_3sigma.values.reshape(1, -1)
        diff = data_values - center
        self.df['Mahalanobis_Distance'] = np.sqrt(np.einsum('ij,jk,ik->i', diff, self.inv_cov, diff))

    def detect_abnormal(self, md_quantile=0.99):
        self.calculate_md()
        res = self.df.copy()
        
        # Threshold based on the OK group's distribution
        ok_mds = res[res[self.status_col] == self.normal_status]['Mahalanobis_Distance']
        self.md_threshold = ok_mds.quantile(md_quantile)
        
        # Categorize the rejects
        res['is_abnormal'] = (res['Mahalanobis_Distance'] > self.md_threshold) | (res[self.status_col] != self.normal_status)
        res['abnormal_type'] = 'OK'
        
        # Mark 3-Sigma Rejects
        sigma_mask = pd.Series(False, index=res.index)
        for f in self.features:
            upper = self.median_3sigma[f] + 3 * self.std_3sigma[f]
            lower = self.median_3sigma[f] - 3 * self.std_3sigma[f]
            sigma_mask |= (res[f] > upper) | (res[f] < lower)
            
        res.loc[res['Mahalanobis_Distance'] > self.md_threshold, 'abnormal_type'] = 'MD_Reject'
        res.loc[sigma_mask, 'abnormal_type'] = '3_Sigma_Reject'
        res.loc[res[self.status_col] != self.normal_status, 'abnormal_type'] = 'Initial_IR_NG'
        
        return res

    def plot_results(self, result_df, save_filename=None):
        # Setup for the 6x3 report
        fig, axes = plt.subplots(6, 3, figsize=(18, 24))
        
        # Log scaling for the log rows
        log_df = result_df.copy()
        for f in self.features:
            log_df[f'ln({f})'] = np.log(log_df[f].replace(0, np.nan))
        
        # Drawing logic
        # (Simplified for stability, ensuring no NameErrors)
        for i, col in enumerate(self.features):
            # Row 1: Raw Histogram
            sns.histplot(result_df[col], ax=axes[0, i], color='gray', kde=True)
            axes[0, i].set_title(f"Raw Dist: {col}")
            
            # Row 2: Clean Histogram
            sns.histplot(self.normal_data[col], ax=axes[1, i], color='blue', kde=True)
            axes[1, i].set_title(f"Clean Dist: {col}")

        # MD Scatter plotting
        for j in range(3):
            ax = axes[4, j]
            ax.scatter(result_df.index, result_df['Mahalanobis_Distance'], c='blue', s=10, alpha=0.5)
            ax.axhline(self.md_threshold, color='red', linestyle='--')
            ax.set_title("MD Analysis View")

        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename, dpi=300)
        else:
            plt.show()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to CSV')
    parser.add_argument('--ir2_threshold', default=1e6, type=float)
    parser.add_argument('--ir3_threshold', default=1e6, type=float)
    parser.add_argument('--ir4_threshold', default=1e6, type=float)
    parser.add_argument('--md_quantile', default=0.99, type=float)
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.input):
        print("File not found.")
        sys.exit(1)

    raw_df = pd.read_csv(args.input)
    raw_df.columns = [total_clean(c) for c in raw_df.columns]
    raw_df['original_row_number'] = raw_df.index + 1

    # 2. Map Columns
    features = []
    for f_name in ['IR2', 'IR3', 'IR4']:
        found = [c for c in raw_df.columns if f_name in c.upper() and 'STATUS' not in c.upper()]
        if found: features.append(found[0])

    # 3. Process
    thresh_map = {features[0]: args.ir2_threshold, features[1]: args.ir3_threshold, features[2]: args.ir4_threshold}
    labeled_df = unsupervised_filter_and_label(raw_df, features, thresh_map)
    
    analyzer = MTS(labeled_df, features, ir_thresholds=thresh_map)
    final_results = analyzer.detect_abnormal(md_quantile=args.md_quantile)

    # 4. Output
    print(f"Analysis complete. Found {final_results['is_abnormal'].sum()} outliers.")
    analyzer.plot_results(final_results, save_filename="MTS_Report.png")
