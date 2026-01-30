import os
import subprocess
import argparse
import sys
import pandas as pd

# Use the current python executable (handles both script and .exe mode)
python_exe = sys.executable

def prepare_files_for_comparison(mts_file, machine_file):
    if not os.path.exists(mts_file) or not os.path.exists(machine_file):
        print(f"!! Error: Missing files for harmonization.")
        return

    def robust_clean(df, label):
        id_col = df.columns[0]
        status_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                unique_vals = df[col].dropna().unique()
                if any(v in ['OK', 'NG'] for v in unique_vals):
                    status_cols.append(col)
        
        new_names = {id_col: 'Sample no.'}
        for i, col in enumerate(status_cols[:3]):
            new_names[col] = f'IR{i+2}'
            
        df_clean = df[[id_col] + status_cols[:3]].rename(columns=new_names)
        df_clean['Sample no.'] = df_clean['Sample no.'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return df_clean

    try:
        df_mts = pd.read_csv(mts_file, encoding='utf-8-sig')
        df_mac = pd.read_csv(machine_file, encoding='utf-8-sig')
        df_mts_final = robust_clean(df_mts, "MTS")
        df_mac_final = robust_clean(df_mac, "Machine")
        df_mts_final.to_csv(mts_file, index=False, encoding='utf-8-sig')
        df_mac_final.to_csv(machine_file, index=False, encoding='utf-8-sig')
        print(f">> Harmonization successful.")
    except Exception as e:
        print(f"!! Harmonization failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="MTS Pipeline Master Script")
    parser.add_argument("--file", required=True, help="Input raw CSV file")
    parser.add_argument("--ir2_threshold", type=str, required=True)
    parser.add_argument("--ir3_threshold", type=str, required=True)
    parser.add_argument("--ir4_threshold", type=str, required=True)
    args = parser.parse_args()

    input_path = os.path.abspath(args.file)
    raw_filename = os.path.basename(input_path)
    base_no_ext = os.path.splitext(raw_filename)[0]
    output_folder = f"sorting analysis result_{base_no_ext}"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    machine_out = f"Machine_NG_ONLY_{raw_filename}"
    normalised_out = f"Normalised_{raw_filename}"
    mts_ng_out = f"MTS_NG_Normalised_{raw_filename}" 

    print(f"\n{'='*50}\nSTARTING IR ANALYSIS PIPELINE\n{'='*50}")

    try:
        print(f">> Step 1: Filtering Machine Data...")
        subprocess.run([python_exe, "data_machine.py", input_path], check=True)

        print(f">> Step 2.1: Normalizing MTS Data...")
        subprocess.run([python_exe, "data_MTS.py", input_path], check=True)

        print(f">> Step 2.2: Running IR Test Logic (test12)...")
        subprocess.run([
            python_exe, "IR_test12.py", 
            "--file", normalised_out, 
            "--ir2_threshold", args.ir2_threshold, 
            "--ir3_threshold", args.ir3_threshold, 
            "--ir4_threshold", args.ir4_threshold
        ], check=True)

        print(f">> Step 2.3: Harmonizing Headers...")
        prepare_files_for_comparison(mts_ng_out, machine_out)

        print(f">> Step 3: Generating Venn Diagrams...")
        subprocess.run([python_exe, "NG_compare.py", mts_ng_out, machine_out, output_folder], check=True)

        print(f"\n{'='*50}\nSUCCESS: All Reports Generated!\n{'='*50}")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed at command: {' '.join(e.cmd)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
