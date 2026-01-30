import streamlit as st
import os
import subprocess
import sys
import pandas as pd
import glob

# Helper to clean up headers so serial numbers match perfectly for merging
def prepare_files_for_comparison(mts_file, machine_file):
    try:
        for file in [mts_file, machine_file]:
            if os.path.exists(file):
                df = pd.read_csv(file)
                if 'Serial No' in df.columns:
                    df['Serial No'] = df['Serial No'].astype(str).str.strip()
                    df.to_csv(file, index=False)
    except Exception as e:
        st.warning(f"Note on data cleaning: {e}")

# Page Setup
st.set_page_config(page_title="IR System Analyzer", layout="wide")
st.title("🔬 IR System Statistical Analyzer")
st.write("Upload your raw data to run the automated sorting analysis.")

# 1. Sidebar Settings
st.sidebar.header("Analysis Settings")
st.sidebar.write("Adjust pass/fail thresholds:")
ir2_val = st.sidebar.text_input("IR2 Limit", "50")
ir3_val = st.sidebar.text_input("IR3 Limit", "50")
ir4_val = st.sidebar.text_input("IR4 Limit", "50")

# 2. File Uploader
uploaded_file = st.file_uploader("Drop your raw CSV file here", type=["csv"])

if uploaded_file is not None:
    input_filename = uploaded_file.name
    with open(input_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File ready: {input_filename}")

    # 3. Execution Button
    if st.button("🚀 Start Analysis"):
        try:
            base_name = os.path.splitext(input_filename)[0]
            output_folder = f"results_{base_name}"
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # These names must match what your sub-scripts save
            machine_out = f"Machine_NG_ONLY_{input_filename}"
            normalised_out = f"Normalised_{input_filename}"
            mts_ng_out = f"MTS_NG_Normalised_{input_filename}"

            with st.status("Processing...", expanded=True) as status:
                
                # --- TASK 1 & 2: DATA PREP ---
                st.write("🔍 Identifying machine-related failures...")
                subprocess.run([sys.executable, "data_machine.py", input_filename], check=True)
                
                st.write("⚖️ Normalizing electrical measurements...")
                subprocess.run([sys.executable, "data_MTS.py", input_filename], check=True)
                
                # --- TASK 3: STATISTICAL ANALYSIS (IR_test12.py) ---
                if os.path.exists(normalised_out):
                    st.write("📈 Calculating Statistics (Mahalanobis Distance)...")
                    # We capture the output here to catch hidden errors
                    result_ir = subprocess.run([
                        sys.executable, "IR_test12.py", 
                        "--file", normalised_out, 
                        "--ir2_threshold", ir2_val, 
                        "--ir3_threshold", ir3_val, 
                        "--ir4_threshold", ir4_val
                    ], capture_output=True, text=True)
                    
                    if result_ir.returncode != 0:
                        st.error("Error in IR_test12.py (Statistical Analysis)")
                        st.code(result_ir.stderr)
                        st.stop()
                else:
                    st.error(f"Error: {normalised_out} was not found.")
                    st.stop()

                # --- DEBUGGING STEP ---
                st.write("🔎 Debugging: Verifying intermediate files...")
                current_files = os.listdir(".")
                
                # --- TASK 4: STANDARDIZE ---
                st.write("🧹 Standardizing headers for comparison...")
                if machine_out in current_files and mts_ng_out in current_files:
                    prepare_files_for_comparison(mts_ng_out, machine_out)
                else:
                    st.error("Comparison failed: Missing intermediate files.")
                    st.write(f"Looking for: **{machine_out}** and **{mts_ng_out}**")
                    st.stop()

                # --- TASK 5: VENN DIAGRAMS ---
                st.write("🎨 Generating Venn diagrams...")
                result_comp = subprocess.run([
                    sys.executable, "NG_compare.py", 
                    mts_ng_out, machine_out, output_folder
                ], capture_output=True, text=True)
                
                if result_comp.returncode != 0:
                    st.error("Error in NG_compare.py (Comparison Step)")
                    st.code(result_comp.stderr)
                    st.stop()
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # 4. Results Display
            st.header("📊 Final Comparison Results")
            images = glob.glob(f"{output_folder}/*.png")
            if images:
                cols = st.columns(len(images))
                for i, img_path in enumerate(images):
                    cols[i].image(img_path, caption=os.path.basename(img_path))
            else:
                st.warning("No Venn diagrams were found in the results folder.")

            st.info(f"Results are stored in: {output_folder}")
            
        except subprocess.CalledProcessError as e:
            st.error("A background script failed completely.")
            st.code(f"Technical Detail: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
