import streamlit as st
import os
import subprocess
import sys
import pandas as pd
import glob

# Set page title
st.set_page_config(page_title="IR System Analyzer", layout="wide")
st.title("🔬 IR System Statistical Analyzer")

# 1. Sidebar for Parameters
st.sidebar.header("Settings")
ir2_val = st.sidebar.text_input("IR2 Threshold", "50")
ir3_val = st.sidebar.text_input("IR3 Threshold", "50")
ir4_val = st.sidebar.text_input("IR4 Threshold", "50")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Raw CSV Data File", type=["csv"])

if uploaded_file is not None:
    # Save the uploaded file locally so scripts can find it
    input_filename = uploaded_file.name
    with open(input_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Loaded: {input_filename}")

    if st.button("🚀 Run Full Analysis"):
        try:
            # We use the same logic as your main.py
            base_no_ext = os.path.splitext(input_filename)[0]
            output_folder = f"sorting analysis result_{base_no_ext}"
            
            # Step 1: Machine Logic
            with st.status("Processing Data...", expanded=True) as status:
                st.write("Extracting Machine Failures...")
                subprocess.run([sys.executable, "data_machine.py", input_filename], check=True)
                
                st.write("Normalizing Electrical Data...")
                subprocess.run([sys.executable, "data_MTS.py", input_filename], check=True)
                
                st.write("Running Statistical Analysis (MD)...")
                normalised_out = f"Normalised_{input_filename}"
                subprocess.run([
                    sys.executable, "IR_test12.py", 
                    "--file", normalised_out, 
                    "--ir2_threshold", ir2_val, 
                    "--ir3_threshold", ir3_val, 
                    "--ir4_threshold", ir4_val
                ], check=True)

                st.write("Generating Venn Diagrams...")
                mts_ng_out = f"MTS_NG_Normalised_{input_filename}"
                machine_out = f"Machine_NG_ONLY_{input_filename}"
                
                # Note: We call your harmonization logic here if needed
                # (You can paste your prepare_files_for_comparison function here)
                
                subprocess.run([sys.executable, "NG_compare.py", mts_ng_out, machine_out, output_folder], check=True)
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # 3. Display Results
            st.header("📊 Analysis Results")
            
            # Find the Venn Diagram image in the output folder
            venn_images = glob.glob(f"{output_folder}/*.png")
            if venn_images:
                cols = st.columns(len(venn_images))
                for idx, img_path in enumerate(venn_images):
                    cols[idx].image(img_path, caption=os.path.basename(img_path))

            # Provide Download for the Result Folder
            st.info(f"All files saved in: {output_folder}")
            
        except Exception as e:
            st.error(f"Error during processing: {e}")
