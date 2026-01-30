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
            # 1. Setup Paths
            base_no_ext = os.path.splitext(input_filename)[0]
            output_folder = f"sorting_analysis_result_{base_no_ext}" # Removed spaces for safety
            
            # CRITICAL: Create the folder FIRST so the scripts have a place to save
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 2. Define filenames (Ensuring they match what your scripts produce)
            machine_out = f"Machine_NG_ONLY_{input_filename}"
            normalised_out = f"Normalised_{input_filename}"
            mts_ng_out = f"MTS_NG_Normalised_{input_filename}"

            with st.status("Processing Data...", expanded=True) as status:
                st.write("Extracting Machine Failures...")
                subprocess.run([sys.executable, "data_machine.py", input_filename], check=True)
                
                st.write("Normalizing Electrical Data...")
                subprocess.run([sys.executable, "data_MTS.py", input_filename], check=True)
                
                st.write("Running Statistical Analysis...")
                subprocess.run([
                    sys.executable, "IR_test12.py", 
                    "--file", normalised_out, 
                    "--ir2_threshold", ir2_val, 
                    "--ir3_threshold", ir3_val, 
                    "--ir4_threshold", ir4_val
                ], check=True)

                # IMPORTANT: Run the header fix logic right here in streamlit_app.py 
                # to ensure files are ready for NG_compare
                st.write("Standardizing Headers...")
                prepare_files_for_comparison(mts_ng_out, machine_out)

                st.write("Generating Final Comparison...")
                # Pass the output_folder explicitly
                subprocess.run([sys.executable, "NG_compare.py", mts_ng_out, machine_out, output_folder], check=True)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # 3. Display Download Button for the result
            st.success("Success! You can now view the results below.")
            
        except subprocess.CalledProcessError as e:
            st.error(f"Script Error: {e}")
            # This will show you the ACTUAL error from inside the sub-script
            st.code(e.stderr if e.stderr else "Check logs for detail")

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
