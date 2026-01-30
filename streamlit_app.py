import streamlit as st
import os
import subprocess
import sys
import pandas as pd
import glob

# Helper to clean up the data headers before comparing
def prepare_files_for_comparison(mts_file, machine_file):
    try:
        for file in [mts_file, machine_file]:
            if os.path.exists(file):
                df = pd.read_csv(file)
                # We make sure 'Serial No' is a clean string so the scripts can match them
                if 'Serial No' in df.columns:
                    df['Serial No'] = df['Serial No'].astype(str).str.strip()
                    df.to_csv(file, index=False)
    except Exception as e:
        st.warning(f"Note on data cleaning: {e}")

# Page Setup
st.set_page_config(page_title="IR System Analyzer", layout="wide")
st.title("🔬 IR System Statistical Analyzer")
st.write("Upload your data below to begin the automated sorting analysis.")

# 1. Sidebar for User Settings
st.sidebar.header("Analysis Settings")
st.sidebar.write("Set your pass/fail thresholds here:")
ir2_val = st.sidebar.text_input("IR2 Limit", "50")
ir3_val = st.sidebar.text_input("IR3 Limit", "50")
ir4_val = st.sidebar.text_input("IR4 Limit", "50")

# 2. File Uploader
uploaded_file = st.file_uploader("Drop your raw CSV file here", type=["csv"])

if uploaded_file is not None:
    input_filename = uploaded_file.name
    # Temporarily save the file so our sub-scripts can read it
    with open(input_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Successfully loaded: {input_filename}")

    # 3. Action Button
    if st.button("🚀 Start Analysis"):
        try:
            # Setting up the workspace
            base_name = os.path.splitext(input_filename)[0]
            output_folder = f"results_{base_name}"
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Defining our intermediate files
            machine_out = f"Machine_NG_ONLY_{input_filename}"
            normalised_out = f"Normalised_{input_filename}"
            mts_ng_out = f"MTS_NG_Normalised_{input_filename}"

            # Running the engine
            with st.status("Working on your data...", expanded=True) as status:
                st.write("🔍 Identifying machine-related failures...")
                subprocess.run([sys.executable, "data_machine.py", input_filename], check=True)
                
                st.write("⚖️ Normalizing electrical measurements...")
                subprocess.run([sys.executable, "data_MTS.py", input_filename], check=True)
                
                st.write("📈 Calculating Mahalanobis Distance and Statistics...")
                subprocess.run([
                    sys.executable, "IR_test12.py", 
                    "--file", normalised_out, 
                    "--ir2_threshold", ir2_val, 
                    "--ir3_threshold", ir3_val, 
                    "--ir4_threshold", ir4_val
                ], check=True)

                st.write("Cleaning up headers for final comparison...")
                prepare_files_for_comparison(mts_ng_out, machine_out)

                st.write("Generating Venn diagrams...")
                subprocess.run([sys.executable, "NG_compare.py", mts_ng_out, machine_out, output_folder], check=True)
                
                status.update(label="All finished!", state="complete", expanded=False)

            # 4. Show the Results
            st.header("Final Comparison Results")
            
            # Look for any images generated in the results folder
            images = glob.glob(f"{output_folder}/*.png")
            if images:
                cols = st.columns(len(images))
                for i, img_path in enumerate(images):
                    cols[i].image(img_path, caption=os.path.basename(img_path))
            else:
                st.warning("No charts were generated. This usually means no matching failures were found between the machine and the electrical test.")

            st.info(f"You can find all processed files in the folder: {output_folder}")
            
        except subprocess.CalledProcessError as e:
            st.error("One of the background scripts ran into an issue.")
            st.code(f"Technical detail: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
