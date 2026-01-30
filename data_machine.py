import pandas as pd
import sys
import os

def process_machine_data():
    if len(sys.argv) < 2:
        print("Usage: python data_machine.py <your_file.csv>")
        return

    input_file = sys.argv[1]
    
    try:
        # Load data
        df = pd.read_csv(input_file, sep=None, engine='python')
        df = df.dropna(how='all').reset_index(drop=True) 
        
        # 1. Add 'Serial No' (Crucial for the NG_compare matching)
        # We use the Sample number as the Serial No for the join
        if 'Serial No' not in df.columns:
            df.insert(0, 'Serial No', range(1, len(df) + 1)) 
            # Convert to string to ensure matching works later
            df['Serial No'] = df['Serial No'].astype(str)

        # 2. Map the columns
        if len(df.columns) >= 7:
            # We preserve the Serial No at index 0
            new_cols = ['Serial No', 'IR2_Val', 'Status_IR2', 'IR3_Val', 'Status_IR3', 'IR4_Val', 'Status_IR4']
            df.columns = new_cols[:len(df.columns)]
        
        # 3. Standardize Status Values
        status_cols = [c for c in df.columns if 'Status' in c]
        for col in status_cols:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].apply(lambda x: 'OK' if x in ['OK', 'NORMAL', 'PASS'] else 'NG')

        # 4. Filter for NG Rows (The comparison script expects only failures)
        mask2 = df['Status_IR2'] == 'NG' if 'Status_IR2' in df.columns else False
        mask3 = df['Status_IR3'] == 'NG' if 'Status_IR3' in df.columns else False
        mask4 = df['Status_IR4'] == 'NG' if 'Status_IR4' in df.columns else False
        
        df_ng = df[mask2 | mask3 | mask4].reset_index(drop=True)

        # 5. Save output with the EXACT name Streamlit is looking for
        # We use ONLY the filename, no paths, to stay in the root folder
        base_filename = os.path.basename(input_file)
        output_name = f"Machine_NG_ONLY_{base_filename}"
        
        df_ng.to_csv(output_name, index=False, encoding='utf-8-sig')
        
        print(f"✅ Machine NG list created: {output_name}")

    except Exception as e:
        print(f"❌ Error in data_machine.py: {e}")
        sys.exit(1) # Tell Streamlit something went wrong

if __name__ == "__main__":
    process_machine_data()
