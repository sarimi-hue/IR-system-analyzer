import pandas as pd
import sys
import os

def process_mts_data():
    if len(sys.argv) < 2:
        print("Usage: python data_MTS.py <your_file.csv>")
        return

    input_file = sys.argv[1]
    base_name = os.path.basename(input_file)
    
    try:
        # 1. Load the data
        df = pd.read_csv(input_file, sep=None, engine='python')
        df = df.dropna(how='all').reset_index(drop=True) 

        # 2. Add 'Serial No' if it doesn't exist (needed for the final join)
        if 'Serial No' not in df.columns:
            df.insert(0, 'Serial No', range(1, len(df) + 1))
        
        df['Serial No'] = df['Serial No'].astype(str).str.strip()

        # 3. Identify and Rename Columns for the Math script
        new_cols = list(df.columns)
        for i in range(len(new_cols)):
            if "Status" in str(new_cols[i]):
                # Looks at the column to the left to name the status (e.g., Status_IR2)
                prev_col = str(new_cols[i-1]).strip() if i > 0 else "Unknown"
                new_cols[i] = f"Status_{prev_col}"
        
        df.columns = new_cols

        # --- IMPORTANT: Keep actual data! ---
        # We don't force them to "Normal" here because IR_test12 needs to see the real values.

        # 4. Save the "Normalised" file (Main input for IR_test12.py)
        normalised_name = f"Normalised_{base_name}"
        df.to_csv(normalised_name, index=False)
        print(f"✅ Normalised file saved: {normalised_name}")

        # 5. Create the "MTS_NG" list for the final Venn diagram comparison
        # We define NG as anything that isn't 'OK' or 'NORMAL' in the status columns
        status_cols = [c for c in df.columns if c.startswith("Status_")]
        
        # Filter for rows where any status column contains an NG value
        # Adjust the ['OK', 'NORMAL'] list if your MTS machine uses different 'Pass' text
        is_ng = df[status_cols].apply(lambda x: ~x.astype(str).str.upper().isin(['OK', 'NORMAL', 'PASS'])).any(axis=1)
        df_ng = df[is_ng].reset_index(drop=True)

        # 6. Save the NG-only file for NG_compare.py
        mts_ng_name = f"MTS_NG_Normalised_{base_name}"
        df_ng.to_csv(mts_ng_name, index=False)
        print(f"✅ MTS NG list saved: {mts_ng_name}")

    except Exception as e:
        print(f"❌ Error in data_MTS.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    process_mts_data()
