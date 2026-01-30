import pandas as pd
import sys
import os

def process_machine_data():
    if len(sys.argv) < 2:
        print("Usage: python NG_table_fix.py <your_file.csv>")
        return

    input_file = sys.argv[1]
    
    try:
        # 1. Load the data
        df = pd.read_csv(input_file, sep=None, engine='python')

        # --- UPDATED: Delete only truly empty rows at the end ---
        # This removes rows where EVERY column is empty/NaN
        df = df.dropna(how='all').reset_index(drop=True) 
        
        # Alternatively, if "empty" means they contain only whitespace or empty strings:
        # df = df[df.astype(str).replace(r'^\s*$', None, regex=True).notna().any(axis=1)]

        # 2. Identify and Rename Columns
        new_cols = list(df.columns)
        for i in range(len(new_cols)):
            if "Status" in str(new_cols[i]):
                prev_col = str(new_cols[i-1]).strip() if i > 0 else "Unknown"
                new_cols[i] = f"Status_{prev_col}"
        
        df.columns = new_cols

        # 3. Update the values in Status columns
        for col in df.columns:
                    if col.startswith("Status_"):
                        # This line now sets every value in these columns to "Normal"
                        df[col] = "Normal"
        # 4. Calculation Logic
        def get_mask(search_term):
            for c in df.columns:
                if search_term.upper() in c.upper() and "STATUS" in c.upper():
                    return df[c] == 'NG'
            return pd.Series([False] * len(df))

        mask2 = get_mask("IR2")
        mask3 = get_mask("IR3")
        mask4 = get_mask("IR4")

        # Calculations
        a1, a2, a3 = mask2.sum(), mask3.sum(), mask4.sum()
        b1 = (mask2 & mask3).sum()
        b2 = (mask2 & mask4).sum()
        b3 = (mask3 & mask4).sum()
        c = (mask2 & mask3 & mask4).sum()
        total_ng_mask = (mask2 | mask3 | mask4)
        total_ng_count = total_ng_mask.sum()

        # Print results
        print("="*35)
        print("📊 TERMINAL NG SUMMARY REPORT")
        print("-" * 35)
        print(f"A1 (IR2 NG): {a1}")
        print(f"A2 (IR3 NG): {a2}")
        print(f"A3 (IR4 NG): {a3}")
        print("-" * 35)
        print(f"TOTAL UNIQUE NG: {total_ng_count}")
        print("="*35)

        # 5. Save output
        output_name = f"Normalised_{os.path.basename(input_file)}"
        df.to_csv(output_name, index=False)
        print(f"✅ Processed file saved as: {output_name}")

    import sys
# ... at the very end of the script ...
input_filename = sys.argv[1]
df.to_csv(f"MTS_NG_Normalised_{input_filename}", index=False)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    process_machine_data()
