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
        
        # 1. Add Sample no. FIRST
        df.insert(0, 'Sample no.', range(1, len(df) + 1)) 

        # 2. FIXED HEADER MAPPING
        # We manually map the columns to ensure IR2, IR3, and IR4 are preserved
        # This assumes your raw machine CSV has: [Val1, Stat1, Val2, Stat2, Val3, Stat3]
        # after we added the 'Sample no.' at index 0.
        
        if len(df.columns) >= 7:
            new_cols = ['Sample no.', 'IR2_Val', 'Status_IR2', 'IR3_Val', 'Status_IR3', 'IR4_Val', 'Status_IR4']
            # Only apply if column count matches expectations
            df.columns = new_cols[:len(df.columns)]
        else:
            # Fallback for unexpected formats
            print("!! Warning: Unexpected column count. Using dynamic naming.")
            new_cols = list(df.columns)
            for i in range(1, len(new_cols)):
                if "Status" in str(new_cols[i]) or i % 2 == 0:
                    new_cols[i] = f"Status_IR{(i+1)//2}"
                else:
                    new_cols[i] = f"IR{(i+1)//2}_Val"
            df.columns = new_cols

        # 3. Standardize Status Values
        status_cols = [c for c in df.columns if 'Status' in c]
        for col in status_cols:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].apply(lambda x: 'OK' if x in ['OK', 'NORMAL', 'PASS'] else 'NG')

        # 4. Identify NG Rows
        # We check the status columns for 'NG'
        mask2 = df['Status_IR2'] == 'NG' if 'Status_IR2' in df.columns else False
        mask3 = df['Status_IR3'] == 'NG' if 'Status_IR3' in df.columns else False
        mask4 = df['Status_IR4'] == 'NG' if 'Status_IR4' in df.columns else False
        
        any_ng_mask = (mask2 | mask3 | mask4)

        # 5. KEEP ALL ROWS for the comparison script, but mark the output
        # If you strictly want ONLY NG rows, uncomment the next line:
        # df = df[any_ng_mask].reset_index(drop=True)

        # 6. Save output
        output_name = f"Machine_NG_{os.path.basename(input_file)}"
        df.to_csv(output_name, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*40)
        print(f"✅ Processed: {os.path.basename(input_file)}")
        print(f"Headers: {list(df.columns)}")
        print(f"File saved as: {output_name}")
        print("="*40)

    except Exception as e:
        print(f"❌ Error in data_machine.py: {e}")

if __name__ == "__main__":
    process_machine_data()
