import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

def draw_2way(m_set, s_sensor, s_md, l1, l2, title, ax, small=False):
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal'); ax.axis('off')
    r = 0.6
    
    # Draw Circles
    ax.add_patch(plt.Circle((-0.3, 0), r, color='steelblue', alpha=0.4, ec='black', lw=1.5))
    ax.add_patch(plt.Circle((0.3, 0), r, color='darkorange', alpha=0.4, ec='black', lw=1.5))
    
    f_size = 10 if small else 13
    
    # 1. Left (Machine Only)
    m_only = len(m_set - s_sensor)
    ax.text(-0.6, 0, str(m_only), ha='center', va='center', fontweight='bold', fontsize=f_size)
    
    # 2. Middle (Overlap)
    overlap_count = len(m_set & s_sensor)
    ax.text(0, 0, str(overlap_count), ha='center', va='center', fontweight='bold', fontsize=f_size)
    
    # 3. Right (MTS Only)
    mts_only_sensor = len(s_sensor - m_set)
    mts_only_md = len(s_md - m_set)
    mts_total_text = str(mts_only_sensor + mts_only_md)
    
    ax.text(0.6, 0, mts_total_text, ha='center', va='center', fontweight='bold', fontsize=f_size)
    
    # Labels
    ax.text(-0.6, 0.75, l1, ha='center', va='center', fontsize=f_size, fontweight='bold')
    ax.text(0.6, 0.75, l2, ha='center', va='center', fontsize=f_size, fontweight='bold')
    ax.set_title(title, fontsize=f_size+1, fontweight='bold', pad=10)

def draw_3way_comparison(m_sets, s_sets, md_total_set, title, ax):
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect('equal'); ax.axis('off')
    centers = [(0, 0.55), (-0.5, -0.3), (0.5, -0.3)]
    colors = ['#ff9999','#66b3ff','#99ff99']
    for i, c in enumerate(centers):
        ax.add_patch(plt.Circle(c, 0.85, color=colors[i], alpha=0.3, ec='black', lw=1.2))
    
    ax.text(0, 1.5, 'IR2', fontsize=12, fontweight='bold', ha='center')
    ax.text(-1.3, -0.8, 'IR3', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.3, -0.8, 'IR4', fontsize=12, fontweight='bold', ha='center')
    
    m2, m3, m4 = m_sets['IR2'], m_sets['IR3'], m_sets['IR4']
    s2, s3, s4 = s_sets['IR2'], s_sets['IR3'], s_sets['IR4']
    
    m_reg = {'o2':m2-m3-m4, 'o3':m3-m2-m4, 'o4':m4-m2-m3, '23':(m2&m3)-m4, '24':(m2&m4)-m3, '34':(m3&m4)-m2, 'all':m2&m3&m4}
    s_reg = {'o2':s2-s3-s4, 'o3':s3-s2-s4, 'o4':s4-s2-s3, '23':(s2&s3)-s4, '24':(s2&s4)-s3, '34':(s3&s4)-s2, 'all':s2&s3&s4}
    
    coords = {'o2':(0,0.9), 'o3':(-0.8,-0.4), 'o4':(0.8,-0.4), '23':(-0.45,0.2), '24':(0.45,0.2), '34':(0,-0.6), 'all':(0,0)}
    for k, (x, y) in coords.items():
        ms, ss = m_reg[k], s_reg[k]
        txt = f"M:{len(ms-ss)}\nS:{len(ss-ms)}\nB:{len(ms&ss)}"
        ax.text(x, y, txt, ha='center', va='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.text(0, -1.3, f"MTS MD Only (No IR NG)\nTotal Samples: {len(md_total_set)}", 
            ha='center', va='center', fontsize=10, fontweight='bold', 
            bbox=dict(facecolor='#f0f0f0', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

def get_scen_set(df, ir_keys, exclusive, suffix):
    mask = pd.Series(True, index=df.index)
    for ir in ir_keys:
        col_list = [c for c in df.columns if ir in c and suffix in c]
        if col_list:
            mask &= (df[col_list[0]] == 'NG')
            
    if exclusive:
        others = [i for i in ['IR2', 'IR3', 'IR4'] if i not in ir_keys]
        for ir in others:
            col_list = [c for c in df.columns if ir in c and suffix in c]
            if col_list:
                mask &= (df[col_list[0]] == 'OK')
    return set(df[mask]['Unified_ID'])

def generate_reports(machine_file, mts_file):
    # 1. Load Data
    df_m = pd.read_csv(machine_file, encoding='utf-8-sig')
    df_s = pd.read_csv(mts_file, encoding='utf-8-sig')
    for d in [df_m, df_s]: d.columns = d.columns.str.replace('\ufeff', '', regex=False).str.strip()
    
    m_id, s_id = df_m.columns[0], df_s.columns[0]
    
    # Robustly detect columns containing IR2, IR3, IR4 or Status
    target_keywords = ['Status', 'IR2', 'IR3', 'IR4']
    ir_cols_s = [c for c in df_s.columns if any(k in c for k in target_keywords)]
    ir_cols_m = [c for c in df_m.columns if any(k in c for k in target_keywords)]

    # 2. Prepare Detailed Sets
    m_total_ng = set(df_m[df_m[ir_cols_m].eq('NG').any(axis=1)][m_id])
    s_sensor_ng = set(df_s[df_s[ir_cols_s].eq('NG').any(axis=1)][s_id])
    s_md_only = set(df_s[s_id]) - s_sensor_ng 

    def get_ir_sets(df, cols):
        res = {}
        for ir in ['IR2', 'IR3', 'IR4']:
            c_found = [col for col in cols if ir in col]
            if c_found:
                res[ir] = set(df[df[c_found[0]] == 'NG'][df.columns[0]])
            else:
                res[ir] = set()
        return res

    m_sets = get_ir_sets(df_m, ir_cols_m)
    s_sets = get_ir_sets(df_s, ir_cols_s)

    # 3. Join for scenarios
    df_j = pd.merge(df_m, df_s, left_on=m_id, right_on=s_id, how='outer', suffixes=('_m', '_s'))
    df_j['Unified_ID'] = df_j[m_id].fillna(df_j[s_id])
    
    status_cols = [c for c in df_j.columns if any(k in c for k in target_keywords) and c != 'Unified_ID']
    df_j[status_cols] = df_j[status_cols].fillna('OK')

    # 4. Generate Disagreement Table
    m_ng_mask = df_j[[c for c in df_j.columns if c.endswith('_m') and any(k in c for k in target_keywords)]].eq('NG').any(axis=1)
    s_ng_mask = df_j[[c for c in df_j.columns if c.endswith('_s') and any(k in c for k in target_keywords)]].eq('NG').any(axis=1)
    
    df_disagree = df_j[m_ng_mask != s_ng_mask].copy()
    df_disagree['Disagreement_Type'] = ''
    df_disagree.loc[m_ng_mask & ~s_ng_mask, 'Disagreement_Type'] = 'Machine NG / MTS OK'
    df_disagree.loc[~m_ng_mask & s_ng_mask, 'Disagreement_Type'] = 'Machine OK / MTS NG'
    
    cols_order = ['Unified_ID', 'Disagreement_Type'] + [c for c in df_disagree.columns if c not in ['Unified_ID', 'Disagreement_Type']]
    df_disagree[cols_order].to_csv('disagreement_details.csv', index=False, encoding='utf-8-sig')

    # 5. Summary Plots (Saved directly to current directory)
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(18, 8))
    draw_2way(m_total_ng, s_sensor_ng, s_md_only, "Machine", "MTS", "Total NG Comparison", ax1a)
    draw_3way_comparison(m_sets, s_sets, s_md_only, "Detailed IR Breakdown", ax1b)
    plt.tight_layout(pad=4.0)
    fig1.savefig('System_comparison.png', dpi=300)

    # 6. Detailed Breakdown Grid
    fig2, axes = plt.subplots(4, 3, figsize=(18, 20))
    scens = [(['IR2'],0,"With IR2 NG"),(['IR3'],0,"With IR3 NG"),(['IR4'],0,"With IR4 NG"),
             (['IR2'],1,"Only IR2 NG"),(['IR3'],1,"Only IR3 NG"),(['IR4'],1,"Only IR4 NG"),
             (['IR2','IR3'],0,"IR2 & IR3"),(['IR2','IR4'],0,"IR2 & IR4"),
             (['IR3','IR4'],0,"IR3 & IR4"),(['IR2','IR3','IR4'],0,"All IR NG")]
    
    for i, (k, e, lbl) in enumerate(scens):
        ax = axes[i // 3, i % 3]
        draw_2way(get_scen_set(df_j, k, e, '_m'), get_scen_set(df_j, k, e, '_s'), set(), "Machine", "MTS", lbl, ax, small=True)
    
    for j in range(len(scens), 12): axes[j // 3, j % 3].axis('off')
    fig2.savefig('Detailed_Breakdown_Grid.png', dpi=300)
    
    print("\n✅ Process Complete!")
    print("Files saved: disagreement_details.csv, System_comparison.png, Detailed_Breakdown_Grid.png")

    
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        generate_reports(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python NG_compare.py <machine_file> <mts_file>")
