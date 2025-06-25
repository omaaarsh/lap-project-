import os
import re
import io
import zipfile
import shutil
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from functools import reduce
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import gseapy as gp
from gseapy.plot import barplot

# === 📁 FOLDERS ===
OUTPUT_FOLDER = "output_zips"
ANALYSIS_FOLDER = "drug_Analysis_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

# === 📊 STREAMLIT UI ===
st.set_page_config(page_title="Drug Gene Expression Analyzer", layout="centered", page_icon="🧬")
st.title("🧬 Drug Gene Expression Analyzer")

compound = st.text_input("Enter compound name (e.g. vorinostat):", "vorinostat")
top_genes = st.number_input("Number of top genes to fetch", min_value=100, max_value=5000, value=1000, step=100)

st.markdown("### ⚙️ Analysis Parameters")
log2fc_threshold = st.number_input("Log2 Fold Change Threshold (log2FC)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
consistency_threshold = st.slider("Minimum Consistency Across Cell Lines (%)", 1, 100, 50)
pvalue_threshold = st.number_input("Significance p-value Threshold", min_value=0.001, max_value=0.1, value=0.05, step=0.001)

# === 🔧 UTILITY FUNCTIONS ===
def extract_concentration(d):
    conc_str = d.get("concentration", "").lower()
    match = re.match(r"(\d+(?:\.\d+)?)\s*u[mM]", conc_str)
    return float(match.group(1)) if match else -1

def get_signature_metadata(compound_name):
    url = f"https://www.ilincs.org/api/SignatureMeta/findTermWithSynonyms?term={compound_name}&library=LIB_5&limit=100"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()["data"]
        tissue_groups = {}
        for d in data:
            tissue = d.get("tissue")
            if tissue:
                tissue_groups.setdefault(tissue, []).append(d)
        selected_entries = []
        for entries in tissue_groups.values():
            ten_um = next((d for d in entries if d.get("concentration") == "10uM"), None)
            selected_entries.append(ten_um if ten_um else max(entries, key=extract_concentration))
        return [
            {
                "SignatureId": d["signatureid"],
                "Perturbagen": d["compound"],
                "Tissue": d["tissue"],
                "CellLine": d["cellline"]
            }
            for d in selected_entries if d
        ]
    except Exception as e:
        st.error(f"❌ Failed to get metadata: {e}")
        return []

def get_signature_data(sig_ids, top_genes=1000):
    url = "https://www.ilincs.org/api/ilincsR/downloadSignature"
    payload = {
        "sigID": ",".join(sig_ids),
        "noOfTopGenes": top_genes,
        "display": True
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        return response.json().get("data", {}).get("signature", [])
    except Exception as e:
        st.error(f"❌ Failed to get signature data: {e}")
        return []

def save_gene_data_to_zip(metadata, gene_df, output_zip_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a") as zipf:
        for row in metadata:
            sig_id = row['SignatureId']
            tissue = (row['Tissue'] or "UnknownTissue").replace(" ", "_")
            cell_line = (row['CellLine'] or "UnknownCell").replace(" ", "_")
            perturbagen = (row['Perturbagen'] or "Unknown").replace(" ", "_")
            matched = gene_df[gene_df['signatureID'] == sig_id]
            if matched.empty:
                continue
            filename = f"{perturbagen}_{tissue}_{cell_line}.csv"
            print(filename)
            zipf.writestr(filename, matched.to_csv(index=False).encode('utf-8'))
    with open(output_zip_path, "wb") as f:
        f.write(zip_buffer.getvalue())

def run_enrichment(gene_index, label, drug_name, output_folder):
    if isinstance(gene_index, pd.MultiIndex) and gene_index.empty:
        st.warning(f"⚠️ No genes for enrichment in {label}")
        return False

    symbols = [g[1] for g in gene_index]
    try:
        enr = gp.enrichr(
            gene_list=symbols,
            gene_sets=["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"],
            organism='Human',
            outdir=None,
            cutoff=0.05
        )
        if enr.results.empty:
            return False

        enr.results.to_csv(os.path.join(output_folder, f"{label.lower()}_enrichment.csv"), index=False)
        fig, ax = plt.subplots(figsize=(100, 30))
        barplot(enr.results.sort_values('Adjusted P-value').head(10), title=f"{drug_name} - {label}", ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{label.lower()}_enrichment.png"))
        plt.close()
        return True
    except Exception as e:
        st.warning(f"⚠️ Enrichment error for {label}: {e}")
        return False

def analyze_zip(zip_path, drug_name):
    cell_line_data = {}
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for csv_name in zipf.namelist():
            if csv_name.endswith(".csv"):
                with zipf.open(csv_name) as f:
                    df = pd.read_csv(f)
                    df = df[df['Significance_pvalue'] <= pvalue_threshold]
                    cell_line = csv_name.split(" - ")[0].replace(".csv", "").strip()
                    cell_line_data[cell_line] = df

    if not cell_line_data:
        print(f"{drug_name} ❌ No valid cell line data")
        return False
    gene_dfs = []
    for cell_line, df in cell_line_data.items():
        temp = df[['ID_geneid', 'Name_GeneSymbol', 'Value_LogDiffExp']].copy()
        temp = temp.rename(columns={'Value_LogDiffExp': cell_line})
        gene_dfs.append(temp)

    merged_df = reduce(lambda l, r: pd.merge(l, r, on=['ID_geneid', 'Name_GeneSymbol'], how='outer'), gene_dfs)
    merged_df.set_index(['ID_geneid', 'Name_GeneSymbol'], inplace=True)

    cell_line_count = len(cell_line_data)
    min_consistent = int(cell_line_count * (consistency_threshold / 100))
    up_mask = (merged_df > log2fc_threshold).sum(axis=1)
    down_mask = (merged_df < -log2fc_threshold).sum(axis=1)

    consistently_up = up_mask[up_mask >= min_consistent].index
    consistently_down = down_mask[down_mask >= min_consistent].index

    consistent_genes = consistently_up.union(consistently_down)
    heatmap_data = merged_df.loc[consistent_genes].copy()
    heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0).fillna(0)
    heatmap_data = heatmap_data.loc[:, heatmap_data.std() > 0]

    heatmap_ok = heatmap_data.shape[0] >= 2 and heatmap_data.shape[1] >= 2
    up_ok = len(consistently_up) > 0
    down_ok = len(consistently_down) > 0

    if heatmap_ok and up_ok and down_ok:
        drug_folder = os.path.join(ANALYSIS_FOLDER, drug_name)
        os.makedirs(drug_folder, exist_ok=True)

        # Save heatmap
        # Hierarchical clustering
        row_linkage = linkage(pdist(heatmap_data, metric='correlation'), method='average')
        col_linkage = linkage(pdist(heatmap_data.T, metric='correlation'), method='average')
    
        # Simple but polished heatmap
        sns.set(font_scale=0.8)  # Smaller font for better fit
        g = sns.clustermap(
            heatmap_data,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            cmap='vlag',            # Simple diverging color palette (blue-white-red)
            center=0,               # Centered around 0 for log2FC
            linewidths=0.5,
            figsize=(15, 15)
        )
    
        # Add title above the whole figure
        plt.suptitle(f"{drug_name} - Clustering Heatmap", fontsize=14, y=1.02)
    
        # Save clean figure
        g.savefig(os.path.join(drug_folder, f"{drug_name}_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()


        up_success = run_enrichment(consistently_up, "Upregulated", drug_name, drug_folder)
        down_success = run_enrichment(consistently_down, "Downregulated", drug_name, drug_folder)

        if not up_success or not down_success:
            shutil.rmtree(drug_folder)
            return False
        return True
    else:
        return False

def zip_analysis_folder(drug_name):
    folder_path = os.path.join(ANALYSIS_FOLDER, drug_name)
    if not os.path.exists(folder_path):
        return None
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

# === 🚀 RUN ANALYSIS ===
if st.button("Run Analysis"):
    metadata = get_signature_metadata(compound)
    if not metadata:
        st.error(f"❌ No metadata found for {compound}")
    else:
        sig_ids = [m["SignatureId"] for m in metadata]
        gene_data = get_signature_data(sig_ids, top_genes)
        if not gene_data:
            st.error(f"❌ No gene data for {compound}")
        else:
            gene_df = pd.DataFrame(gene_data)
            zip_path = os.path.join(OUTPUT_FOLDER, f"{compound.replace(' ', '_')}.zip")
            save_gene_data_to_zip(metadata, gene_df, zip_path)
            success = analyze_zip(zip_path, compound)

            if success:
                st.success(f"{compound} ✅ Analysis complete")
                zip_bytes = zip_analysis_folder(compound)
                if zip_bytes:
                    st.download_button(
                        label="📦 Download Full Analysis Folder",
                        data=zip_bytes,
                        file_name=f"{compound}_analysis.zip",
                        mime="application/zip"
                    )
            else:
                st.error(f"{compound} ❌ Analysis failed")
