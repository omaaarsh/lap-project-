import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import gseapy as gp
from gseapy.plot import barplot
import zipfile
import io

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🧬 Consistent Gene Signature Analysis from Drug Treatment (ZIP Mode)")
st.markdown("""
Upload a **main ZIP file**, where each **sub-ZIP** represents one **drug**, and contains **CSV files for cell lines** treated with that drug.

🔹 Each inner file name should follow: `DrugName cell_line_name.xls - DrugName cell_line_name.xls.csv`  
🔹 Each CSV must contain: `ID_geneid`, `Name_GeneSymbol`, `Value_LogDiffExp`, `Significance_pvalue`
""")

# --- USER INPUT ---
log2fc_threshold = st.sidebar.slider("log₂ Fold Change Threshold", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
consistency_threshold = st.sidebar.slider("Cell Line Consistency Threshold (%)", min_value=30, max_value=100, value=50, step=10)
pvalue_threshold = st.sidebar.slider("Significance P-Value Threshold", min_value=0.0, max_value=0.2, value=0.2, step=0.01)

# --- HELPER: Extract nested ZIPs ---
def extract_nested_zip_files(zip_file) -> dict:
    drug_data = {}
    with zipfile.ZipFile(zip_file) as main_zip:
        for inner_name in main_zip.namelist():
            if inner_name.endswith('.zip'):
                try:
                    with main_zip.open(inner_name) as nested_zip_file:
                        nested_zip_bytes = io.BytesIO(nested_zip_file.read())
                        with zipfile.ZipFile(nested_zip_bytes) as nested_zip:
                            drug_name = os.path.splitext(os.path.basename(inner_name))[0]
                            drug_data[drug_name] = []
                            for file_name in nested_zip.namelist():
                                if file_name.endswith(".csv"):
                                    with nested_zip.open(file_name) as csv_file:
                                        file_bytes = io.BytesIO(csv_file.read())
                                        drug_data[drug_name].append((file_name, file_bytes))
                except zipfile.BadZipFile:
                    st.warning(f"⚠️ Skipped invalid ZIP file: {inner_name}")
    return drug_data

# --- UPLOAD MAIN ZIP ---
uploaded_zip = st.file_uploader("Upload main ZIP file with sub-ZIPs (one per drug):", type="zip")

if uploaded_zip:
    drug_files_map = extract_nested_zip_files(uploaded_zip)
    
    if not drug_files_map:
        st.error("No sub-ZIP files with CSVs found.")
    else:
        selected_drug = st.selectbox("Choose a drug to analyze:", list(drug_files_map.keys()))
        selected_files = drug_files_map[selected_drug]
        
        cell_line_data = {}
        for file_name, file_bytes in selected_files:
            try:
                df = pd.read_csv(file_bytes)
                if " - " in file_name:
                    cell_line = file_name.split(" - ")[0].replace(".xls", "").strip()
                else:
                    cell_line = os.path.splitext(file_name)[0]
                df = df[df['Significance_pvalue'] <= pvalue_threshold]
                cell_line_data[cell_line] = df
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")

        if not cell_line_data:
            st.warning("No valid CSV data was loaded.")
        else:
            st.success(f"Loaded {len(cell_line_data)} cell lines for drug: {selected_drug}")
            st.write(list(cell_line_data.keys()))

            # --- MERGE ---
            gene_dfs = []
            for cell_line, df in cell_line_data.items():
                temp = df[['ID_geneid', 'Name_GeneSymbol', 'Value_LogDiffExp']].copy()
                temp = temp.rename(columns={'Value_LogDiffExp': cell_line})
                gene_dfs.append(temp)

            merged_df = reduce(lambda left, right: pd.merge(left, right, on=['ID_geneid', 'Name_GeneSymbol'], how='outer'), gene_dfs)
            merged_df.set_index(['ID_geneid', 'Name_GeneSymbol'], inplace=True)

            # --- FILTER FOR CONSISTENCY ---
            cell_line_count = len(cell_line_data)
            min_consistent = int(cell_line_count * (consistency_threshold / 100))

            upregulated_mask = (merged_df > log2fc_threshold).sum(axis=1)
            downregulated_mask = (merged_df < -log2fc_threshold).sum(axis=1)

            consistently_up_genes = upregulated_mask[upregulated_mask >= min_consistent].index
            consistently_down_genes = downregulated_mask[downregulated_mask >= min_consistent].index

            st.subheader("📊 Summary")
            st.write(f"**Total genes in matrix:** {merged_df.shape[0]}")
            st.write(f"**Genes upregulated in ≥{consistency_threshold}% cell lines:** {len(consistently_up_genes)}")
            st.write(f"**Genes downregulated in ≥{consistency_threshold}% cell lines:** {len(consistently_down_genes)}")

            # --- HEATMAP ---
            consistent_genes = consistently_up_genes.union(consistently_down_genes)
            # Prepare heatmap data
            heatmap_data = merged_df.loc[consistent_genes].copy()
            heatmap_data = heatmap_data.fillna(0)
            heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0)

            row_linkage = linkage(pdist(heatmap_data, metric='correlation'), method='average')
            col_linkage = linkage(pdist(heatmap_data.T, metric='correlation'), method='average')

            st.subheader("🧯 Hierarchical Clustering Heatmap")
            fig = sns.clustermap(
                heatmap_data,
                row_linkage=row_linkage,
                col_linkage=col_linkage,
                cmap='RdBu_r',
                center=0,
                linewidths=0.5,
                figsize=(10, 10)
            )
            st.pyplot(fig.fig)
            plt.clf()

            # --- ENRICHMENT ANALYSIS ---
            st.subheader("🔬 Pathway Enrichment (Enrichr via GSEApy)")

            def run_enrichment(gene_set, label):
                if len(gene_set) == 0:
                    st.warning(f"No genes available for enrichment in {label} set.")
                    return None
                symbols = [gene[1] for gene in gene_set]
                try:
                    enr = gp.enrichr(
                        gene_list=symbols,
                        gene_sets=["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"],
                        organism='Human',
                        outdir=None,
                        cutoff=0.05
                    )
                    if enr.results.empty:
                        st.info(f"No significant enrichment found for {label}.")
                    else:
                        st.write(f"**Top enriched pathways in {label} genes:**")
                        st.dataframe(enr.results.head(10))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        barplot(enr.results.sort_values('Adjusted P-value').head(10), title=f"{label} Enrichment", ax=ax)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error in enrichment analysis for {label}: {e}")

            run_enrichment(consistently_up_genes, "Upregulated")
            run_enrichment(consistently_down_genes, "Downregulated")
