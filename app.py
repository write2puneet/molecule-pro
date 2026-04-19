import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw, FilterCatalog, rdMolDescriptors
from streamlit_ketcher import st_ketcher
from fpdf import FPDF
import base64
import io
import math

# --- CORE UTILITIES ---
def check_pains(mol):
    """Check for Pan-Assay Interference Compounds (Toxicity Filter)."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return "🚩 PAINS Alert" if catalog.HasMatch(mol) else "✅ Safe"

def make_radar_chart(mw, logp, hbd, hba):
    """Generate a spider plot for Lipinski properties."""
    categories = ['MolWt', 'LogP', 'H-Donors', 'H-Acceptors']
    values = [mw/500, logp/5, hbd/5, hba/10]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.plot(angles, [1.0]*len(angles), color='red', linestyle='--', linewidth=1)
    return fig

def create_pdf(smiles, mw, logp, hbd, hba, pains):
    """Generate a discovery report PDF without emoji encoding errors."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="MoleculePro Discovery Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"SMILES: {smiles}", ln=True)
    pdf.ln(5)
    
    clean_pains = "PAINS Alert" if "Alert" in pains else "Safe"
    pdf.cell(200, 10, txt=f"Molecular Weight: {mw:.2f} | LogP: {logp:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"H-Bond Donors: {hbd} | H-Bond Acceptors: {hba}", ln=True)
    pdf.cell(200, 10, txt=f"Toxicity Screening (PAINS): {clean_pains}", ln=True)
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- UI SETUP ---
st.set_page_config(page_title="MoleculePro Dashboard", layout="wide")
st.title("🧪 MoleculePro: Ultimate Discovery Suite")

# Sidebar
st.sidebar.header("Data Control")
examples = {
    "None": "c1ccccc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}
selection = st.sidebar.selectbox("Load Preset Drug:", list(examples.keys()))
upload_file = st.sidebar.file_uploader("Import Discovery CSV (Tab 2)", type=["csv"])

tab1, tab2 = st.tabs(["Individual Analysis", "Batch & CSV Suite"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with tab1:
    col_draw, col_smiles = st.columns(2)
    with col_draw:
        st.subheader("Structure Editor")
        drawn_smiles = st_ketcher(examples[selection])
    
    with col_smiles:
        st.subheader("Editable SMILES Notation")
        smiles_text = st.text_area("Edit SMILES manually:", value=drawn_smiles, height=350)
        # Apply button to sync the text box to the analysis
        if st.button("🚀 Apply & Sync Changes", use_container_width=True):
            final_smiles = smiles_text
        else:
            final_smiles = drawn_smiles

    if final_smiles:
        mol = Chem.MolFromSmiles(final_smiles)
        if mol:
            st.divider()
            c_props, c_radar = st.columns([1, 1.2])
            with c_props:
                mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
                hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
                pains = check_pains(mol)
                
                try:
                    pdf_bytes = create_pdf(final_smiles, mw, logp, hbd, hba, pains)
                    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="molecule_report.pdf", use_container_width=True)
                except Exception as e:
                    st.error(f"PDF Generation failed: {e}")

                st.metric("PAINS Toxicity Filter", pains)
                st.table(pd.DataFrame({
                    "Property": ["MW", "LogP", "HBD", "HBA"], 
                    "Value": [round(mw,2), round(logp,2), hbd, hba], 
                    "Status": ["✅" if mw < 500 else "❌", "✅" if logp < 5 else "❌", "✅" if hbd <= 5 else "❌", "✅" if hba <= 10 else "❌"]
                }))
            
            with c_radar:
                st.subheader("Lipinski Radar Chart")
                st.pyplot(make_radar_chart(mw, logp, hbd, hba))

            st.divider()
            st.subheader("Ionization Profile")
            is_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))
            is_base = mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2,NH1,NH0;D2,D3,D4]"))
            pka_val = 4.5 if is_acid else (9.5 if is_base else 7.0)
            ph = np.linspace(1, 14, 100)
            y = 100/(1+10**(pka_val-ph)) if is_acid else (100/(1+10**(ph-pka_val)) if is_base else np.zeros_like(ph))
            
            fig_ion, ax_ion = plt.subplots(figsize=(10, 3))
            ax_ion.plot(ph, y, color='#0077b6', lw=2)
            ax_ion.axvline(7.4, color='red', linestyle='--', label="Physiological pH (7.4)")
            ax_ion.set_ylabel("% Ionized")
            ax_ion.set_xlabel("pH")
            st.pyplot(fig_ion)

# # --- TAB 2: BATCH & CSV SUITE (AUTOMATED CALCULATIONS) ---
with tab2:
    st.subheader("Discovery Data Grid")
    
    if upload_file:
        # 1. Initialize session state
        if "discovery_df" not in st.session_state:
            df = pd.read_csv(upload_file, dtype={"DOI": str, "PMID": str, "PID": str, "PatentID": str})
            # Clean technical columns
            for c in ["DOI", "PMID", "PID", "PatentID"]:
                if c in df.columns: df[c] = df[c].fillna("")
            st.session_state.discovery_df = df

        # 2. Dynamic Grid Editor
        # Note: Users can manually edit Mol Weight/Formula here, but the Lab will auto-calc them
        edited_df = st.data_editor(
            st.session_state.discovery_df,
            column_config={
                "Structure": st.column_config.TextColumn("SMILES Structure", width="large"),
                "Mol Weight": st.column_config.NumberColumn("MW (Da)", format="%.2f"),
                "Formula": st.column_config.TextColumn("MF"),
                "DOI": st.column_config.LinkColumn("DOI", disabled=True),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="discovery_editor"
        )
        st.session_state.discovery_df = edited_df

        st.divider()

        # 3. Master-Detail Laboratory with Auto-Calculations
        st.subheader("🔍 Selected Structure Laboratory")
        
        # Use selection mode to identify which row to work on
        selection = st.dataframe(
            st.session_state.discovery_df, 
            on_select="rerun", 
            selection_mode="single-row",
            hide_index=True
        )

        if selection.selection.rows:
            # Extract row index
            sel_idx = selection.selection.rows[0]
            # Handle potential dataframe filtering/sorting index issues
            real_idx = st.session_state.discovery_df.index[sel_idx]
            row_data = st.session_state.discovery_df.loc[real_idx]
            
            lab_col1, lab_col2 = st.columns([1.5, 1])
            
            with lab_col1:
                st.info(f"Modifying: {row_data.get('Compound Name', 'Unnamed Entry')}")
                
                # Load existing Structure into Ketcher
                updated_smiles = st_ketcher(row_data["Structure"])
                
                # 4. Trigger Auto-Calculation on Apply
                if updated_smiles != row_data["Structure"]:
                    temp_mol = Chem.MolFromSmiles(updated_smiles)
                    if temp_mol:
                        new_mw = round(Descriptors.MolWt(temp_mol), 2)
                        new_mf = rdMolDescriptors.CalcMolFormula(temp_mol)
                        
                        st.success(f"New Properties Detected: MW {new_mw} | MF {new_mf}")
                        
                        if st.button("🚀 Sync Structure & Properties to Grid"):
                            st.session_state.discovery_df.at[real_idx, "Structure"] = updated_smiles
                            st.session_state.discovery_df.at[real_idx, "Mol Weight"] = new_mw
                            st.session_state.discovery_df.at[real_idx, "Formula"] = new_mf
                            st.rerun()
                    else:
                        st.error("Invalid Structure: Cannot calculate properties.")

            with lab_col2:
                # High-res visual confirmation
                m = Chem.MolFromSmiles(updated_smiles)
                if m:
                    img = Draw.MolToImage(m, size=(400, 400))
                    st.image(img, caption="Live 2D Preview", use_container_width=True)

        else:
            st.info("💡 **Select a row** in the table above to load it into the Laboratory for drawing and auto-calculations.")

        # 5. Export
        st.divider()
        csv_export = st.session_state.discovery_df.to_csv(index=False)
        st.download_button("📥 Export Updated Discovery CSV", data=csv_export, file_name="MoleculePro_Updated_Data.csv")
    else:
        st.info("Upload your CSV file to enable the Interactive Grid and Property Calculator.")
