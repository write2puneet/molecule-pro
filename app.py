import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw, FilterCatalog
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher

# Page Configuration
st.set_page_config(page_title="MoleculePro Dashboard", layout="wide")
st.title("🧪 MoleculePro: Ultimate Discovery Suite")

# --- CORE UTILITIES ---
def calculate_sa_score(mol):
    if not mol: return 0
    return round((Descriptors.MolWt(mol) / 100) + Descriptors.NumRotatableBonds(mol) + (0.5 * Descriptors.RingCount(mol)), 2)

def check_pains(mol):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return "🚩 PAINS Alert" if catalog.HasMatch(mol) else "✅ Safe"

def make_radar_chart(mw, logp, hbd, hba):
    # Standardise values for a 0-1 scale relative to Lipinski limits
    # Limits: MW=500, LogP=5, HBD=5, HBA=10
    categories = ['MolWt', 'LogP', 'H-Donors', 'H-Acceptors']
    values = [mw/500, logp/5, hbd/5, hba/10]
    values += values[:1] # Repeat first value to close the circle
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    # Add a red line showing the "Limit" (1.0)
    ax.plot(angles, [1.0]*len(angles), color='red', linestyle='--', linewidth=1, label='Limit')
    return fig

# --- SIDEBAR ---
st.sidebar.header("Data Control")
examples = {
    "None": "c1ccccc1",
    "Chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
}
selection = st.sidebar.selectbox("Load Preset:", list(examples.keys()))
upload_file = st.sidebar.file_uploader("Batch Upload (CSV)", type=["csv"])

# --- MAIN INTERFACE TABS ---
tab1, tab2 = st.tabs(["Individual Analysis", "Batch Screening"])

with tab1:
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.subheader("Structure Editor")
        drawn_smiles = st_ketcher(examples[selection])
    
    if drawn_smiles:
        mol = Chem.MolFromSmiles(drawn_smiles)
        if mol:
            with col_left:
                st.divider()
                st.subheader("Lipinski Analysis")
                mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
                hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
                
                # Metrics & Radar Chart side-by-side
                met_col, rad_col = st.columns([1, 1.2])
                with met_col:
                    st.metric("SA Score", calculate_sa_score(mol))
                    st.metric("PAINS", check_pains(mol))
                with rad_col:
                    st.pyplot(make_radar_chart(mw, logp, hbd, hba))

                # Lipinski Table
                rules = {
                    "Property": ["Weight", "LogP", "H-Donors", "H-Acceptors"],
                    "Value": [round(mw, 2), round(logp, 2), hbd, hba],
                    "Status": ["✅" if mw < 500 else "❌", "✅" if logp < 5 else "❌", "✅" if hbd <= 5 else "❌", "✅" if hba <= 10 else "❌"]
                }
                st.table(pd.DataFrame(rules))

            with col_right:
                st.subheader("3D Surface Rendering")
                try:
                    m3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                    mblock = Chem.MolToMolBlock(m3d)
                    view = py3Dmol.view(width=500, height=450)
                    view.addModel(mblock, 'mol')
                    view.setStyle({'stick': {}})
                    view.addSurface(py3Dmol.SAS, {'opacity': 0.4, 'color': 'lightblue'})
                    view.zoomTo()
                    showmol(view, height=450, width=550)
                except: st.error("3D generation failed.")

            # --- IONIZATION PROFILE ---
            st.divider()
            st.subheader("Ionization Profile")
            is_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))
            is_base = mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2,NH1,NH0;D2,D3,D4]"))
            pka_val = 4.5 if is_acid else (9.5 if is_base else 7.0)
            
            ph = np.linspace(1, 14, 100)
            y = 100/(1+10**(pka_val-ph)) if is_acid else (100/(1+10**(ph-pka_val)) if is_base else np.zeros_like(ph))
            
            fig_ion, ax_ion = plt.subplots(figsize=(10, 3.5))
            ax_ion.plot(ph, y, color='#0077b6', lw=2.5)
            ax_ion.axvline(7.4, color='red', linestyle='--', label="Blood pH (7.4)")
            if is_acid or is_base: ax_ion.axvline(pka_val, color='green', linestyle=':', label=f"pKa ({pka_val})")
            ax_ion.set_title(f"Predicted Behavior (Estimated pKa: {pka_val})")
            ax_ion.set_xlabel("pH")
            ax_ion.set_ylabel("% Ionized")
            ax_ion.legend()
            st.pyplot(fig_ion)

with tab2:
    st.subheader("Batch Molecular Screening")
    if upload_file:
        df = pd.read_csv(upload_file)
        if "SMILES" in df.columns:
            results = []
            for sm in df["SMILES"]:
                m = Chem.MolFromSmiles(sm)
                if m:
                    results.append({"SMILES": sm, "MW": round(Descriptors.MolWt(m), 2), "LogP": round(Descriptors.MolLogP(m),2), "SA Score": calculate_sa_score(m), "Toxicity": check_pains(m)})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
            st.download_button("Download CSV Result", res_df.to_csv(index=False), "screening_results.csv")
