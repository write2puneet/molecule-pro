import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw, FilterCatalog
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher
from fpdf import FPDF
import base64

# Page Configuration
st.set_page_config(page_title="MoleculePro Dashboard", layout="wide")
st.title("🧪 MoleculePro: Ultimate Discovery Suite")

# --- CORE UTILITIES ---
def calculate_sa_score(mol):
    if not mol: return 0
    return round((Descriptors.MolWt(mol) / 100) + Descriptors.NumHDonors(mol) + (0.5 * Descriptors.RingCount(mol)), 2)

def check_pains(mol):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return "PAINS Alert" if catalog.HasMatch(mol) else "Safe"

def make_radar_chart(mw, logp, hbd, hba):
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

def create_pdf(smiles, mw, logp, hbd, hba, sa, pains):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="MoleculePro Discovery Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"SMILES: {smiles}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Molecular Weight: {mw:.2f} Da", ln=True)
    pdf.cell(200, 10, txt=f"LogP: {logp:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"H-Bond Donors: {hbd}", ln=True)
    pdf.cell(200, 10, txt=f"H-Bond Acceptors: {hba}", ln=True)
    pdf.cell(200, 10, txt=f"Synthetic Accessibility (SA): {sa}", ln=True)
    pdf.cell(200, 10, txt=f"PAINS Status: {pains}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR & LAYOUT ---
st.sidebar.header("Data Control")
examples = {"None": "c1ccccc1", "Aspirin": "CC(=O)Oc1ccccc1C(=O)O", "Chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"}
selection = st.sidebar.selectbox("Load Preset:", list(examples.keys()))
upload_file = st.sidebar.file_uploader("Batch Upload (CSV)", type=["csv"])

st.sidebar.divider()
st.sidebar.subheader("⚖️ Scientific Disclaimer")
st.sidebar.info("Physicochemical: 🟢 | Structural: 🟡 | Biological: 🟠")

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
                mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
                hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
                sa, pains = calculate_sa_score(mol), check_pains(mol)
                
                # Report Export Button
                pdf_data = create_pdf(drawn_smiles, mw, logp, hbd, hba, sa, pains)
                st.download_button(label="📄 Download Project PDF Report", data=pdf_data, file_name="molecule_report.pdf", mime="application/pdf")
                
                met_col, rad_col = st.columns([1, 1.2])
                met_col.metric("SA Score", sa, help="1=Easy, 10=Complex")
                met_col.metric("PAINS", pains)
                rad_col.pyplot(make_radar_chart(mw, logp, hbd, hba))

                rules = {"Property": ["Weight", "LogP", "H-Donors", "H-Acceptors"], "Value": [round(mw, 2), round(logp, 2), hbd, hba], "Status": ["✅" if mw < 500 else "❌", "✅" if logp < 5 else "❌", "✅" if hbd <= 5 else "❌", "✅" if hba <= 10 else "❌"]}
                st.table(pd.DataFrame(rules))

            with col_right:
                st.subheader("3D Surface Rendering")
                try:
                    m3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                    view = py3Dmol.view(width=500, height=400)
                    view.addModel(Chem.MolToMolBlock(m3d), 'mol')
                    view.setStyle({'stick': {}})
                    view.addSurface(py3Dmol.SAS, {'opacity': 0.4, 'color': 'lightblue'})
                    view.zoomTo()
                    showmol(view, height=400, width=550)
                except: st.error("3D Error")

            st.divider()
            st.subheader("Ionization Profile")
            is_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))
            is_base = mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2,NH1,NH0;D2,D3,D4]"))
            pka = 4.5 if is_acid else (9.5 if is_base else 7.0)
            ph = np.linspace(1, 14, 100)
            y = 100/(1+10**(pka-ph)) if is_acid else (100/(1+10**(ph-pka)) if is_base else np.zeros_like(ph))
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(ph, y, color='#0077b6', lw=2)
            ax.axvline(7.4, color='red', linestyle='--')
            st.pyplot(fig)

with tab2:
    st.subheader("Batch Molecular Screening")
    if upload_file:
        df = pd.read_csv(upload_file)
        if "SMILES" in df.columns:
            res = [{"SMILES": s, "MW": round(Descriptors.MolWt(Chem.MolFromSmiles(s)), 2), "SA": calculate_sa_score(Chem.MolFromSmiles(s))} for s in df["SMILES"] if Chem.MolFromSmiles(s)]
            st.dataframe(pd.DataFrame(res))
