import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher

# Page Configuration
st.set_page_config(page_title="Mini-Schrödinger Utility", layout="wide")

st.title("🧪 Mini-Schrödinger Utility")

# --- SIDEBAR: PRESETS ---
st.sidebar.header("Drug Presets")
examples = {
    "Chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Nicotine": "CN1CCCC1c2cccnc2",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
}
selection = st.sidebar.selectbox("Load a preset:", ["None"] + list(examples.keys()))
default_smiles = examples[selection] if selection != "None" else "c1ccccc1"

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw or Edit Structure")
    # The Molecule Editor
    drawn_smiles = st_ketcher(default_smiles)
    st.caption("Draw your molecule and click the 'Apply/Check' button to update analysis.")

# Process the molecule from the drawer
if drawn_smiles:
    mol = Chem.MolFromSmiles(drawn_smiles)
    if mol:
        with col1:
            st.divider()
            st.subheader("Lipinski's Rule of Five")
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            rules = {
                "Property": ["Weight", "LogP", "H-Donors", "H-Acceptors"],
                "Value": [round(mw, 2), round(logp, 2), hbd, hba],
                "Limit": ["< 500", "< 5", "< 5", "< 10"],
                "Status": ["✅" if mw < 500 else "❌", "✅" if logp < 5 else "❌", "✅" if hbd <= 5 else "❌", "✅" if hba <= 10 else "❌"]
            }
            st.table(pd.DataFrame(rules))

        with col2:
            st.subheader("3D Interactive View")
            try:
                m3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                mblock = Chem.MolToMolBlock(m3d)
                view = py3Dmol.view(width=500, height=400)
                view.addModel(mblock, 'mol')
                view.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
                view.zoomTo()
                showmol(view, height=400, width=600)
            except:
                st.error("Could not generate 3D shape for this structure.")

        # --- IONIZATION SECTION ---
        st.divider()
        st.subheader("Ionization Profile")
        
        is_acid = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))
        is_base = mol.HasSubstructMatch(Chem.MolFromSmarts("[NH2,NH1,NH0;D2,D3,D4]"))
        pka_val = 4.5 if is_acid else (9.5 if is_base else 7.0)
        
        ph = np.linspace(1, 14, 100)
        if is_acid:
            y = 100 / (1 + 10**(pka_val - ph))
            title = f"Acidic pKa: {pka_val}"
        elif is_base:
            y = 100 / (1 + 10**(ph - pka_val))
            title = f"Basic pKa: {pka_val}"
        else:
            y = np.zeros_like(ph)
            title = "Neutral"

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ph, y, color='#0077b6', lw=2, label="% Ionized")
        ax.axvline(7.4, color='#e63946', linestyle='--', label="Blood pH (7.4)")
        if is_acid or is_base:
            ax.axvline(pka_val, color='green', linestyle=':', label=f"pKa ({pka_val})")
        ax.set_title(title)
        ax.set_xlabel("pH")
        ax.set_ylabel("% Ionized")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("The drawn structure is not a valid SMILES string.")
