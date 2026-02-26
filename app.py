# ============================================================
# CaGS-AP FINAL CLEAN PRODUCTION VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---- Suppress sklearn feature warnings ----
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Safe Matplotlib Backend ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PDF Report ----
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="CaGS-AP | CURAJ", page_icon="🧬", layout="wide")

st.markdown("<h1 style='text-align:center;'>CaGS-AP</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI-Driven Antifungal Activity Predictor</h4>", unsafe_allow_html=True)

# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_DIR = "final_models"
CHUNK_SIZE = 10000

AVAILABLE_MODELS = {
    "Logistic Regression": "tuned_logistic_regression_model.pkl",
    "KNN": "tuned_k-nearest_neighbors_(knn)_model.pkl",
    "Support Vector Machine": "tuned_support_vector_machine_(svm)_model.pkl",
    "MLP Neural Network": "tuned_mlp_neural_network_model.pkl",
    "Random Forest": "tuned_random_forest_model.pkl",
    "XGBoost": "tuned_xgboost_model.pkl",
    "AdaBoost": "tuned_adaboost_model.pkl"
}

# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    return {
        "scaler": joblib.load(os.path.join(MODEL_DIR,"standard_scaler.pkl")),
        "var_thresh": joblib.load(os.path.join(MODEL_DIR,"variance_threshold_selector.pkl")),
        "feat_selector": joblib.load(os.path.join(MODEL_DIR,"model_feature_selector.pkl"))
    }

pipeline = load_pipeline()

def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR,name))

# ============================================================
# FINGERPRINTS (UPDATED TO REMOVE DEPRECATION WARNING)
# ============================================================

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np


def fingerprints_from_smiles(smiles: str):
    """
    Generate combined ECFP4, FCFP4, and MACCS fingerprints
    Returns numpy array or None if invalid SMILES
    """

    if not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        # ECFP4 (Morgan radius=2)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024
        )

        # FCFP4 (Feature-based Morgan)
        fcfp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024,
            useFeatures=True
        )

        # MACCS keys (167 bits)
        maccs = MACCSkeys.GenMACCSKeys(mol)

        # Convert to numpy arrays
        ecfp_arr = np.array(ecfp)
        fcfp_arr = np.array(fcfp)
        maccs_arr = np.array(maccs)

        # Concatenate all fingerprints
        return np.concatenate([ecfp_arr, fcfp_arr, maccs_arr])

    except Exception:
        return None

# ============================================================
# CONSENSUS METRICS
# ============================================================

def compute_consensus_metrics(df):
    pred_cols = [c for c in df.columns if c.endswith("_Pred")]
    prob_cols = [c for c in df.columns if c.endswith("_Prob")]

    df["Mean_Prob"] = df[prob_cols].mean(axis=1)
    df["Prob_SD"] = df[prob_cols].std(axis=1)
    df["Model_Vote"] = df[pred_cols].sum(axis=1)

    return df

def assign_confidence(row):
    if row["Prob_SD"] < 0.05:
        return "High"
    elif row["Prob_SD"] < 0.15:
        return "Moderate"
    return "Low"

# ============================================================
# SCREENING ENGINE
# ============================================================

def run_screening(df, smiles_col, models):

    results_all = []
    total = len(df)
    prog = st.progress(0.0)

    for i in range(0,total,CHUNK_SIZE):

        chunk = df.iloc[i:i+CHUNK_SIZE].copy()

        fps = []
        keep = []

        for j,s in enumerate(chunk[smiles_col]):
            fp = fingerprints_from_smiles(s)
            if fp is not None:
                fps.append(fp)
                keep.append(chunk.index[j])

        if not fps:
            continue

        X = pd.DataFrame(fps, index=keep)

        X = pipeline["var_thresh"].transform(X)
        X = pipeline["feat_selector"].transform(X)
        X = pipeline["scaler"].transform(X)

        res = chunk.loc[keep].copy()
        pcols = []

        for m in models:
            model = load_model(AVAILABLE_MODELS[m])

            pred = model.predict(X)
            res[f"{m}_Pred"] = pred

            prob = model.predict_proba(X)[:,1]
            cname = f"{m}_Prob"
            res[cname] = np.round(prob,4)
            pcols.append(cname)

        res["Consensus_Probability"] = res[pcols].mean(axis=1)
        results_all.append(res)

        prog.progress(min((i+len(chunk))/total,1.0))

    prog.empty()

    if not results_all:
        st.warning("No valid molecules found.")
        return pd.DataFrame()

    final = pd.concat(results_all)
    return final.sort_values("Consensus_Probability", ascending=False)

# ============================================================
# APP LOGIC
# ============================================================

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose Option", ["Upload CSV","Predict from SMILES"])

models = st.sidebar.multiselect(
    "Select Models",
    list(AVAILABLE_MODELS.keys()),
    default=list(AVAILABLE_MODELS.keys())
)

if mode == "Upload CSV":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)
        st.dataframe(df.head())

        smiles_col = next((c for c in df.columns if "smile" in c.lower()), None)

        if smiles_col is None:
            st.error("No SMILES column detected in uploaded file.")
        else:

            if st.button("Start Virtual Screening"):

                results = run_screening(df, smiles_col, models)

                if results.empty:
                    st.stop()

                results = compute_consensus_metrics(results)
                results["Confidence"] = results.apply(assign_confidence, axis=1)
                results["Scaffold"] = results[smiles_col].apply(get_scaffold)

                st.subheader("Screening Results")
                st.dataframe(results)

                st.download_button("Download Results",
                                   results.to_csv(index=False),
                                   "CaGS_AP_results.csv")

else:

    smiles = st.text_area("Paste SMILES here:")

    if st.button("Predict Activity"):

        fp = fingerprints_from_smiles(smiles)

        if fp is None:
            st.error("Invalid SMILES")
        else:

            X = pd.DataFrame([fp])
            X = pipeline["var_thresh"].transform(X)
            X = pipeline["feat_selector"].transform(X)
            X = pipeline["scaler"].transform(X)

            prob_dict = {}
            votes = 0

            for m in models:
                model = load_model(AVAILABLE_MODELS[m])
                pred = int(model.predict(X)[0])
                votes += pred
                prob = float(model.predict_proba(X)[0,1])
                prob_dict[m] = prob

            probs = list(prob_dict.values())
            mean_prob = np.mean(probs)
            sd_prob = np.std(probs)

            st.success(f"Predicted Activity Probability: {mean_prob:.4f}")
            st.write(f"Model Vote: {votes}/{len(models)}")
            st.write(f"Std Dev: {sd_prob:.4f}")

            st.table(pd.DataFrame(prob_dict,index=["Probability"]).T)


