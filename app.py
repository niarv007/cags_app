# ============================================================
# CaGS-AP FINAL MERGED APPLICATION
# UI + Analytics + Predictor (Publication Ready)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
# import matplotlib.pyplot as plt
# import seaborn as sns

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CaGS-AP | CURAJ",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>

/* ===== GLOBAL FONT OVERRIDE (CRITICAL) ===== */
/* Apply Georgia ONLY to app text */
.stApp,
.block-container,
.title,
.subtitle,
.hero,
.section-title,
p,
label,
h1, h2, h3, h4, h5, h6 {
    font-family: Georgia, serif !important;
}

/* ❌ EXCLUDE interactive widgets */
div[data-testid="stDataFrame"],
div[role="menu"],
div[role="menu"] *,
div[class*="Mui"],
div[class*="ag-"],
div[class*="slick"],
canvas {
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
    font-family: Georgia, serif !important;
}

/* Streamlit app background */
.stApp{
    background-image:url("background.png");
    background-size:cover;
    background-attachment:fixed;
}

/* Main content container */
.block-container{
    background:rgba(255,255,255,0.92);
    padding:1.8rem;
    border-radius:20px;
}

/* Titles */
.title{
    text-align:center;
    font-size:36px;
    font-weight:700;
    color:#1f3c88;
}

.subtitle{
    text-align:center;
    font-size:15px;
    color:#555;
}

/* Hero */
.hero{
    background:linear-gradient(90deg,#1f4e79,#3c8dbc);
    padding:22px;
    border-radius:16px;
    color:white;
    text-align:center;
    margin-bottom:15px;
}

/* Section titles */
.section-title{
    font-size:22px;
    font-weight:600;
    text-align:center;
    margin:10px 0 6px 0;
}

/* Workflow image */
.workflow-img img{
    max-height:280px;
    object-fit:contain;
}

/* Morphology images */
.morph-card img{
    max-height:110px;
    border-radius:12px;
    box-shadow:0 6px 16px rgba(0,0,0,0.2);
}

/* Info box spacing */
div[data-testid="stAlert"]{
    margin-top:8px !important;
    margin-bottom:14px !important;
    padding-bottom:12px !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("<div class='title'>CaGS-AP Info</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>FPGE Lab | Central University of Rajasthan</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h2>CaGS-AP</h2>
<h4>AI-Driven Biosolutions for Antifungal Drug Discovery</h4>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    ["ℹ️ CaGS-AP Overview", "🧪 Activity Predictor"],
    horizontal=True
)

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
# FINGERPRINTS
# ============================================================

def fingerprints_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,1024)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,1024,useFeatures=True)
    maccs = MACCSkeys.GenMACCSKeys(mol)

    return np.concatenate([np.array(ecfp),np.array(fcfp),np.array(maccs)])

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
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

        X = pd.DataFrame(fps,index=keep)
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

    final = pd.concat(results_all)
    return final.sort_values("Consensus_Probability",ascending=False)

# ============================================================
# PLOTS
# ============================================================

# def plot_probability(df):
#     fig,ax = plt.subplots()
#     sns.histplot(df["Consensus_Probability"],kde=True,ax=ax)
#     st.pyplot(fig)

# def plot_heatmap(df):
#     prob_cols = [c for c in df.columns if c.endswith("_Prob")]
#     fig,ax = plt.subplots(figsize=(8,5))
#     sns.heatmap(df[prob_cols].head(30),cmap="viridis",ax=ax)
#     st.pyplot(fig)
def plot_probability(df):
    st.subheader("Consensus Probability Distribution")
    st.bar_chart(df["Consensus_Probability"])

def plot_heatmap(df):
    st.subheader("Model Probability Comparison")
    prob_cols = [c for c in df.columns if c.endswith("_Prob")]
    st.dataframe(df[prob_cols].head(30))
# ============================================================
# SINGLE SMILES
# ============================================================

def single_smiles_predict(smiles, models):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES")
        return

    # st.image(Draw.MolToImage(mol,size=(300,300)))

    fp = fingerprints_from_smiles(smiles)
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

    if sd_prob < 0.05:
        conf = "High"
    elif sd_prob < 0.15:
        conf = "Moderate"
    else:
        conf = "Low"

    direction = "Active" if mean_prob >= 0.5 else "Not Active"

    scaffold = get_scaffold(smiles)

    st.success(f"Predicted Activity Probability: **{mean_prob:.4f}**")
    st.write(f"Model Vote: **{votes}/{len(models)}**")
    st.write(f"Std Dev: **{sd_prob:.4f}**")
    st.write(f"Confidence: **{conf} ({direction})**")
    st.write(f"Scaffold: `{scaffold}`")

    st.write("### Model-wise Probabilities")
    st.table(pd.DataFrame(prob_dict,index=["Probability"]).T)

# ============================================================
# OVERVIEW PAGE
# ============================================================

def show_overview():

    st.markdown(
        "<div class='section-title'>Machine Learning Workflow</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='workflow-img'>", unsafe_allow_html=True)
    st.image("Graphical_Abstract_1200_dpi.png", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'><i>Candida albicans</i> Morphological Forms</div>",
        unsafe_allow_html=True
    )

    IMG_DIR = "images"

    cols = st.columns(3)

    forms = [
        ("Yeast_form.png", "Yeast Form"),
        ("Hyphal_form.png", "Hyphal Form"),
        ("Bioflim_form.png", "Biofilm")
    ]

    for col, (img, label) in zip(cols, forms):

        with col:

            st.image(
                os.path.join(IMG_DIR, img),
                width=230
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-weight:700;
                    font-size:18px;
                    font-family: Georgia, serif;
                    margin-top:-5px;
                ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# PREDICTOR PAGE
# ============================================================

def show_predictor():

    st.markdown("<div class='section-title'>CaGS-AP Activity Predictor</div>",unsafe_allow_html=True)

    st.sidebar.header("Input Mode")

    mode = st.sidebar.radio("Choose Option",["Upload CSV","Predict from SMILES"])

    models = st.sidebar.multiselect(
        "Select Models",
        list(AVAILABLE_MODELS.keys()),
        default=list(AVAILABLE_MODELS.keys())
    )

    if mode == "Upload CSV":

        file = st.file_uploader("Upload CSV",type=["csv"])

        if file:

            df = pd.read_csv(file)
            st.dataframe(df.head())

            smiles_col = next((c for c in df.columns if "smile" in c.lower()),None)

            if st.button("Start Virtual Screening"):

                results = run_screening(df,smiles_col,models)

                results = compute_consensus_metrics(results)
                results["Confidence"] = results.apply(assign_confidence,axis=1)
                results["Scaffold"] = results[smiles_col].apply(get_scaffold)

                st.dataframe(results)

                plot_probability(results)
                plot_heatmap(results)

                st.download_button(
                    "Download Results",
                    results.to_csv(index=False),
                    "CaGS_AP_results.csv"
                )

    else:

        smiles = st.text_area("Paste SMILES here:")

        if st.button("Predict Activity"):
            single_smiles_predict(smiles,models)

# ============================================================
# ROUTER
# ============================================================

if page == "ℹ️ CaGS-AP Overview":
    show_overview()
else:
    show_predictor()

st.info(
    "**CaGS-AP** is an AI-driven platform for predicting inhibitors of "
    "*Candida albicans* **β-1,3-glucan synthase**, supporting antifungal drug discovery."
)



