"""
Dashboard Marketing Intelligence — Vue Business
Application Streamlit professionnelle orientée métier marketing.
Lancer avec : streamlit run app_marketing.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from preprocessing import FeatureEngineer
from segmentation_model import kmeans



# ─────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────
ROSE        = "#F5B8CA"
ROSE_DEEP   = "#D97090"
PEACH       = "#F5C8A0"
PEACH_DEEP  = "#D98A50"
YELLOW      = "#F5E0A0"
YELLOW_DEEP = "#C8A830"
GREEN       = "#B0D8B8"
GREEN_DEEP  = "#4A9E6A"
MINT        = "#A8D8D0"
MINT_DEEP   = "#3A9E96"
LAVENDER    = "#C8B8E8"
LAV_DEEP    = "#7058B8"
BG          = "#FDFAF6"
CARD        = "#FFFFFF"
BORDER      = "#F0E8E0"
TEXT        = "#2E2E2E"
SUBTEXT     = "#7A7A7A"
LIGHT_TEXT  = "#A8A0A0"


# ─────────────────────────────────────────────────────────────
# DARK / LIGHT MODE
# ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

if st.session_state.dark_mode:
    BG         = "#1A1A2E"
    CARD       = "#16213E"
    BORDER     = "#2E2E50"
    TEXT       = "#F0EAF8"
    SUBTEXT    = "#B8A8D0"
    LIGHT_TEXT = "#7A6A90"

    # Insight box backgrounds — sombres et lisibles
    ROSE_DEEP   = "#E8809A"
    PEACH_DEEP  = "#E8945A"
    YELLOW_DEEP = "#D8B840"
    GREEN_DEEP  = "#5ABE7A"
    MINT_DEEP   = "#4AAEA6"
    LAV_DEEP    = "#8068C8"
    ROSE        = "#3A2030"
    PEACH       = "#3A2818"
    YELLOW      = "#3A3010"
    GREEN       = "#1A3020"
    MINT        = "#182830"
    LAVENDER    = "#28204A"
    INSIGHT_GREEN  = "#1A2E20"
    INSIGHT_YELLOW = "#2E2A10"
    INSIGHT_ROSE   = "#2E1520"
    INSIGHT_BLUE   = "#101828"
    INSIGHT_PURPLE = "#1E1530"
else:
    INSIGHT_GREEN  = "#F0FAF2"
    INSIGHT_YELLOW = "#FFF8EC"
    INSIGHT_ROSE   = "#FFF0F5"
    INSIGHT_BLUE   = "#F0F8FF"
    INSIGHT_PURPLE = "#F5F0FF"



PLOTLY_COLORS = [
    ROSE_DEEP, PEACH_DEEP, GREEN_DEEP, MINT_DEEP,
    YELLOW_DEEP, LAV_DEEP, "#C85050", "#5090C8"
]

SEGMENT_META = {
    0: {"name": "Clients VIP",         "icon": "👑", "color": ROSE_DEEP,   "bg": INSIGHT_ROSE ,
        "desc": "Clients premium a fort revenu et depenses elevees. Priorite absolue pour les offres exclusives."},
    1: {"name": "Gros depensiers",      "icon": "💰", "color": PEACH_DEEP,  "bg": PEACH,
        "desc": "Fort volume d'achats. Sensibles aux promotions et offres en volume."},
    2: {"name": "Clients fideles",      "icon": "🤝", "color": GREEN_DEEP,  "bg": INSIGHT_GREEN,
        "desc": "Historique d'achat stable. Apprecient la reconnaissance et les programmes de fidelite."},
    3: {"name": "Clients occasionnels", "icon": "🛍️", "color": MINT_DEEP,   "bg": MINT,
        "desc": "Achats sporadiques. Un programme de relance peut augmenter leur frequence."},
    4: {"name": "Clients inactifs",     "icon": "💤", "color": "#A0A0A0",   "bg": INSIGHT_YELLOW,
        "desc": "N'ont pas achete depuis longtemps. Campagne de re-engagement recommandee."},
    5: {"name": "Nouveaux clients",     "icon": "✨", "color": YELLOW_DEEP, "bg": INSIGHT_YELLOW,
        "desc": "Recemment integres. Accompagnement et fidelisation rapide indispensables."},
    6: {"name": "Clients a risque",     "icon": "⚠️", "color": "#C85050",   "bg": "#FFF0F0",
        "desc": "Signes de desengagement. Attention particuliere et offres personnalisees necessaires."},
    7: {"name": "Clients digitaux",     "icon": "💻", "color": LAV_DEEP,    "bg": INSIGHT_BLUE,
        "desc": "Tres actifs sur les canaux en ligne. Privilegier les campagnes email et web."},
}

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Lato', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FFF8F2 0%, #FFF0F8 50%, #F8FFF8 100%);
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

/* Hide streamlit branding */
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* Hero */
.hero {{
    background: linear-gradient(130deg, {ROSE} 0%, {PEACH} 50%, {YELLOW} 100%);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 6px 30px rgba(0,0,0,0.08);
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px;
    background:rgba(255,255,255,0.18); border-radius:50%;
}}
.hero::after {{
    content:''; position:absolute; bottom:-50px; left:50px;
    width:130px; height:130px;
    background:rgba(255,255,255,0.12); border-radius:50%;
}}
.hero h1 {{
    font-family:'Playfair Display',serif; font-size:2.2rem;
    color:#1E1E1E; margin:0 0 0.3rem 0;
    letter-spacing:-0.5px; position:relative;
}}
.hero p {{
    color:#3E3E3E; font-size:1rem; margin:0;
    position:relative; font-weight:400;
}}

/* KPI grid */
.kpi-grid {{
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(155px, 1fr));
    gap:0.9rem; margin-bottom:1.4rem;
}}
.kpi-card {{
    background:{CARD}; border-radius:16px;
    padding:1.2rem 1.4rem;
    box-shadow:0 2px 16px rgba(0,0,0,0.05);
    border-top:4px solid {ROSE};
    transition:transform 0.2s ease, box-shadow 0.2s ease;
}}
.kpi-card:hover {{ transform:translateY(-2px); box-shadow:0 6px 24px rgba(0,0,0,0.09); }}
.kpi-icon {{ font-size:1.5rem; margin-bottom:0.35rem; }}
.kpi-value {{
    font-family:'Playfair Display',serif;
    font-size:1.85rem; color:{TEXT}; line-height:1; margin-bottom:0.2rem;
}}
.kpi-label {{
    font-size:0.75rem; color:{SUBTEXT}; font-weight:500;
    text-transform:uppercase; letter-spacing:0.06em;
}}

/* Section title */
.section-title {{
    font-family:'Playfair Display',serif; font-size:1.4rem; color:{TEXT};
    margin:1.8rem 0 0.7rem 0;
    display:flex; align-items:center; gap:0.5rem;
}}
.section-title::after {{
    content:''; flex:1; height:1px;
    background:linear-gradient(to right, {BORDER}, transparent);
    margin-left:0.8rem;
}}

/* Insight box */
.insight-box {{
    border-radius:14px; padding:0.9rem 1.2rem; margin:0.5rem 0;
    font-size:0.9rem; line-height:1.6;
    display:flex; align-items:flex-start; gap:0.6rem;
    box-shadow:0 2px 10px rgba(0,0,0,0.04);
}}
.insight-icon {{ font-size:1.1rem; flex-shrink:0; margin-top:0.15rem; }}

/* ── Insight boxes — adaptation dark mode ── */
.insight-box {{
    background: {CARD} !important;
    border-left-color: inherit;
}}
.insight-box span {{
    color: {TEXT} !important;
}}


/* Segment card */
.seg-card {{
    border-radius:13px; padding:1.1rem 1.3rem; margin-bottom:0.7rem;
    border-left:5px solid; box-shadow:0 2px 12px rgba(0,0,0,0.04);
    transition:transform 0.15s ease;
}}
.seg-card:hover {{ transform:translateX(3px); }}

/* Progress bar */
.prog-wrap {{ background:#F0E8E0; border-radius:30px; height:9px; overflow:hidden; margin:0.35rem 0; }}
.prog-fill {{ height:100%; border-radius:30px; transition:width 0.4s ease; }}

/* Recommendation pill */
.pill {{
    display:inline-flex; align-items:center; gap:0.4rem;
    border-radius:30px; padding:0.45rem 1rem;
    font-size:0.84rem; font-weight:500; margin:0.25rem 0.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}}

/* Table */
.clean-table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
.clean-table th {{
    background:#FBF4EE; padding:0.65rem 0.9rem; text-align:left;
    font-weight:600; font-size:0.75rem; text-transform:uppercase;
    letter-spacing:0.05em; color:{SUBTEXT}; border-bottom:2px solid {BORDER};
}}
.clean-table td {{
    padding:0.65rem 0.9rem; border-bottom:1px solid {BORDER}; color:{TEXT};
}}
.clean-table tr:last-child td {{ border-bottom:none; }}
.clean-table tr:hover td {{ background:#FDFAF6; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:0.4rem; background:transparent; }}
.stTabs [data-baseweb="tab"] {{
    border-radius:10px 10px 0 0; padding:0.5rem 1.2rem;
    background:#F5EDE8; font-weight:500; color:{SUBTEXT}; font-size:0.88rem;
}}
.stTabs [aria-selected="true"] {{
    background:{CARD} !important; color:{TEXT} !important;
    border-bottom:3px solid {PEACH_DEEP} !important;
}}

/* Upload zone */
.upload-zone {{
    background:linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.25));
    border:2px dashed {PEACH}; border-radius:18px;
    padding:3rem 2rem; text-align:center;
}}

/* Download button */
.stDownloadButton button {{
    background:linear-gradient(135deg, {PEACH_DEEP}, {ROSE_DEEP}) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    box-shadow:0 3px 12px rgba(0,0,0,0.12) !important;
}}

hr.soft {{ border:none; border-top:1px solid {BORDER}; margin:1.4rem 0; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}


/* ── Sidebar dark/light adaptatif ── */
section[data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSlider span {{
    color: {SUBTEXT} !important;
}}
section[data-testid="stSidebar"] .stFileUploader label {{
    color: {TEXT} !important;
}}

/* ── Fond page principal ── */
.stApp, .stApp > div {{
    background-color: {BG} !important;
}}
[data-testid="stHeader"] {{
    background-color: {BG} !important;
}}
[data-testid="stToolbar"] {{
    background-color: {BG} !important;
}}

/* ── Texte général compatible fond sombre ── */
html, body, [class*="css"], p, span, div, label {{
    color: {TEXT};
}}

/* ── Inputs et widgets ── */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stTextArea textarea {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
}}
.dataframe thead th {{
    background-color: {CARD} !important;
    color: {SUBTEXT} !important;
}}
.dataframe tbody tr {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
}}

/* ── Boutons ── */
.stButton button {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
}}
.stButton button:hover {{
    border-color: {PEACH_DEEP} !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {{
    background: {CARD} !important;
    color: {SUBTEXT} !important;
}}
.stTabs [aria-selected="true"] {{
    background: {BG} !important;
    color: {TEXT} !important;
    border-bottom: 3px solid {PEACH_DEEP} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}


/* Mode sombre — fond page forcé */
.stApp {{ background-color: {BG} !important; }}
[data-testid="stHeader"] {{ background-color: {BG} !important; }}
/* Zone principale upload */
[data-testid="stFileUploader"] {{
    background-color: rgb(181, 50, 100,0.25);
    border: 2px dashed rgb(181, 50, 100,0.25);
    border-radius: 18px;
    padding: 1rem;
    }}

/* Texte */
[data-testid="stFileUploader"] label {{
    color: #7A4E2D;
    font-weight: 600;}}

/* Bouton Browse files */
[data-testid="stFileUploader"] button {{
    background-color: rgb(181,50, 100,0.75);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.4rem 1rem;
    font-weight: 600;
}}

/* Hover bouton */
[data-testid="stFileUploader"] button:hover {{
    background-color: rgb(181, 50, 100,0.25);
    color: white;
}}

/* Texte drag & drop */
[data-testid="stFileUploader"] small {{
    color:  rgb(181, 50, 100,0.25);
}}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def section_title(icon, text):
    st.markdown(f'<div class="section-title">{icon} {text}</div>', unsafe_allow_html=True)

def insight(icon, text, bg="#FFF8EC", border=YELLOW_DEEP):
    st.markdown(f"""
    <div class="insight-box" style="background:{bg};border-left:4px solid {border};">
        <span class="insight-icon">{icon}</span><span>{text}</span>
    </div>""", unsafe_allow_html=True)

def kpi_card(icon, value, label, color=ROSE_DEEP):
    return f"""
    <div class="kpi-card" style="border-top-color:{color};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""

def pill(icon, text, bg, color):
    return f'<span class="pill" style="background:{bg};color:{color};">{icon} {text}</span>'

def prog_bar(pct, color):
    return f"""<div class="prog-wrap">
        <div class="prog-fill" style="width:{pct:.0f}%;background:{color};"></div>
    </div>"""

def get_seg_meta(seg_id):
    return SEGMENT_META.get(int(seg_id) % len(SEGMENT_META), SEGMENT_META[0])

def perf_badge(score):
    if score >= 0.85: return "Excellent", GREEN_DEEP, "#F0FAF2"
    elif score >= 0.70: return "Tres bon", MINT_DEEP, "#F0FAFA"
    elif score >= 0.55: return "Bon", YELLOW_DEEP, "#FDFAEC"
    else: return "A ameliorer", ROSE_DEEP, "#FFF0F5"

def fmt_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.0f}k"
    return str(int(n))

def plotly_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Playfair Display", size=15, color=TEXT)),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family="Lato", color=TEXT),
        height=height,
        margin=dict(t=45, b=30, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#F5EDE8", tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#F5EDE8", tickfont=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    response_model = joblib.load("models/response_model.pkl")
    seg_pipeline   = joblib.load("models/seg_pipeline.pkl")
    return response_model, seg_pipeline

response_model, seg_pipeline = load_models()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:0.5rem 0 1.5rem 0;'>
        <div style='font-size:2rem;margin-bottom:0.3rem;'>📊</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.2rem;color:{TEXT};font-weight:700;'>
            Marketing Intelligence
        </div>
        <div style='font-size:0.73rem;color:{LIGHT_TEXT};margin-top:0.2rem;'>Vue Business — v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Toggle dark/light — AJOUTER ICI
    mode_icon  = "🌙" if not st.session_state.dark_mode else "☀️"
    mode_label = "Mode sombre" if not st.session_state.dark_mode else "Mode clair"
    st.button(
        f"{mode_icon}  {mode_label}",
        on_click=toggle_theme,
        use_container_width=True,
        help="Basculer entre le mode clair et sombre"
    )
    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:0.8rem 0'>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Importer les donnees clients",
        type=["csv", "txt", "tsv"],
        help="Fichier marketing_campaign (TSV ou CSV)"
    )

    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:1rem 0'>",
                unsafe_allow_html=True)

    with st.expander("📘 Variables attendues dans le fichier CSV"):

        expected_cols = pd.DataFrame({
            "Variable": [
                "ID",
                "Year_Birth",
                "Education",
                "Marital_Status",
                "Income",
                "Kidhome",
                "Teenhome",
                "Recency",
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
                "NumDealsPurchases",
                "NumWebPurchases",
                "NumCatalogPurchases",
                "NumStorePurchases",
                "NumWebVisitsMonth",
                "AcceptedCmp1",
                "AcceptedCmp2",
                "AcceptedCmp3",
                "AcceptedCmp4",
                "AcceptedCmp5",
                "Complain",
            ],

            "Description": [
                "Identifiant client",
                "Année de naissance",
                "Niveau d'étude",
                "Situation matrimoniale",
                "Revenu annuel",
                "Nombre d'enfants",
                "Nombre d'adolescents",
                "Dernier achat (jours)",
                "Dépenses vins",
                "Dépenses fruits",
                "Dépenses viande",
                "Dépenses poisson",
                "Dépenses sucreries",
                "Dépenses produits premium",
                "Achats avec promotions",
                "Achats web",
                "Achats catalogue",
                "Achats magasin",
                "Visites web/mois",
                "Campagne 1 acceptée",
                "Campagne 2 acceptée",
                "Campagne 3 acceptée",
                "Campagne 4 acceptée",
                "Campagne 5 acceptée",
                "Réclamation client",
            ]
        })

        st.dataframe(
            expected_cols,
            use_container_width=True,
            height=450
        )

    if uploaded_file:
        st.markdown("**🗺️ Navigation**")
        page = st.radio("Navigation", [
            "🏠  Vue d'ensemble",
            "🧹  Qualite des donnees",
            "👥  Segments clients",
            "🎯  Probabilite de reponse",
            "📦  Analyse produits",
            "📣  Campagnes marketing",
            "🏆  Top clients",
            "📤  Export",
        ], label_visibility="collapsed")

        st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:1rem 0'>",
                    unsafe_allow_html=True)
        st.markdown("**⚙️ Reglages**")
        threshold = st.slider("Seuil de ciblage (%)", 20, 80, 50, 5) / 100

        st.markdown("**Nombre de Segments**")
        n_cluster = st.slider("Nombre de segments", 2, 6, 3, 1)
    else:
        page = "🏠  Vue d'ensemble"
        threshold = 0.5


# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📊 Marketing Intelligence</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# NO FILE
# ─────────────────────────────────────────────────────────────
if not uploaded_file:
    st.markdown(f"""
    <div class="upload-zone">
        <div style='font-size:3rem;margin-bottom:1rem;'>📂</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.6rem;color:{TEXT};margin-bottom:0.5rem;'>
            Importez votre base clients pour commencer
        </div>
        <div style='color:{SUBTEXT};font-size:0.92rem;max-width:480px;margin:0 auto 1.5rem;'>
            Deposez votre fichier <code>marketing_campaign.csv</code> dans la barre laterale.
            Tous les indicateurs seront calcules automatiquement.
        </div>
        <div style='display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;font-size:0.83rem;color:{SUBTEXT};'>
            <span>✅ Segmentation automatique</span>
            <span>✅ Prediction de reponse</span>
            <span>✅ Analyse produits & campagnes</span>
            <span>✅ Export CSV</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def process_data(file_bytes, file_name):
    import io
    content = file_bytes.decode("utf-8")
    sep = "\t" if "\t" in content.split("\n")[0] else ","
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)

    MODEL_COLUMNS = [
        "Year_Birth",
        "Education",
        "Marital_Status",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5"
    ]

    missing_cols = [
        col for col in MODEL_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        st.error(
            f"""
        Colonnes obligatoires manquantes :

        {missing_cols}

        Veuillez importer un fichier compatible.
        """
        )

        st.stop()


    # Nettoyage
    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single"})
        df = df[~df["Marital_Status"].isin(["YOLO", "Absurd"])]

    df.dropna(subset=["Income"], inplace=True)

    # Feature engineering
    pipe_fe = Pipeline([("fe", FeatureEngineer())])
    df_proc = pipe_fe.fit_transform(df)


    return df, df_proc

raw_bytes = uploaded_file.read()
uploaded_file.seek(0)

with st.spinner("Analyse en cours..."):
    df_raw, df_proc = process_data(raw_bytes, uploaded_file.name)

    # Segmentation
    X_seg = seg_pipeline.transform(df_raw)

    kmeans(n_cluster)

    kmeans_model = joblib.load("models/kmeans_model.pkl")
    segments = kmeans_model.predict(X_seg)

    # Response prediction
    try:
        X_resp = df_proc.drop("Response", axis=1)
    except Exception:
        X_resp = df_proc.copy()

    response_proba = response_model.predict_proba(X_resp)[:, 1]
    response_pred  = response_model.predict(X_resp)

    df_proc["Probabilite_Reponse"] = response_proba
    df_proc["Prediction_Reponse"]  = response_pred
    df_proc["Segment"]             = segments

    # =========================================================
    # PROFIL DES SEGMENTS
    # =========================================================

    segment_profile = (
        df_proc
        .groupby("Segment")[[
            "TotalSpend",
            "Income",
            "TotalPurchases",
            "Probabilite_Reponse",
            "NumWebPurchases",
            "Recency",
            "TotalCampaignsAccepted"
        ]]
        .mean()
    )

    # =========================================================
    # NORMALISATION
    # =========================================================

    norm = (
        segment_profile - segment_profile.min()
    ) / (
        segment_profile.max() - segment_profile.min()
    )

    # =========================================================
    # CALCUL DES SCORES METIERS
    # =========================================================

    segment_scores = {}

    for seg in norm.index:

        row = norm.loc[seg]

        spend      = row["TotalSpend"]
        income     = row["Income"]
        purchases  = row["TotalPurchases"]
        response   = row["Probabilite_Reponse"]
        web        = row["NumWebPurchases"]
        recency    = 1 - row["Recency"]  # faible recency = meilleur
        campaigns  = row["TotalCampaignsAccepted"]

        scores = {

            # VIP = riches + grosses dépenses + forte réponse
            "Clients VIP":
                (0.35 * spend) +
                (0.30 * income) +
                (0.20 * response) +
                (0.15 * campaigns),

            # Fidèles = beaucoup d’achats réguliers
            "Clients fideles":
                (0.45 * purchases) +
                (0.30 * campaigns) +
                (0.25 * response),

            # Digitaux = achats web élevés
            "Clients digitaux":
                (0.70 * web) +
                (0.20 * purchases) +
                (0.10 * response),

            # Gros dépensiers = dépenses très élevées
            "Gros depensiers":
                (0.60 * spend) +
                (0.25 * income) +
                (0.15 * purchases),

            # Occasionnels = peu d’achats
            "Clients occasionnels":
                (1 - purchases) * 0.6 +
                (1 - spend) * 0.4,

            # Nouveaux clients
            "Nouveaux clients":
                (0.70 * recency) +
                ((1 - purchases) * 0.30),

            # Inactifs
            "Clients inactifs":
                ((1 - response) * 0.4) +
                ((1 - purchases) * 0.3) +
                ((1 - spend) * 0.3)
        }

        segment_scores[seg] = scores

    # =========================================================
    # ATTRIBUTION UNIQUE DES NOMS
    # =========================================================

    segment_names = {}

    used_names = set()

    for seg, scores in segment_scores.items():

        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for name, score in sorted_scores:

            if name not in used_names:
                segment_names[seg] = name
                used_names.add(name)
                break

    # =========================================================
    # AJOUT AU DATAFRAME
    # =========================================================

    df_proc["Segment_Nom"] = (
        df_proc["Segment"]
        .map(segment_names)
    )
    print("--------------------------------------------------------------------------------")
    print(segment_names)
    print(df_proc["Segment"].value_counts())
    print(df_proc["Segment_Nom"].value_counts())

n_total   = len(df_proc)
n_high    = (df_proc["Probabilite_Reponse"] >= threshold).sum()
pct_high  = n_high / n_total * 100
avg_spend = df_proc["TotalSpend"].mean() if "TotalSpend" in df_proc.columns else 0
avg_income = df_proc["Income"].mean() if "Income" in df_proc.columns else 0

PURCHASE_COLS = [c for c in ["MntWines","MntFruits","MntMeatProducts",
                               "MntFishProducts","MntSweetProducts","MntGoldProds"]
                 if c in df_proc.columns]
CAMPAIGN_COLS = [c for c in ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                               "AcceptedCmp4","AcceptedCmp5"]
                 if c in df_proc.columns]
PURCHASE_LABELS = {
    "MntWines": "Vins", "MntFruits": "Fruits",
    "MntMeatProducts": "Viandes", "MntFishProducts": "Poissons",
    "MntSweetProducts": "Sucreries", "MntGoldProds": "Or/Luxe"
}
CAMPAIGN_LABELS = {
    "AcceptedCmp1": "Campagne 1", "AcceptedCmp2": "Campagne 2",
    "AcceptedCmp3": "Campagne 3", "AcceptedCmp4": "Campagne 4",
    "AcceptedCmp5": "Campagne 5"
}


# ══════════════════════════════════════════════════════════════
# KPI BANNER (always visible)
# ══════════════════════════════════════════════════════════════
kpi_html = '<div class="kpi-grid">'
kpi_html += kpi_card("👥", fmt_num(n_total),        "Clients au total",            MINT_DEEP)
kpi_html += kpi_card("🎯", f"{pct_high:.1f}%",      f"Clients a cibler (≥{threshold*100:.0f}%)", GREEN_DEEP)
kpi_html += kpi_card("🗂️", str(df_proc["Segment"].nunique()), "Segments identifies", PEACH_DEEP)
kpi_html += kpi_card("💳", f"{avg_spend:,.0f} €",   "Depense moyenne",             YELLOW_DEEP)
kpi_html += kpi_card("💼", f"{avg_income:,.0f} €",  "Revenu moyen",                ROSE_DEEP)
kpi_html += kpi_card("📣", str(len(CAMPAIGN_COLS)), "Campagnes analysees",         LAV_DEEP)
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════
if "Vue d'ensemble" in page:

    # Alertes intelligentes
    section_title("🔔", "Alertes et insights")

    if pct_high >= 30:
        insight("🟢", f"<b>{pct_high:.1f}%</b> de vos clients ({n_high:,}) ont une forte probabilite de repondre. C'est le bon moment pour lancer une campagne.", "#F0FAF2", GREEN_DEEP)
    else:
        insight("🟡", f"Seulement <b>{pct_high:.1f}%</b> de vos clients semblent prets a repondre. Envisagez d'adapter votre message marketing.", "#FDFAEC", YELLOW_DEEP)

    best_seg_id   = df_proc.groupby("Segment")["Probabilite_Reponse"].mean().idxmax()
    best_seg_name = get_seg_meta(best_seg_id)["name"]
    insight("💡", f"Le segment <b>{best_seg_name}</b> affiche la meilleure probabilite de reponse moyenne. Privilegiez ce groupe pour vos prochaines actions.", "#FFF8EC", PEACH_DEEP)

    if PURCHASE_COLS:
        top_cat_raw = df_proc[PURCHASE_COLS].sum().idxmax()
        top_cat     = PURCHASE_LABELS.get(top_cat_raw, top_cat_raw)
        top_amt     = df_proc[PURCHASE_COLS].sum().max()
        insight("📦", f"La categorie <b>{top_cat}</b> represente le plus fort volume de depenses ({top_amt:,.0f} €). Exploitez ce levier dans vos offres.", "#F5F0FF", LAV_DEEP)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        section_title("🗂️", "Repartition des segments")
        seg_counts = df_proc["Segment_Nom"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Effectif"]
        seg_counts["Pct"] = (seg_counts["Effectif"] / n_total * 100).round(1)

        colors_pie = [get_seg_meta(i)["color"] for i in range(len(seg_counts))]
        fig_pie = go.Figure(go.Pie(
            labels=seg_counts["Segment"], values=seg_counts["Effectif"],
            hole=0.42,
            marker=dict(colors=PLOTLY_COLORS[:len(seg_counts)], line=dict(color="white", width=2.5)),
            textinfo="percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value} clients (%{percent})<extra></extra>"
        ))
        fig_pie.update_layout(
            paper_bgcolor=CARD, font=dict(family="Lato"),
            height=340, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(font=dict(size=10), orientation="v")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        section_title("🎯", "Potentiel de reponse par segment")
        seg_proba = df_proc.groupby("Segment_Nom")["Probabilite_Reponse"].mean().sort_values() * 100
        bar_colors = [GREEN_DEEP if v >= threshold*100 else (PEACH_DEEP if v >= 30 else "#B0B0B0")
                      for v in seg_proba.values]
        fig_bar = go.Figure(go.Bar(
            x=seg_proba.values, y=seg_proba.index, orientation="h",
            marker=dict(color=bar_colors, line=dict(color="white", width=1)),
            text=[f"{v:.0f}%" for v in seg_proba.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"
        ))
        fig_bar.add_vline(x=threshold*100, line=dict(color=TEXT, width=1.5, dash="dash"))
        fig_bar = plotly_layout(fig_bar, height=340)
        fig_bar.update_layout(xaxis_title="Probabilite de reponse (%)", yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)

    # Scatter revenu vs depenses
    section_title("🗺️", "Carte clients — Revenu vs Depenses")
    if "Income" in df_proc.columns and "TotalSpend" in df_proc.columns:
        fig_scatter = px.scatter(
            df_proc,
            x="Income", y="TotalSpend",
            color="Segment_Nom",
            size="Probabilite_Reponse",
            size_max=18,
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"Income": "Revenu (€)", "TotalSpend": "Depenses totales (€)",
                    "Segment_Nom": "Segment"},
            hover_data={"Probabilite_Reponse": ":.2f"},
            opacity=0.7,
        )
        fig_scatter = plotly_layout(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)


elif "Qualite" in page:

    section_title("🧹", "Qualite des donnees")

    missing_data = pd.DataFrame({
        "Variable": df_raw.columns,
        "Valeurs manquantes": df_raw.isnull().sum().values,
        "% Manquant": (
            df_raw.isnull().mean() * 100
        ).round(2).values
    })

    st.dataframe(
        missing_data.sort_values(
            "% Manquant",
            ascending=False
        ),
        use_container_width=True
    )

    insight(
        "📌",
        "Cette vue permet d'identifier les problemes de qualite des donnees avant les campagnes marketing.",
        "#FFF8EC",
        YELLOW_DEEP
    )

# ══════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTS CLIENTS
# ══════════════════════════════════════════════════════════════
elif "Segments" in page:
    section_title("👥", "Vos segments clients")
    # insight("💡",
    #         "Votre base a ete divisee automatiquement en groupes homogenes. "
    #         "Chaque segment partage des comportements d'achat similaires.",
    #         "#FFF8EC", YELLOW_DEEP)

    seg_counts = df_proc["Segment_Nom"].value_counts()
    cols = st.columns(2)
    for i, (seg_name, count) in enumerate(seg_counts.items()):
        meta = next((v for v in SEGMENT_META.values() if v["name"] == seg_name),
                    list(SEGMENT_META.values())[i % len(SEGMENT_META)])
        col = cols[i % 2]
        pct = count / n_total * 100
        proba_avg = df_proc[df_proc["Segment_Nom"] == seg_name]["Probabilite_Reponse"].mean() * 100
        color, bg, icon, desc = meta["color"], meta["bg"], meta["icon"], meta["desc"]


        with col:
            reco_pills = {
                "Gros depensiers":     [("💰","Bundle produits",PEACH_DEEP),("🏷️","Reduction volume",YELLOW_DEEP)],
                "Clients VIP":         [("👑","Offre premium",LAV_DEEP),("🎁","Fidelisation VIP",ROSE_DEEP)],
                "Clients fideles":     [("🤝","Carte fidelite",GREEN_DEEP),("🎉","Recompense anniversaire",MINT_DEEP)],
                "Clients occasionnels":[("📧","Email personnalise",MINT_DEEP),("⏰","Offre limitee",YELLOW_DEEP)],
                "Clients digitaux":    [("💻","Campagne digitale",LAV_DEEP),("📱","Push notification",MINT_DEEP)],
                "Nouveaux clients":    [("✨","Email bienvenue",LAV_DEEP),("📘","Guide produits",GREEN_DEEP)],
                "Clients inactifs":    [("💌","Re-engagement",ROSE_DEEP),("🎯","Offre retour",PEACH_DEEP)],
                "Clients a risque":    [("⚠️","Alerte equipe",  "#C85050"),("📞","Suivi personnalise",PEACH_DEEP)],
            }
            pills_html = "".join([
                f'<span class="pill" style="background:{bg};color:{c};">{ic} {tx}</span>'
                for ic, tx, c in reco_pills.get(seg_name, [])
            ])

            st.markdown(f"""
            <div class="seg-card" style="background:{bg};border-left-color:{color};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div style="font-family:'Playfair Display',serif;font-size:1.05rem;
                                    font-weight:600;color:{color};">{icon} {seg_name}</div>
                        <div style="display:flex;align-items:baseline;gap:0.4rem;margin:0.2rem 0;">
                            <span style="font-family:'Playfair Display',serif;font-size:1.7rem;">{count:,}</span>
                            <span style="font-size:0.82rem;color:{SUBTEXT};">clients · {pct:.1f}%</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.72rem;color:{SUBTEXT};margin-bottom:0.15rem;">Prob. reponse</div>
                        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                                    color:{GREEN_DEEP if proba_avg>=50 else PEACH_DEEP};">{proba_avg:.0f}%</div>
                    </div>
                </div>
                {prog_bar(pct, color)}
                <div style="font-size:0.85rem;color:{SUBTEXT};margin:0.4rem 0 0.6rem;">{desc}</div>
                <div>{pills_html}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    section_title("📊", "Profil moyen par segment")

    compare_vars = {c: l for c, l in {
        "TotalSpend": "Depenses totales (€)",
        "Income": "Revenu (€)",
        "TotalPurchases": "Nombre d'achats",
        "TotalCampaignsAccepted": "Campagnes acceptees",
        "Recency": "Recence (jours)",
        "NumWebPurchases": "Achats en ligne"
    }.items() if c in df_proc.columns}
# get_seg_meta
    if compare_vars:
        profile = df_proc.groupby("Segment_Nom")[list(compare_vars.keys())].mean().round(1)
        profile.columns = list(compare_vars.values())
        st.dataframe(
            profile.style.background_gradient(cmap="YlOrRd", axis=0).format("{:.1f}"),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════
# PAGE 3 — PROBABILITE DE REPONSE
# ══════════════════════════════════════════════════════════════
elif "Probabilite" in page:
    section_title("🎯", "Probabilite de reponse a vos campagnes")

    n_medium = ((df_proc["Probabilite_Reponse"] >= 0.3) &
                (df_proc["Probabilite_Reponse"] < threshold)).sum()
    n_low    = (df_proc["Probabilite_Reponse"] < 0.3).sum()

    kpi2 = '<div class="kpi-grid">'
    kpi2 += kpi_card("🚀", f"{n_high:,}",   f"A cibler (≥{threshold*100:.0f}%)", GREEN_DEEP)
    kpi2 += kpi_card("📧", f"{n_medium:,}", "A relancer (moyen)",                YELLOW_DEEP)
    kpi2 += kpi_card("💤", f"{n_low:,}",    "A ne pas solliciter",               "#909090")
    kpi2 += kpi_card("📈", f"{pct_high:.1f}%","Taux de reponse prevu",           PEACH_DEEP)
    kpi2 += '</div>'
    st.markdown(kpi2, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("📊", "Repartition du potentiel")
        fig_pot = go.Figure(go.Bar(
            x=[f"Fort (≥{threshold*100:.0f}%)", "Moyen (30–{:.0f}%)".format(threshold*100), "Faible (<30%)"],
            y=[n_high, n_medium, n_low],
            marker=dict(color=[GREEN_DEEP, PEACH_DEEP, "#B0B0B0"],
                        line=dict(color="white", width=2)),
            text=[f"{n_high:,}", f"{n_medium:,}", f"{n_low:,}"],
            textposition="outside",
        ))
        fig_pot = plotly_layout(fig_pot, "Clients par niveau de potentiel", height=320)
        fig_pot.update_layout(yaxis_title="Effectif", showlegend=False)
        st.plotly_chart(fig_pot, use_container_width=True)

    with col2:
        section_title("📈", "Distribution des probabilites")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_proc["Probabilite_Reponse"] * 100, nbinsx=40,
            marker=dict(color=MINT, line=dict(color="white", width=0.5)),
            name="Tous les clients"
        ))
        fig_hist.add_vline(x=threshold*100, line=dict(color=GREEN_DEEP, width=2, dash="dash"))
        fig_hist.add_vline(x=30, line=dict(color="#C0C0C0", width=1.5, dash="dot"))
        fig_hist = plotly_layout(fig_hist, "Distribution des probabilites (%)", height=320)
        fig_hist.update_layout(xaxis_title="Probabilite (%)", yaxis_title="Effectif", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    section_title("💡", "Recommandations strategiques")

    recos = [
        (GREEN_DEEP, GREEN, "🚀", "Ciblage prioritaire",
         f"<b>{n_high:,} clients</b> ont une probabilite de reponse elevee. "
         f"Lancez une campagne immediate pour maximiser votre ROI."),
        (PEACH_DEEP, PEACH, "📧", "Relance moderee",
         f"<b>{n_medium:,} clients</b> ont un potentiel moyen. "
         f"Un message personnalise avec une offre attractive peut les convaincre."),
        ("#909090", "#F5F5F5", "💤", "Pause strategique",
         f"<b>{n_low:,} clients</b> ne semblent pas prets. "
         f"Evitez de les solliciter pour preserver l'image de marque."),
    ]
    for c_d, c_l, ico, title, txt in recos:
        st.markdown(f"""
        <div class="insight-box" style="background:{c_l};border-left:4px solid {c_d};">
            <span class="insight-icon">{ico}</span>
            <div>
                <div style="font-weight:700;margin-bottom:0.2rem;color:{c_d};">{title}</div>
                <div style="font-size:0.88rem;">{txt}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSE PRODUITS
# ══════════════════════════════════════════════════════════════
elif "produits" in page.lower():
    section_title("📦", "Analyse des achats par categorie")

    if not PURCHASE_COLS:
        st.warning("Aucune colonne de depenses trouvee dans le dataset.")
    else:
        totals = df_proc[PURCHASE_COLS].sum()
        labels = [PURCHASE_LABELS.get(c, c) for c in PURCHASE_COLS]

        col1, col2 = st.columns(2)

        with col1:
            section_title("🥧", "Repartition des depenses")
            fig_pie_prod = go.Figure(go.Pie(
                labels=labels, values=totals.values,
                hole=0.38,
                marker=dict(colors=PLOTLY_COLORS[:len(labels)],
                            line=dict(color="white", width=2)),
                textinfo="label+percent", textfont=dict(size=10),
            ))
            fig_pie_prod.update_layout(
                paper_bgcolor=CARD, font=dict(family="Lato"),
                height=340, margin=dict(t=10,b=10,l=10,r=10),
                showlegend=False
            )
            st.plotly_chart(fig_pie_prod, use_container_width=True)

        with col2:
            section_title("💰", "Montant total par categorie")
            fig_bar_prod = go.Figure(go.Bar(
                x=labels, y=totals.values,
                marker=dict(color=PLOTLY_COLORS[:len(labels)],
                            line=dict(color="white", width=1.5)),
                text=[f"{v:,.0f} €" for v in totals.values],
                textposition="outside"
            ))
            fig_bar_prod = plotly_layout(fig_bar_prod, height=340)
            fig_bar_prod.update_layout(yaxis_title="Montant total (€)", showlegend=False)
            st.plotly_chart(fig_bar_prod, use_container_width=True)

        st.markdown("<hr class='soft'>", unsafe_allow_html=True)
        section_title("🗂️", "Depenses moyennes par segment")

        seg_prod = df_proc.groupby("Segment_Nom")[PURCHASE_COLS].mean().round(1)
        seg_prod.columns = [PURCHASE_LABELS.get(c, c) for c in seg_prod.columns]

        fig_heatmap = go.Figure(go.Heatmap(
            z=seg_prod.values,
            x=seg_prod.columns.tolist(),
            y=seg_prod.index.tolist(),
            colorscale=[[0, "#FFF8EC"], [0.5, PEACH], [1, PEACH_DEEP]],
            text=seg_prod.values.round(0),
            texttemplate="%{text:.0f} €",
            textfont=dict(size=10),
        ))
        fig_heatmap = plotly_layout(fig_heatmap, "Depenses moyennes par segment et categorie", height=380)
        fig_heatmap.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_heatmap, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — CAMPAGNES MARKETING
# ══════════════════════════════════════════════════════════════
elif "Campagnes" in page:
    section_title("📣", "Performance des campagnes marketing")

    if not CAMPAIGN_COLS:
        st.warning("Aucune colonne de campagne trouvee.")
    else:
        cmp_labels = [CAMPAIGN_LABELS.get(c, c) for c in CAMPAIGN_COLS]
        cmp_totals = [df_proc[c].sum() for c in CAMPAIGN_COLS]
        cmp_rates  = [df_proc[c].mean() * 100 for c in CAMPAIGN_COLS]

        # KPI campagnes
        kpi_cmp = '<div class="kpi-grid">'
        for lbl, tot, rate, color in zip(cmp_labels, cmp_totals, cmp_rates, PLOTLY_COLORS):
            kpi_cmp += kpi_card("📣", f"{tot:,}", f"{lbl} — {rate:.1f}% taux", color)
        kpi_cmp += '</div>'
        st.markdown(kpi_cmp, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            section_title("📊", "Taux d'acceptation par campagne")
            fig_cmp_bar = go.Figure(go.Bar(
                x=cmp_labels, y=cmp_rates,
                marker=dict(color=PLOTLY_COLORS[:len(CAMPAIGN_COLS)],
                            line=dict(color="white", width=1.5)),
                text=[f"{r:.1f}%" for r in cmp_rates],
                textposition="outside"
            ))
            fig_cmp_bar = plotly_layout(fig_cmp_bar, height=340)
            fig_cmp_bar.update_layout(yaxis_title="Taux d'acceptation (%)", showlegend=False)
            st.plotly_chart(fig_cmp_bar, use_container_width=True)

        with col2:
            section_title("🗂️", "Acceptations par segment")
            seg_cmp = df_proc.groupby("Segment_Nom")[CAMPAIGN_COLS].mean() * 100
            seg_cmp.columns = cmp_labels
            fig_seg_cmp = go.Figure()
            for i, col_name in enumerate(seg_cmp.columns):
                fig_seg_cmp.add_trace(go.Bar(
                    name=col_name, x=seg_cmp.index, y=seg_cmp[col_name],
                    marker_color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
                ))
            fig_seg_cmp.update_layout(barmode="group")
            fig_seg_cmp = plotly_layout(fig_seg_cmp, height=340)
            fig_seg_cmp.update_layout(yaxis_title="Taux d'acceptation (%)")
            st.plotly_chart(fig_seg_cmp, use_container_width=True)

        st.markdown("<hr class='soft'>", unsafe_allow_html=True)
        section_title("💡", "Insights campagnes")

        best_cmp_idx = int(np.argmax(cmp_rates))
        worst_cmp_idx = int(np.argmin(cmp_rates))
        insight("🏆", f"<b>{cmp_labels[best_cmp_idx]}</b> est la campagne la plus performante avec un taux d'acceptation de <b>{cmp_rates[best_cmp_idx]:.1f}%</b>. Reproduisez son approche dans les prochaines actions.", "#F0FAF2", GREEN_DEEP)
        insight("📉", f"<b>{cmp_labels[worst_cmp_idx]}</b> affiche le taux d'acceptation le plus faible (<b>{cmp_rates[worst_cmp_idx]:.1f}%</b>). Revisez le message ou le ciblage de cette campagne.", "#FFF0F0", ROSE_DEEP)

        total_accepted = sum(cmp_totals)
        avg_rate = np.mean(cmp_rates)
        insight("📊", f"En moyenne, <b>{avg_rate:.1f}%</b> des clients acceptent une campagne. Sur les {n_total:,} clients, cela represente environ <b>{total_accepted//len(CAMPAIGN_COLS):,}</b> reponses positives par campagne.", "#FFF8EC", YELLOW_DEEP)


# ══════════════════════════════════════════════════════════════
# PAGE 6 — TOP CLIENTS
# ══════════════════════════════════════════════════════════════
elif "Top" in page:
    section_title("🏆", "Top clients les plus rentables")
    insight("💡", "Ces clients representent votre base la plus precieuse. Accordez-leur une attention particuliere et des offres exclusives.", "#FFF8EC", YELLOW_DEEP)

    top_cols = [c for c in ["Income","TotalSpend","TotalPurchases",
                             "Probabilite_Reponse","Segment_Nom"]
                if c in df_proc.columns]

    tab_a, tab_b, tab_c = st.tabs(["Par depenses", "Par probabilite de reponse", "Par revenu"])

    def top_table(df_sorted, n=15):
        df_t = df_sorted.head(n)[top_cols].copy().reset_index(drop=True)
        df_t.index = range(1, len(df_t) + 1)
        df_t.rename(columns={
            "Income": "Revenu (€)", "TotalSpend": "Depenses (€)",
            "TotalPurchases": "Achats", "Probabilite_Reponse": "Prob. reponse",
            "Segment_Nom": "Segment"
        }, inplace=True)
        if "Depenses (€)" in df_t.columns:
            df_t["Depenses (€)"] = df_t["Depenses (€)"].apply(lambda x: f"{x:,.0f}")
        if "Revenu (€)" in df_t.columns:
            df_t["Revenu (€)"] = df_t["Revenu (€)"].apply(lambda x: f"{x:,.0f}")
        if "Prob. reponse" in df_t.columns:
            df_t["Prob. reponse"] = df_t["Prob. reponse"].apply(lambda x: f"{x*100:.0f}%")
        st.dataframe(df_t, use_container_width=True, height=450)

    with tab_a:
        if "TotalSpend" in df_proc.columns:
            top_table(df_proc.sort_values("TotalSpend", ascending=False))
    with tab_b:
        top_table(df_proc.sort_values("Probabilite_Reponse", ascending=False))
    with tab_c:
        if "Income" in df_proc.columns:
            top_table(df_proc.sort_values("Income", ascending=False))

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    section_title("📊", "Concentration des depenses (Top 10% vs reste)")

    if "TotalSpend" in df_proc.columns:
        n10 = max(1, int(n_total * 0.1))
        top10_spend  = df_proc.nlargest(n10, "TotalSpend")["TotalSpend"].sum()
        rest_spend   = df_proc["TotalSpend"].sum() - top10_spend
        pct_top10    = top10_spend / (top10_spend + rest_spend) * 100

        fig_conc = go.Figure(go.Pie(
            labels=["Top 10% clients", "Reste 90%"],
            values=[top10_spend, rest_spend],
            hole=0.45,
            marker=dict(colors=[ROSE_DEEP, "#E0E0E0"], line=dict(color="white", width=3)),
            textinfo="label+percent", textfont=dict(size=12),
        ))
        fig_conc.update_layout(paper_bgcolor=CARD, font=dict(family="Lato"),
                                height=300, margin=dict(t=10,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig_conc, use_container_width=True)
        insight("💡", f"Les <b>10% de vos meilleurs clients</b> representent <b>{pct_top10:.1f}%</b> des depenses totales. Protegez cette base et investissez dans leur fidelisation.", "#FFF0F5", ROSE_DEEP)


# ══════════════════════════════════════════════════════════════
# PAGE 7 — EXPORT
# ══════════════════════════════════════════════════════════════
elif "Export" in page:
    section_title("📤", "Export et rapports")
    insight("💡", "Telechargez vos donnees enrichies avec segments et probabilites de reponse pour les utiliser dans vos outils marketing.", "#FFF8EC", YELLOW_DEEP)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)

    col_e1, col_e2, col_e3 = st.columns(3)

    def export_card(col, icon, title, desc, color):
        with col:
            st.markdown(f"""
            <div style='background:{CARD};border-radius:16px;padding:1.4rem;
                        box-shadow:0 2px 16px rgba(0,0,0,0.05);
                        border-top:4px solid {color};margin-bottom:1rem;'>
                <div style='font-size:1.7rem;margin-bottom:0.4rem;'>{icon}</div>
                <div style='font-family:"Playfair Display",serif;font-size:1rem;
                            font-weight:700;margin-bottom:0.35rem;'>{title}</div>
                <div style='font-size:0.83rem;color:{SUBTEXT};line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    export_card(col_e1, "📋", "Base clients enrichie",
                "Tous les clients avec segment et probabilite de reponse.", GREEN_DEEP)
    export_card(col_e2, "🗂️", "Profil des segments",
                "Statistiques moyennes par segment.", PEACH_DEEP)
    export_card(col_e3, "🎯", "Clients prioritaires",
                f"Liste des clients a fort potentiel (≥{threshold*100:.0f}%).", ROSE_DEEP)

    # Export 1
    export_cols = [c for c in df_proc.columns]
    df_exp1 = df_proc[export_cols].copy()
    df_exp1["Prob_Reponse_pct"] = (df_proc["Probabilite_Reponse"] * 100).round(1)
    with col_e1:
        st.download_button("⬇️ Telecharger CSV",
                           df_exp1.to_csv(index=False).encode("utf-8"),
                           "clients_enrichis.csv", "text/csv",
                           use_container_width=True)

    # Export 2
    cmp_cols_proc = [c for c in df_proc.columns if c in
                     ["TotalSpend","Income","TotalPurchases","TotalCampaignsAccepted",
                      "Recency","NumWebPurchases","Segment_Nom","Probabilite_Reponse"]]
    df_exp2 = df_proc[cmp_cols_proc].groupby("Segment_Nom").mean().round(2)
    df_exp2.insert(0, "Effectif", df_proc.groupby("Segment_Nom").size())
    with col_e2:
        st.download_button("⬇️ Telecharger CSV",
                           df_exp2.to_csv().encode("utf-8"),
                           "profil_segments.csv", "text/csv",
                           use_container_width=True)

    # Export 3
    df_exp3 = df_proc[df_proc["Probabilite_Reponse"] >= threshold].sort_values(
        "Probabilite_Reponse", ascending=False
    )
    with col_e3:
        st.download_button(f"⬇️ Telecharger ({len(df_exp3):,} clients)",
                           df_exp3.to_csv(index=False).encode("utf-8"),
                           "clients_prioritaires.csv", "text/csv",
                           use_container_width=True)

    # Rapport texte
    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    section_title("📄", "Rapport de synthese")

    if st.button("🔄 Generer le rapport"):
        best_seg = df_proc.groupby("Segment_Nom")["Probabilite_Reponse"].mean().idxmax()
        best_cmp_txt = ""
        if CAMPAIGN_COLS:
            best_cmp_col = max(CAMPAIGN_COLS, key=lambda c: df_proc[c].mean())
            best_cmp_txt = f"\n  Meilleure campagne         : {CAMPAIGN_LABELS.get(best_cmp_col, best_cmp_col)}"

        lines = [
            "RAPPORT DE SYNTHESE MARKETING",
            "=" * 50,
            f"Nombre de clients          : {n_total:,}",
            f"Segments identifies        : {df_proc['Segment'].nunique()}",
            f"Clients a fort potentiel   : {n_high:,} ({pct_high:.1f}%)",
            f"Depense moyenne            : {avg_spend:,.0f} EUR",
            f"Revenu moyen               : {avg_income:,.0f} EUR",
            "",
            "─" * 50,
            "SEGMENTS",
            "─" * 50,
        ]
        for sn, cnt in df_proc["Segment_Nom"].value_counts().items():
            lines.append(f"  {sn:<30} : {cnt:>5,} clients ({cnt/n_total*100:.1f}%)")
        lines += [
            "",
            "─" * 50,
            "RECOMMANDATIONS",
            "─" * 50,
            f"  1. Cibler en priorite les {n_high:,} clients a fort potentiel.",
            f"  2. Segment le plus reactif : {best_seg}.",
            best_cmp_txt,
            "  3. Lancer une campagne de re-engagement pour les clients inactifs.",
            "",
            "─" * 50,
            "Rapport genere par Marketing Intelligence Dashboard",
        ]
        report_txt = "\n".join(lines)
        st.text_area("Apercu", report_txt, height=300)
        st.download_button("⬇️ Telecharger le rapport (.txt)",
                           report_txt.encode("utf-8"),
                           "rapport_marketing.txt", "text/plain")

    # Apercu base enrichie
    st.markdown("<hr class='soft'>", unsafe_allow_html=True)
    section_title("👀", "Apercu de la base enrichie")
    preview_cols = [c for c in ["Income","TotalSpend","TotalPurchases",
                                 "Segment_Nom","Probabilite_Reponse","Prediction_Reponse"]
                    if c in df_proc.columns]
    df_preview = df_proc[preview_cols].head(100).copy()
    df_preview["Probabilite_Reponse"] = (df_preview["Probabilite_Reponse"] * 100).round(1).astype(str) + "%"
    st.dataframe(df_preview, use_container_width=True, height=380)
    st.caption(f"Affichage des 100 premiers clients sur {n_total:,} au total.")