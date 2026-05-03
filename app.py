import logging

# Suppress the harmless "missing ScriptRunContext" warning that fires during
# Streamlit's init phase. A content-based filter works across all Streamlit versions.
class _NoScriptRunCtxWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "missing ScriptRunContext" not in record.getMessage()

logging.getLogger("streamlit").addFilter(_NoScriptRunCtxWarning())

import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import io

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Career Analyzer | AI Resume Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1a0a2e 100%);
    color: #e2e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Main container styling */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 3rem 0 2rem 0;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border-radius: 24px;
    border: 1px solid rgba(99,102,241,0.3);
    margin-bottom: 2.5rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #94a3b8;
    margin-top: 0.75rem;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

/* Text areas */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: rgba(99, 102, 241, 0.9) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* Score card */
.score-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
    border: 1px solid rgba(99,102,241,0.5);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
}

.score-number {
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
}

.score-label {
    font-size: 1.2rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* Keyword pills */
.keyword-pill-missing {
    display: inline-block;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.25rem;
}

.keyword-pill-found {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #6ee7b7;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.25rem;
}

/* Info cards */
.info-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    backdrop-filter: blur(5px);
}

.info-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
    font-weight: 600;
}

/* Analyze button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 3rem !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid rgba(99, 102, 241, 0.2) !important;
    margin: 2rem 0 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

/* Columns gap */
[data-testid="column"] {
    padding: 0 0.75rem !important;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a1a; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }

</style>
""", unsafe_allow_html=True)


# ─── Load SpaCy Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None  # Handled in main thread to avoid ScriptRunContext warning

nlp = load_nlp()
if nlp is None:
    st.error("⚠️ SpaCy model not found. Run: `python -m spacy download en_core_web_sm`")
    st.stop()


# ─── NLP Helper Functions ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove special chars, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_match_score(resume: str, jd: str) -> float:
    """TF-IDF Cosine Similarity between resume and job description."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(resume), clean_text(jd)])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score) * 100, 1)
    except Exception:
        return 0.0


def extract_keywords(text: str, nlp_model) -> set:
    """Extract important technical keywords using SpaCy + regex."""
    tech_pattern = re.compile(
        r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|'
        r'sql|nosql|mongodb|postgresql|mysql|redis|docker|kubernetes|aws|gcp|azure|'
        r'git|ci/cd|tensorflow|pytorch|keras|scikit.learn|pandas|numpy|flask|'
        r'django|fastapi|rest|graphql|html|css|linux|agile|scrum|machine learning|'
        r'deep learning|nlp|data science|power bi|tableau|excel|spark|hadoop|'
        r'c\+\+|c#|ruby|php|swift|kotlin|r\b|matlab|scala|airflow|mlops|devops|'
        r'label encoding|feature engineering|data analysis|visualization)\b',
        re.IGNORECASE
    )
    tech_matches = set(m.lower() for m in tech_pattern.findall(text))

    doc = nlp_model(text[:50000])  # spaCy limit guard
    spacy_tokens = set()
    for chunk in doc.noun_chunks:
        token = chunk.text.strip().lower()
        if 2 <= len(token.split()) <= 4 and len(token) > 3:
            spacy_tokens.add(token)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
            spacy_tokens.add(ent.text.strip().lower())

    return tech_matches | spacy_tokens


def keyword_analysis(resume: str, jd: str, nlp_model) -> dict:
    """Return found and missing keywords + coverage score."""
    jd_keywords   = extract_keywords(jd, nlp_model)
    res_keywords  = extract_keywords(resume, nlp_model)
    resume_lower  = clean_text(resume)

    found, missing = [], []
    for kw in sorted(jd_keywords):
        if kw in res_keywords or kw in resume_lower:
            found.append(kw)
        else:
            missing.append(kw)

    total = len(jd_keywords) if jd_keywords else 1
    coverage = round(len(found) / total * 100, 1)
    return {"found": found, "missing": missing, "coverage": coverage, "total": total}


def score_color(score: float) -> str:
    if score >= 75: return "#34d399"   # green
    if score >= 50: return "#fbbf24"   # yellow
    return "#f87171"                   # red


def score_label(score: float) -> str:
    if score >= 80: return "🟢 Excellent Match"
    if score >= 60: return "🟡 Good Match — Polish Needed"
    if score >= 40: return "🟠 Moderate Match — Needs Work"
    return "🔴 Low Match — Major Gaps Found"


def extract_pdf_text(uploaded_file) -> str:
    raw_bytes = uploaded_file.read()

    # ── Stage 1: PyPDF2 (fast path for text-based PDFs) ─────────────────────
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        text = " ".join(page.extract_text() or "" for page in reader.pages).strip()
        if len(text) > 50:          # meaningful text found → done
            return text
    except Exception:
        pass

    # ── Stage 2: OCR fallback for scanned / image-based PDFs ────────────────
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        # Common Tesseract install paths on Windows
        import os
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in tesseract_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

        with st.spinner("🔍 Scanned PDF detected — running OCR (may take a few seconds)..."):
            images = convert_from_bytes(raw_bytes, dpi=300)
            ocr_text = " ".join(
                pytesseract.image_to_string(img, lang="eng") for img in images
            )
        return ocr_text.strip()

    except ImportError:
        st.warning(
            "📦 OCR packages not installed. Run: `pip install pytesseract pdf2image` "
            "and install **Tesseract OCR** from https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return ""
    except Exception as e:
        st.warning(f"OCR failed: {e}. Try pasting the text instead.")
        return ""


def extract_docx_text(uploaded_file) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(uploaded_file.read()))
        return " ".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.warning(f"DOCX read error: {e}. Try pasting text instead.")
        return ""


# ─── Plotly Gauge Chart ──────────────────────────────────────────────────────────
def make_gauge(score: float, title: str) -> go.Figure:
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Inter"}},
        title={"text": title, "font": {"size": 16, "color": "#94a3b8", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"color": "#475569", "family": "Inter"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(15,23,42,0.0)",
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                {"range": [70,100], "color": "rgba(52,211,153,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        height=280,
        margin=dict(l=30, r=30, t=60, b=10)
    )
    return fig


def make_bar_chart(found: list, missing: list) -> go.Figure:
    labels = ["✅ Found Keywords", "❌ Missing Keywords"]
    values = [len(found), len(missing)]
    colors = ["rgba(52,211,153,0.8)", "rgba(239,68,68,0.8)"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values, textposition='outside',
        textfont={"color": "#e2e8f0", "size": 18, "family": "Inter"},
        width=0.5
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#94a3b8"},
        xaxis={"gridcolor": "rgba(99,102,241,0.1)", "tickfont": {"size": 13}},
        yaxis={"gridcolor": "rgba(99,102,241,0.1)"},
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    return fig


# ─── Improvement Tips ────────────────────────────────────────────────────────────
def get_tips(score: float, missing: list) -> list:
    tips = []
    if score < 50:
        tips.append("📝 **Rewrite your summary** to mirror the language used in the job description.")
    if missing:
        top = ", ".join([f"`{k}`" for k in missing[:5]])
        tips.append(f"🔑 **Add missing keywords** naturally into your experience/skills: {top}.")
    if score < 75:
        tips.append("📊 **Quantify achievements** — use numbers, percentages, and impact metrics.")
    tips.append("🎯 **Tailor your resume** for each application — one size never fits all.")
    tips.append("🔗 **Match section headings** to the job description terminology.")
    return tips


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✨ AI-Powered · NLP · Real-Time Analysis</div>
    <h1 class="hero-title">🚀 Smart Career Analyzer</h1>
    <p class="hero-subtitle">
        Instantly compare your resume against any job description.<br>
        Get a <strong>Match Score</strong>, discover <strong>missing keywords</strong>, and land more interviews.
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Input Section ──────────────────────────────────────────────────────────────
# Initialise session state buckets for uploaded-file text (survives reruns)
if "resume_file_text" not in st.session_state:
    st.session_state["resume_file_text"] = ""
if "jd_file_text" not in st.session_state:
    st.session_state["jd_file_text"] = ""

col_resume, col_jd = st.columns(2, gap="large")

with col_resume:
    st.markdown('<div class="section-header">📄 Your Resume</div>', unsafe_allow_html=True)
    input_mode_r = st.radio(
        "Input mode", ["Paste Text", "Upload File (PDF/DOCX)"],
        horizontal=True, key="resume_mode", label_visibility="collapsed"
    )
    resume_text = ""
    if input_mode_r == "Paste Text":
        # Clear any cached file text when switching to paste mode
        st.session_state["resume_file_text"] = ""
        resume_text = st.text_area(
            "Resume Text", height=320,
            placeholder="Paste your resume content here...\n\nInclude your skills, experience, education, and achievements.",
            key="resume_input", label_visibility="collapsed"
        )
    else:
        uploaded_resume = st.file_uploader(
            "Upload Resume", type=["pdf", "docx"], key="resume_file", label_visibility="collapsed"
        )
        if uploaded_resume:
            # Extract text and cache it; keyed by filename so we re-extract on new uploads
            cache_key = f"resume_{uploaded_resume.name}_{uploaded_resume.size}"
            if st.session_state.get("resume_cache_key") != cache_key:
                if uploaded_resume.name.endswith(".pdf"):
                    extracted = extract_pdf_text(uploaded_resume)
                else:
                    extracted = extract_docx_text(uploaded_resume)
                st.session_state["resume_file_text"] = extracted
                st.session_state["resume_cache_key"] = cache_key
            resume_text = st.session_state["resume_file_text"]
            if resume_text:
                st.success(f"✅ Loaded: {uploaded_resume.name}")
            else:
                st.error("❌ Could not extract text. Ensure Tesseract OCR is installed for scanned PDFs, or paste the content instead.")
        else:
            st.session_state["resume_file_text"] = ""
            st.session_state.pop("resume_cache_key", None)

with col_jd:
    st.markdown('<div class="section-header">💼 Job Description</div>', unsafe_allow_html=True)
    input_mode_j = st.radio(
        "Input mode", ["Paste Text", "Upload File (PDF/DOCX)"],
        horizontal=True, key="jd_mode", label_visibility="collapsed"
    )
    jd_text = ""
    if input_mode_j == "Paste Text":
        st.session_state["jd_file_text"] = ""
        jd_text = st.text_area(
            "Job Description Text", height=320,
            placeholder="Paste the job description here...\n\nInclude requirements, responsibilities, and qualifications.",
            key="jd_input", label_visibility="collapsed"
        )
    else:
        uploaded_jd = st.file_uploader(
            "Upload JD", type=["pdf", "docx"], key="jd_file", label_visibility="collapsed"
        )
        if uploaded_jd:
            cache_key = f"jd_{uploaded_jd.name}_{uploaded_jd.size}"
            if st.session_state.get("jd_cache_key") != cache_key:
                if uploaded_jd.name.endswith(".pdf"):
                    extracted = extract_pdf_text(uploaded_jd)
                else:
                    extracted = extract_docx_text(uploaded_jd)
                st.session_state["jd_file_text"] = extracted
                st.session_state["jd_cache_key"] = cache_key
            jd_text = st.session_state["jd_file_text"]
            if jd_text:
                st.success(f"✅ Loaded: {uploaded_jd.name}")
            else:
                st.error("❌ Could not extract text. Ensure Tesseract OCR is installed for scanned PDFs, or paste the content instead.")
        else:
            st.session_state["jd_file_text"] = ""
            st.session_state.pop("jd_cache_key", None)


# ─── Analyze Button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_clicked = st.button("⚡ Analyze My Resume", key="analyze_btn", type="primary")


# ─── Results ────────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not resume_text.strip() or not jd_text.strip():
        st.warning("⚠️ Please provide both your resume and the job description before analyzing.")
    else:
        with st.spinner("🤖 AI is analyzing your resume..."):
            match_score  = get_match_score(resume_text, jd_text)
            kw_data      = keyword_analysis(resume_text, jd_text, nlp)
            tips         = get_tips(match_score, kw_data["missing"])
            color        = score_color(match_score)
            label        = score_label(match_score)

        st.markdown("---")
        st.markdown('<h2 style="text-align:center;color:#a78bfa;font-size:2rem;font-weight:800;">📊 Analysis Results</h2>', unsafe_allow_html=True)

        # ── Top Metrics Row ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Match Score",      f"{match_score}%")
        m2.metric("✅ Found Keywords",    len(kw_data["found"]))
        m3.metric("❌ Missing Keywords",  len(kw_data["missing"]))
        m4.metric("📋 Keyword Coverage",  f"{kw_data['coverage']}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Verdict Banner ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="
            text-align:center;
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
            border: 1px solid {color}44;
            border-radius: 16px; padding: 1.2rem 2rem; margin: 0.5rem 0 1.5rem 0;">
            <span style="font-size:1.5rem; font-weight:800; color:{color};">{label}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauges + Bar Chart ───────────────────────────────────────────────────
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(make_gauge(match_score, "Overall Match Score (TF-IDF Cosine Similarity)"),
                            use_container_width=True, config={"displayModeBar": False})
        with gc2:
            st.plotly_chart(make_bar_chart(kw_data["found"], kw_data["missing"]),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")

        # ── Keyword Deep Dive ────────────────────────────────────────────────────
        tab_missing, tab_found = st.tabs(["❌  Missing Keywords (Add These!)", "✅  Found Keywords (Great Job!)"])

        with tab_missing:
            if kw_data["missing"]:
                st.markdown(
                    "<p style='color:#94a3b8;margin-bottom:1rem;'>These terms appear in the job description but are <strong style='color:#f87171;'>not found</strong> in your resume. Add them where relevant.</p>",
                    unsafe_allow_html=True
                )
                pills = "".join([f'<span class="keyword-pill-missing">⚠ {kw}</span>' for kw in sorted(kw_data["missing"])])
                st.markdown(f'<div style="line-height:2.5;">{pills}</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 Wow! You haven't missed any key terms. Perfect keyword coverage!")

        with tab_found:
            if kw_data["found"]:
                st.markdown(
                    "<p style='color:#94a3b8;margin-bottom:1rem;'>These terms from the job description are <strong style='color:#34d399;'>present</strong> in your resume. Keep them!</p>",
                    unsafe_allow_html=True
                )
                pills = "".join([f'<span class="keyword-pill-found">✓ {kw}</span>' for kw in sorted(kw_data["found"])])
                st.markdown(f'<div style="line-height:2.5;">{pills}</div>', unsafe_allow_html=True)
            else:
                st.warning("No matching keywords found. Try tailoring your resume more closely to the job description.")

        st.markdown("---")

        # ── Improvement Tips ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Actionable Improvement Tips</div>', unsafe_allow_html=True)
        for i, tip in enumerate(tips):
            st.markdown(f"""
            <div class="info-card">
                <p style="margin:0;font-size:0.95rem;color:#e2e8f0;">{tip}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Download Report ──────────────────────────────────────────────────────
        report = f"""SMART CAREER ANALYZER — REPORT
===============================
Match Score       : {match_score}%
Keyword Coverage  : {kw_data['coverage']}%
Verdict           : {label}

MISSING KEYWORDS ({len(kw_data['missing'])})
{chr(10).join(f'  - {k}' for k in sorted(kw_data['missing'])) or '  None! Great job.'}

FOUND KEYWORDS ({len(kw_data['found'])})
{chr(10).join(f'  + {k}' for k in sorted(kw_data['found'])) or '  None found.'}

TIPS
{chr(10).join(f'  {i+1}. {t}' for i,t in enumerate(tips))}
"""
        _, dl_col, _ = st.columns([1, 2, 1])
        with dl_col:
            st.download_button(
                "📥 Download Full Report (.txt)",
                data=report,
                file_name="career_analysis_report.txt",
                mime="text/plain",
                key="download_report"
            )


# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.85rem; padding: 1rem 0;">
    Built with ❤️ using <strong style="color:#6366f1;">Python</strong> ·
    <strong style="color:#8b5cf6;">SpaCy NLP</strong> ·
    <strong style="color:#a78bfa;">Scikit-learn</strong> ·
    <strong style="color:#60a5fa;">Streamlit</strong>
</div>
""", unsafe_allow_html=True)
