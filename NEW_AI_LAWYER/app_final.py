# -*- coding: utf-8 -*-
"""app.py - Saul Goodman Premium Dark UI"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import time
import base64
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# 1. CẤU HÌNH HỆ THỐNG
# ======================================================

st.set_page_config(
    page_title="Saul Goodman & Associates | Premium Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ASSETS CONFIG ---
LOADING_IMG_PATH = "loading.gif"
AVATAR_IMG_PATH = "avatar.png"
MODEL_FILE_PATH = "phobert_model.pkl"

def get_img_as_base64(file_path):
    if not os.path.exists(file_path): return None
    with open(file_path, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

# ======================================================
# 2. PREMIUM DARK CSS & ANIMATIONS
# ======================================================

st.markdown("""
<style>
    /* 1. TYPOGRAPHY IMPORT */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Poppins:wght@200;300;400;500;600&display=swap');

    /* 2. GLOBAL DARK THEME & RESET */
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        color: #e0e0e0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* BACKGROUND ANIMATION */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        opacity: 0.05;
        z-index: 0;
        pointer-events: none;
    }

    h1, h2, h3 { font-family: 'Playfair Display', serif; }

    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 3. HERO HEADER - GLASSMORPHISM & SHIMMER */
    .hero-container {
        position: relative;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 60px 40px;
        border-radius: 24px;
        margin-bottom: 40px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        overflow: hidden;
        animation: slideDown 1s ease-out;
    }

    .hero-content {
        display: flex;
        align-items: center;
        gap: 40px;
        position: relative;
        z-index: 2;
    }

    /* Avatar Pulse Effect */
    .avatar-glow {
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(229, 185, 76, 0.7);
        animation: pulse-gold 2s infinite;
        border: 3px solid #e5b94c;
    }

    /* Text Shimmer Effect */
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(to right, #bf953f, #fcf6ba, #b38728, #fbf5b7, #aa771c);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0px 0px 20px rgba(191, 149, 63, 0.3);
        margin-bottom: 10px;
        animation: shimmer 3s infinite linear;
        background-size: 200% auto;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 300;
    }

    /* 4. CUSTOM INTERACTIVE INPUTS */
    /* Override Streamlit Input Styles for Dark Mode */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: #e5b94c !important;
        box-shadow: 0 0 15px rgba(229, 185, 76, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    label { color: #e5b94c !important; font-weight: 600 !important; letter-spacing: 1px; }

    /* 5. BUTTONS - GOLD GRADIENT & RIPPLE */
    div.stButton > button {
        background: linear-gradient(135deg, #b38728 0%, #fcf6ba 50%, #bf953f 100%);
        color: #000;
        border: none;
        padding: 18px 36px;
        font-weight: 700;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 30px rgba(229, 185, 76, 0.4);
        filter: brightness(1.1);
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.3);
    }

    /* 6. CARDS & INSIGHT BOXES */
    .glass-card {
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        border-color: rgba(229, 185, 76, 0.3);
    }

    /* Insight Types */
    .insight-proc { border-left: 4px solid #3B82F6; background: linear-gradient(90deg, rgba(59,130,246,0.1) 0%, rgba(0,0,0,0) 100%); }
    .insight-cyber { border-left: 4px solid #F59E0B; background: linear-gradient(90deg, rgba(245,158,11,0.1) 0%, rgba(0,0,0,0) 100%); }
    .insight-land { border-left: 4px solid #10B981; background: linear-gradient(90deg, rgba(16,185,129,0.1) 0%, rgba(0,0,0,0) 100%); }
    .insight-crim { border-left: 4px solid #EF4444; background: linear-gradient(90deg, rgba(239,68,68,0.1) 0%, rgba(0,0,0,0) 100%); }

    /* 7. EXPANDER STYLING */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        color: #e0e0e0 !important;
        font-family: 'Poppins', sans-serif;
    }
    .streamlit-expanderContent {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #cccccc !important;
    }

    /* 8. TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: #888;
        font-weight: 600;
        transition: all 0.3s;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(229, 185, 76, 0.15) !important;
        color: #e5b94c !important;
        border: 1px solid #e5b94c !important;
        box-shadow: 0 0 15px rgba(229, 185, 76, 0.1);
    }

    /* 9. KEYFRAMES ANIMATIONS */
    @keyframes pulse-gold {
        0% { box-shadow: 0 0 0 0 rgba(229, 185, 76, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(229, 185, 76, 0); }
        100% { box-shadow: 0 0 0 0 rgba(229, 185, 76, 0); }
    }
    
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 10px; background: #000; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: #e5b94c; }

</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. DATA & DEEP LEARNING LOGIC (GIỮ NGUYÊN LOGIC CŨ)
# ======================================================

DATA_FILES = {
    "Bộ luật Dân sự": "Bo_luat_Dan_su_final.xlsx",
    "Bộ luật Tố tụng dân sự": "Bo_luat_To_tung_dan_su_final.xlsx",
    "Luật An ninh mạng": "Bo_luat_An_ninh_mang_final.xlsx",
    "Bộ luật Lao động": "Bo_luat_Lao_dong_final.xlsx",
    "Luật Doanh nghiệp": "Bo_luat_Doanh_nghiep_final.xlsx",
    "Luật Hôn nhân & Gia đình": "Bo_luat_Hon_nhan_va_Gia_dinh_final.xlsx",
    "Luật Đất đai": "Bo_luat_Dat_dai_final.xlsx"
}

def get_law_insight(law_name):
    # Cập nhật icon và màu sắc cho phù hợp dark mode
    insights = {
        "Bộ luật Tố tụng dân sự": {"style": "insight-proc", "icon": "⚖️", "title": "Thủ tục Tố tụng", "msg": "Lưu ý đặc biệt về Thời hiệu khởi kiện và Thẩm quyền Tòa án."},
        "Luật An ninh mạng": {"style": "insight-cyber", "icon": "🛡️", "title": "An ninh mạng", "msg": "Cảnh báo: Mọi vi phạm trên không gian số đều bị lưu vết kỹ thuật."},
        "Luật Đất đai": {"style": "insight-land", "icon": "🏛️", "title": "Thủ tục Đất đai", "msg": "Yêu cầu bắt buộc: Hòa giải tại cơ sở trước khi khởi kiện."},
        "Bộ luật Hình sự": {"style": "insight-crim", "icon": "⚠️", "title": "Cảnh báo Hình sự", "msg": "Rủi ro pháp lý cực cao, liên quan đến chế tài tước quyền tự do."},
    }
    default = {"style": "insight-proc", "icon": "📜", "title": "Cơ sở pháp lý", "msg": "Tham khảo kỹ văn bản gốc để đảm bảo quyền lợi."}
    return insights.get(law_name, default)

def extract_article_number(text):
    if pd.isna(text): return ""
    text = str(text)
    match = re.search(r'(\d+)', text)
    return match.group(1) if match else ""

@st.cache_data(show_spinner=False)
def load_and_prep_data():
    merged_df = pd.DataFrame()
    for law_name, filename in DATA_FILES.items():
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename)
                df.columns = [str(c).strip().lower() for c in df.columns]
                col_article = next((c for c in df.columns if 'điều' in c or 'article' in c), None)
                col_content = next((c for c in df.columns if 'nội dung' in c or 'content' in c), None)
                if col_article and col_content:
                    temp = pd.DataFrame()
                    temp['raw_article'] = df[col_article].astype(str).str.strip()
                    temp['content'] = df[col_content].astype(str).str.strip()
                    temp['law_name'] = law_name
                    temp['article_index'] = temp['raw_article'].apply(extract_article_number)
                    temp['full_text_search'] = temp['law_name'] + " " + temp['content'] 
                    merged_df = pd.concat([merged_df, temp], ignore_index=True)
            except Exception: pass
    return merged_df

# --- PHOBERT DEEP LEARNING ENGINE ---

def get_phobert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    if hasattr(outputs, 'last_hidden_state'):
        last_hidden_states = outputs.last_hidden_state
    else:
        last_hidden_states = outputs.hidden_states[-1]
    
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@st.cache_resource(show_spinner=False)
def build_ai_model(df):
    if df.empty: return None, None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối HuggingFace: {e}")
        return None, None, None

    if not os.path.exists(MODEL_FILE_PATH):
        st.error(f"⚠️ Không tìm thấy file '{MODEL_FILE_PATH}'.")
        return None, None, None
        
    try:
        with open(MODEL_FILE_PATH, 'rb') as f:
            model = pickle.load(f)
        model.eval() 
    except Exception as e:
        st.error(f"⚠️ Lỗi load pickle: {e}")
        return None, None, None

    placeholder = st.empty()
    placeholder.markdown("""
        <div style="background:rgba(229,185,76,0.1); border:1px solid #e5b94c; padding:10px; border-radius:8px; text-align:center;">
            <span style="color:#e5b94c; font-weight:bold;">✨ SYSTEM INITIALIZING...</span><br>
            <small style="color:#ccc">Vectorizing Legal Database with PhoBERT Neural Network</small>
        </div>
    """, unsafe_allow_html=True)
    progress_bar = placeholder.progress(0)
    
    corpus = df['full_text_search'].tolist()
    corpus_embeddings = []
    total_rows = len(corpus)
    
    for i, text in enumerate(corpus):
        emb = get_phobert_embedding(str(text), tokenizer, model)
        corpus_embeddings.append(emb.numpy())
        if i % 10 == 0: 
            progress_bar.progress((i + 1) / total_rows)
            
    embedding_matrix = np.vstack(corpus_embeddings)
    
    progress_bar.empty()
    placeholder.empty()
    
    return tokenizer, model, embedding_matrix

def exact_lookup_engine(df, law, num):
    clean = extract_article_number(num)
    if not clean: return None, "Nhập số."
    mask = (df['law_name'] == law) & (df['article_index'] == clean)
    return df[mask], clean

def semantic_analysis_engine(query, tokenizer, model, embedding_matrix, df, top_k=5):
    if model is None or embedding_matrix is None: return pd.DataFrame()
    query_vec = get_phobert_embedding(query, tokenizer, model).numpy()
    sim_scores = cosine_similarity(query_vec, embedding_matrix).flatten()
    top_idx = sim_scores.argsort()[-top_k:][::-1]
    res = df.iloc[top_idx].copy()
    res['score'] = sim_scores[top_idx]
    return res[res['score'] > 0.35]

# ======================================================
# 4. INITIALIZATION (LOADING GIỮ NGUYÊN)
# ======================================================
loading_ph = st.empty()
if 'data_loaded' not in st.session_state:
    with loading_ph.container():
        # Dùng container nhỏ gọn cho loading
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if os.path.exists(LOADING_IMG_PATH):
                b64 = get_img_as_base64(LOADING_IMG_PATH)
                st.markdown(f'<div style="text-align:center; margin-top:100px"><img src="data:image/gif;base64,{b64}" width="150" style="border-radius:50%; border:2px solid #e5b94c; box-shadow: 0 0 20px #e5b94c;"></div>', unsafe_allow_html=True)
            st.markdown("""
                <h3 style='text-align:center; color:#e5b94c; font-family:"Playfair Display"; margin-top:20px; text-transform:uppercase; letter-spacing:2px;'>
                    Saul Goodman is coming...
                </h3>
            """, unsafe_allow_html=True)

    df_legal = load_and_prep_data()
    tokenizer, model, embedding_matrix = build_ai_model(df_legal)

    st.session_state['df_legal'] = df_legal
    st.session_state['tokenizer'] = tokenizer
    st.session_state['model'] = model
    st.session_state['embedding_matrix'] = embedding_matrix
    st.session_state['data_loaded'] = True
    loading_ph.empty()
else:
    df_legal = st.session_state['df_legal']
    tokenizer = st.session_state.get('tokenizer')
    model = st.session_state.get('model')
    embedding_matrix = st.session_state.get('embedding_matrix')

if df_legal.empty:
    st.error("System Error: Database Connection Failed.")
    st.stop()

# ======================================================
# 5. UI RENDERING - NEW LAYOUT
# ======================================================

# --- HERO HEADER ---
avatar_b64 = get_img_as_base64(AVATAR_IMG_PATH)
avatar_html = f'<img src="data:image/png;base64,{avatar_b64}" width="140" height="140" class="avatar-glow">' if avatar_b64 else '<div class="avatar-glow" style="width:140px;height:140px;background:#000;display:flex;align-items:center;justify-content:center;font-size:3rem;">⚖️</div>'

st.markdown(f"""
<div class="hero-container">
    <div class="hero-content">
        <div class="avatar-box">{avatar_html}</div>
        <div class="hero-text">
            <h1 class="hero-title">SAUL GOODMAN</h1>
            <div class="hero-subtitle">PREMIUM CRIMINAL LEGAL CONSULTANT • PHOBERT POWERED</div>
            <div style="margin-top:15px; display:inline-block; padding:5px 15px; border:1px solid #e5b94c; border-radius:20px; color:#e5b94c; font-size:0.8rem; letter-spacing:1px;">
                "BETTER CALL SAUL" FOR LEGAL ISSUES
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS NAVIGATION ---
tab1, tab2 = st.tabs(["TRA CỨU NHANH ", "TƯ VẤN PHÁP LÝ"])

# TAB 1: EXACT LOOKUP
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_law = st.selectbox("CHỌN BỘ LUẬT", list(DATA_FILES.keys()))
    with col2:
        input_article = st.text_input("SỐ ĐIỀU", placeholder="VD: 132")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("TÌM KIẾM", type="primary", key="btn_exact"):
        if not input_article:
            st.warning("Vui lòng nhập số điều khoản.")
        else:
            res, num = exact_lookup_engine(df_legal, selected_law, input_article)
            if res is not None and not res.empty:
                row = res.iloc[0]
                ins = get_law_insight(selected_law)
                
                # Result Card
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:15px; margin-bottom:15px;">
                        <span style="font-family:'Playfair Display'; font-size:1.5rem; color:#e5b94c;">
                            {row['raw_article']}
                        </span>
                        <span style="background:rgba(255,255,255,0.1); padding:5px 10px; border-radius:5px; font-size:0.8rem; color:#aaa;">
                            {selected_law.upper()}
                        </span>
                    </div>
                    <div style="font-size:1.1rem; line-height:1.6; color:#e0e0e0; text-align:justify;">
                        {row['content']}
                    </div>
                </div>
                
                <div class="glass-card {ins['style']}" style="padding: 20px;">
                    <h4 style="margin:0; color:#fff; display:flex; align-items:center; gap:10px;">
                        {ins['icon']} {ins['title']}
                    </h4>
                    <p style="margin:10px 0 0 0; color:rgba(255,255,255,0.8); font-style:italic;">
                        "{ins['msg']}"
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"CASE CLOSED: Không tìm thấy dữ liệu Điều {num}.")

# TAB 2: SEMANTIC SEARCH (PHOBERT)
with tab2:
    if model is None:
        st.error("CRITICAL ERROR: AI Neural Network Offline.")
    else:
        st.markdown("""
        <div style="margin-bottom:20px; color:#aaa; font-style:italic;">
            "Hãy kể cho tôi nghe vấn đề của bạn. Saul sẽ nghiên cứu hàng ngàn văn bản pháp luật để tìm câu trả lời."
        </div>
        """, unsafe_allow_html=True)
        
        user_query = st.text_area(
            "MÔ TẢ TÌNH HUỐNG:",
            height=150,
            placeholder="VD: Tôi muốn đòi lại đất đã cho mượn 10 năm trước nhưng không có giấy tờ viết tay...",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("PHÂN TÍCH", type="primary", key="btn_semantic"):
            if not user_query:
                st.warning("Hồ sơ trống. Vui lòng nhập thông tin.")
            else:
                with st.spinner("CONNECTING TO LEGAL MAINFRAME..."):
                    ai_res = semantic_analysis_engine(user_query, tokenizer, model, embedding_matrix, df_legal, top_k=5)
                    time.sleep(0.8) # Cinematic delay

                if ai_res.empty:
                    st.warning(" Không tìm thấy tiền lệ pháp lý phù hợp.")
                else:
                    top_law = ai_res['law_name'].value_counts().idxmax()
                    dom_ins = get_law_insight(top_law)

                    # AI Insight Box
                    st.markdown(f"""
                    <div class="glass-card {dom_ins['style']}" style="border:1px solid #e5b94c; background: radial-gradient(circle at top right, rgba(229,185,76,0.1), transparent);">
                        <h3 style="color:#e5b94c; margin-top:0; border-bottom:1px solid rgba(229,185,76,0.3); padding-bottom:10px;">
                            PHÂN TÍCH CỦA SAUL
                        </h3>
                        <p style="font-size:1.2rem; color:#fff; margin: 15px 0;">
                            Hệ thống xác định vấn đề thuộc phạm vi: <b style="color:#e5b94c">{top_law}</b>
                        </p>
                        <div style="background:rgba(0,0,0,0.3); padding:15px; border-radius:8px;">
                            <b style="color:#fff;">Lời khuyên pháp lý:</b> 
                            <span style="color:#ddd;">{dom_ins['msg']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<h3 style="color:#e5b94c; margin-top:30px;">CĂN CỨ PHÁP LÝ</h3>', unsafe_allow_html=True)
                    
                    for idx, r in ai_res.iterrows():
                        score_percent = r['score'] * 100
                        # Custom HTML Expander look-alike using standard streamlit expander but styled via CSS
                        with st.expander(f" Điều {r['raw_article']} - {r['law_name']} (Độ khớp: {score_percent:.1f}%)"):
                            st.markdown(f"""
                            <div style="font-family:'Poppins'; line-height:1.6; color:#ccc;">
                                {r['content']}
                            </div>
                            """, unsafe_allow_html=True)

st.markdown("""
<div class="footer fade-in">
    <div class="footer-logo">SAUL GOODMAN & ASSOCIATES</div>
    <div class="footer-divider"></div>
    <p style="margin:15px 0; font-size:1rem; color:rgba(255,255,255,0.7);">
        Premium Legal Consulting Services
    </p>
    <p style="margin:10px 0;">
        © 2025 Saul Goodman & Associates. Phát triển bởi <strong style="color:#e5b94c;">Nhóm 7</strong>
    </p>
    <p style="font-size: 0.8rem; color:rgba(255,255,255,0.4); margin-top:15px;">
        Dữ liệu chỉ mang tính chất tham khảo. Đối với các vụ việc phức tạp, vui lòng liên hệ luật sư chính thức để được tư vấn chi tiết.
    </p>
    <p style="font-size: 0.75rem; color:rgba(229,185,76,0.5); margin-top:10px;">
        "Justice matters most when it's least convenient" - Saul Goodman
    </p>
    <div class="footer-social">
        <a href="#">📱 Twitter</a>
        <a href="#">📱 LinkedIn</a>
        <a href="#">📱 Email</a>
    </div>
</div>
""", unsafe_allow_html=True)