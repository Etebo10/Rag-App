import os
import tempfile
import streamlit as st
from rag_engine import ingest_documents, build_qa_chain, has_index

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Ask Your Documents",
    page_icon="🧠",
    layout="wide",
)

# ── CDNs & Particles ──
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA3" crossorigin="anonymous" referrerpolicy="no-referrer" />
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
  particlesJS('particles-js', {
    "particles": {
      "number": { "value": 80, "density": { "enable": true, "value_area": 800 } },
      "color": { "value": ["#8b5cf6", "#06b6d4", "#ec4899"] },
      "shape": { "type": "circle" },
      "opacity": { "value": 0.5, "random": true },
      "size": { "value": 3, "random": true },
      "line_linked": { "enable": true, "distance": 150, "color": "#8b5cf6", "opacity": 0.4, "width": 1 },
      "move": { "enable": true, "speed": 2, "direction": "none", "random": false, "straight": false }
    },
    "interactivity": {
      "detect_on": "canvas",
      "events": { "onhover": { "enable": true, "mode": "repulse" }, "onclick": { "enable": true, "mode": "push" } },
      "modes": { "repulse": { "distance": 100 }, "push": { "particles_nb": 4 } }
    },
    "retina_detect": true
  });
});
</script>
<div id="particles-js" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; opacity: 0.3;"></div>
""", unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
:root {
  --primary: #0f4c81;
  --secondary: #1f6f8b;
  --accent: #48bdae;
  --bg-primary: #0c1327;
  --bg-secondary: #111a2d;
  --glass-bg: rgba(255, 255, 255, 0.08);
  --glass-border: rgba(255, 255, 255, 0.15);
  --text-primary: #f0f4ff;
  --text-secondary: #b2c3d9;
  --text-muted: #7d92ab;
  --shadow-soft: 0 20px 45px rgba(12, 19, 40, 0.45);
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background-image: url('https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80');
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  color: var(--text-primary) !important;
  min-height: 100vh;
  margin: 0;
  padding: 0;
}

/* Mobile viewport optimization */
@media screen and (max-width: 768px) {
  html, body, [class*="css"] {
    background-attachment: scroll;
  }
}

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: rgba(17, 26, 45, 0.65); }
::-webkit-scrollbar-thumb { background: rgba(79, 113, 170, 0.65); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79, 113, 170, 0.85); }

[data-testid="stSidebar"] {
  background: rgba(8, 13, 23, 0.85) !important;
  backdrop-filter: blur(14px);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: var(--shadow-soft);
}
[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
}

.hero {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 1.5rem;
  background: var(--glass-bg);
  backdrop-filter: blur(18px);
  border: 1px solid var(--glass-border);
  border-radius: 24px;
  padding: 2.25rem;
  margin-bottom: 1.6rem;
  box-shadow: var(--shadow-soft);
}

.hero > .hero-left {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.hero h1 {
  margin: 0;
  font-size: clamp(2rem, 2.8vw, 3.4rem);
  line-height: 1.08;
  letter-spacing: 0.01em;
  color: #e8f2ff;
}

.hero p {
  font-size: 1.12rem;
  color: var(--text-secondary);
  margin: 0.95rem 0 1.35rem 0;
  max-width: 42rem;
}

.hero .cta {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(72, 189, 174, 0.95);
  color: #052032;
  border-radius: 999px;
  padding: 0.75rem 1.2rem;
  font-weight: 700;
  text-decoration: none;
  box-shadow: 0 14px 20px rgba(15, 76, 129, 0.25);
}

.hero-right img {
  width: 100%;
  border-radius: 18px;
  height: 100%;
  object-fit: cover;
  box-shadow: var(--shadow-soft);
}

.badge-ok, .badge-warn, .badge-err {
  border-radius: 999px;
  padding: 0.25rem 0.78rem;
  font-size: 0.77rem;
  font-family: 'Inter', sans-serif;
}
.badge-ok { background: #185d4f; color: #c9ffe1; border: 1px solid #2c9b7f; }
.badge-warn { background: #5f432d; color: #fff2c7; border: 1px solid #b68b54; }
.badge-err { background: #4f2b2e; color: #ffd2d8; border: 1px solid #ba5e6c; }

.stTextInput > div > div > input,
.stSelectbox > div > div > div > div,
.stButton > button {
  border-radius: 15px !important;
}

.stButton > button {
  font-weight: 700 !important;
  letter-spacing: 0.02em;
}

.bubble-container {
  display: flex;
  align-items: flex-start;
  margin: 1rem 0;
}
.bubble-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  margin-right: 0.75rem;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: 700;
  background: rgba(15, 76, 129, 0.9);
  color: white;
}
.bubble-user {
  background: rgba(3, 35, 63, 0.7);
  border: 1px solid rgba(111, 176, 243, 0.35);
  border-radius: 20px 20px 5px 20px;
  padding: 1.1rem 1.2rem;
  color: #eaf2ff;
  max-width: 75%;
  box-shadow: 0 10px 22px rgba(4, 18, 36, 0.45);
}
.bubble-bot {
  background: rgba(12, 26, 48, 0.76);
  border: 1px solid rgba(69, 126, 190, 0.35);
  border-radius: 5px 20px 20px 20px;
  padding: 1.1rem 1.2rem;
  color: #f9fafb;
  max-width: 75%;
  box-shadow: 0 10px 22px rgba(6, 18, 35, 0.48);
}
.bubble-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
  color: #8daddc;
}

/* ── Responsive Design ────────────────────────────────────────────────────── */

/* Tablet and smaller screens */
@media (max-width: 1024px) {
  .hero {
    grid-template-columns: 1fr;
    gap: 1rem;
    padding: 1.5rem;
  }

  .hero h1 {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
  }

  .hero p {
    font-size: 1rem;
    margin: 0.75rem 0 1rem 0;
  }

  .hero .cta {
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
  }

  .hero-right img {
    height: 250px;
    margin-top: 1rem;
  }
}

/* Mobile screens */
@media (max-width: 768px) {
  html, body, [class*="css"] {
    background-attachment: scroll; /* Better performance on mobile */
  }

  .hero {
    padding: 1.25rem;
    margin-bottom: 1rem;
    border-radius: 16px;
  }

  .hero h1 {
    font-size: clamp(1.5rem, 5vw, 2.2rem);
    line-height: 1.1;
  }

  .hero p {
    font-size: 0.95rem;
    margin: 0.6rem 0 0.9rem 0;
  }

  .hero .cta {
    padding: 0.55rem 0.9rem;
    font-size: 0.85rem;
    width: 100%;
    justify-content: center;
  }

  .hero-right img {
    height: 200px;
    border-radius: 12px;
  }

  /* Sidebar adjustments */
  [data-testid="stSidebar"] {
    width: 280px !important;
    min-width: 280px !important;
  }

  /* Chat bubbles on mobile */
  .bubble-container {
    margin: 0.75rem 0;
  }

  .bubble-avatar {
    width: 28px;
    height: 28px;
    font-size: 0.9rem;
  }

  .bubble-user, .bubble-bot {
    padding: 0.9rem 1rem;
    font-size: 0.95rem;
    max-width: 85%;
  }

  .bubble-label {
    font-size: 0.65rem;
  }

  /* Input area adjustments */
  .stTextInput > div > div > input {
    font-size: 16px !important; /* Prevents zoom on iOS */
    padding: 0.6rem 0.9rem !important;
  }

  .stButton > button {
    padding: 0.6rem 1.5rem !important;
    font-size: 0.95rem !important;
  }

  /* Badge adjustments */
  .badge-ok, .badge-warn, .badge-err {
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
  }
}

/* Small mobile screens */
@media (max-width: 480px) {
  .hero {
    padding: 1rem;
  }

  .hero h1 {
    font-size: 1.8rem;
  }

  .hero p {
    font-size: 0.9rem;
  }

  .hero .cta {
    padding: 0.5rem 0.8rem;
    font-size: 0.8rem;
  }

  .hero-right img {
    height: 180px;
  }

  /* Sidebar on very small screens */
  [data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
  }

  .bubble-user, .bubble-bot {
    max-width: 90%;
    padding: 0.8rem 0.9rem;
  }

  .bubble-avatar {
    width: 24px;
    height: 24px;
    font-size: 0.8rem;
  }
}

/* Touch device optimizations */
@media (hover: none) and (pointer: coarse) {
  .hero .cta:hover {
    transform: none;
  }

  .stButton > button:hover {
    transform: none !important;
  }

  /* Larger touch targets */
  .stButton > button {
    min-height: 44px;
  }

  .hero .cta {
    min-height: 44px;
  }
}
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
  .hero-right img {
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
  }
}

</style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "qa_chain"  not in st.session_state: st.session_state.qa_chain  = None
if "indexed"   not in st.session_state: st.session_state.indexed   = has_index()
if "groq_key"  not in st.session_state: st.session_state.groq_key  = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; background: var(--glass-bg); border-radius: 16px; backdrop-filter: blur(10px); border: 1px solid var(--glass-border); box-shadow: var(--shadow-glow);"><i class="fas fa-brain fa-2x" style="color: var(--primary);"></i><h2 style="color: var(--text-primary); margin: 0.5rem 0 0 0; font-weight: 800;">DocMind</h2></div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Groq API key ──
    st.markdown('<i class="fas fa-key" style="color: var(--accent-gold); margin-right: 0.5rem;"></i><strong>Groq API key</strong>', unsafe_allow_html=True)
    st.markdown(
        '<a href="https://console.groq.com/keys" target="_blank" '
        'style="color:#a89cff;font-size:.82rem;">Get a free key → console.groq.com</a>',
        unsafe_allow_html=True,
    )
    key_input = st.text_input(
        "Groq key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        value=st.session_state.groq_key,
    )
    if key_input != st.session_state.groq_key:
        st.session_state.groq_key = key_input
        st.session_state.qa_chain = None   # rebuild chain with new key

    if st.session_state.groq_key:
        st.markdown('<span class="badge-ok">● API key set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">○ No API key entered</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Document upload ──
    st.markdown('<i class="fas fa-cloud-upload-alt" style="color: var(--secondary); margin-right: 0.5rem;"></i><strong>Upload documents</strong>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button('<i class="fas fa-database"></i> Index documents', use_container_width=True):
        with st.spinner("Chunking & embedding…"):
            tmp_paths = []
            for f in uploaded:
                suffix = ".pdf" if f.type == "application/pdf" else ".txt"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(f.read())
                tmp.close()
                tmp_paths.append(tmp.name)

            n_chunks = ingest_documents(tmp_paths)
            for p in tmp_paths:
                os.unlink(p)

        st.success(f"✓ Indexed {n_chunks} chunks from {len(uploaded)} file(s)")
        st.session_state.indexed  = True
        st.session_state.qa_chain = None

    st.markdown("---")
st.markdown('<i class="fas fa-chart-line" style="color: var(--primary); margin-right: 0.5rem;"></i><strong>Index status</strong>', unsafe_allow_html=True)
if st.session_state.indexed:
    st.markdown('<span class="badge-ok">● Index ready</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-warn">○ No documents indexed yet</span>', unsafe_allow_html=True)

st.markdown("---")
k = st.slider("Passages to retrieve (k)", 1, 10, 4)

st.markdown("---")
st.markdown('<i class="fas fa-robot" style="color: var(--accent-pink); margin-right: 0.5rem;"></i><strong>Groq model</strong>', unsafe_allow_html=True)
model_choice = st.selectbox(
    "Groq model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"],
    label_visibility="collapsed",
)

st.markdown("---")
if st.button('Clear chat', use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="hero-left">
        <div style="display: flex; align-items: center; gap: .5rem; margin-bottom: .6rem; font-size: .95rem; color: #bccfea;">
          <i class="fas fa-award"></i>
          <span>1% elite design, curated experience</span>
        </div>
        <h1 style="color:green">DocMind — Your Studio-Grade Knowledge Companion</h1>
        <p><i class="fas fa-key" style="margin-right: .4rem;"></i>Enter your Groq key, upload documents, and experience an ultra-responsive knowledge workflow with superior visual clarity and craftsmanship.</p>
        <a href="#" class="cta"><i class="fas fa-rocket"></i> Get started in seconds</a>
      </div>
      <div class="hero-right">
        <img src="https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=1400&q=80" alt="High-end UI concept" />
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="bubble-label user-label">You</div>'
            f'<div class="bubble-user">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bubble-label bot-label">DocMind</div>'
            f'<div class="bubble-bot">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            with st.expander("📄 Source passages", expanded=False):
                for src in msg["sources"]:
                    fname   = os.path.basename(src["file"])
                    snippet = src["text"][:260].replace("\n", " ") + "…"
                    st.markdown(
                        f'<div class="source-card"><b>{fname}</b><br>{snippet}</div>',
                        unsafe_allow_html=True,
                    )

# Input row
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    st.markdown('<i class="fas fa-microphone" style="font-size: 1.5rem; color: var(--text-muted); cursor: pointer;" onclick="startVoice()"></i>', unsafe_allow_html=True)
with col2:
    question = st.text_input(
        "question",
        placeholder="Ask a question about your documents… 💭",
        label_visibility="collapsed",
        key="question_input",
        max_chars=None,
    )
with col3:
    ask = st.button("Send", use_container_width=True, icon="➡️")

# ── Query logic ───────────────────────────────────────────────────────────────
if ask and question.strip():
    if not st.session_state.groq_key:
        st.error("Please enter your Groq API key(password) in the sidebar first.")
    elif not st.session_state.indexed:
        st.warning("Please upload and index at least one document first.")
    else:
        if st.session_state.qa_chain is None:
            with st.spinner("Connecting to Groq…"):
                qa = build_qa_chain(k=k, api_key=st.session_state.groq_key)
                st.session_state.qa_chain = qa

        st.session_state.messages.append({"role": "user", "content": question})

        qa = st.session_state.qa_chain
        with st.spinner("Searching documents & generating answer…"):
            answer = qa["chain"].invoke(question)
            docs = qa["retriever"].invoke(question)
            sources = [
                {"file": doc.metadata.get("source", "unknown"), "text": doc.page_content}
                for doc in docs
            ]
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()