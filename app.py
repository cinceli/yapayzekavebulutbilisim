"""
Streamlit arayüzü. Proje kökünden: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from spam_detector.pipeline import SmsClassifierPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

EXAMPLE_HAM = "Hey, are we still on for coffee at 5? See you there!"
EXAMPLE_SPAM = "WINNER!! You have been selected for a $1000 prize. Text YES to 80000 now to claim."


def _inject_styles() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

  html, body, [class*="stApp"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif;
  }

  /* Üst çubuk — ikonlar SVG currentColor */
  header[data-testid="stHeader"] {
    background-color: #27272a !important;
    border-bottom: 1px solid #3f3f46 !important;
    color: #fafafa !important;
  }
  header[data-testid="stHeader"] button,
  header[data-testid="stHeader"] a {
    color: #ffffff !important;
  }
  header[data-testid="stHeader"] [data-testid="stIconMaterial"],
  header[data-testid="stHeader"] [data-testid="stIconEmoji"] {
    color: #ffffff !important;
  }
  header[data-testid="stHeader"] svg {
    color: #ffffff !important;
    opacity: 1 !important;
  }
  header[data-testid="stHeader"] svg path,
  header[data-testid="stHeader"] svg rect,
  header[data-testid="stHeader"] svg circle,
  header[data-testid="stHeader"] svg line,
  header[data-testid="stHeader"] svg polyline,
  header[data-testid="stHeader"] svg polygon {
    fill: currentColor !important;
    stroke: currentColor !important;
    opacity: 1 !important;
  }

  div[data-testid="stToolbar"] {
    color: #fafafa !important;
  }
  div[data-testid="stToolbar"] button,
  div[data-testid="stToolbar"] a {
    color: #ffffff !important;
  }
  div[data-testid="stToolbar"] [data-testid="stIconMaterial"] {
    color: #ffffff !important;
  }
  div[data-testid="stToolbar"] svg {
    color: #ffffff !important;
    opacity: 1 !important;
  }
  div[data-testid="stToolbar"] svg path,
  div[data-testid="stToolbar"] svg rect {
    fill: currentColor !important;
    stroke: currentColor !important;
    opacity: 1 !important;
  }

  /* Üst dekor şeridi — bazen ikonlar bu bölümle çakışıyor */
  div[data-testid="stDecoration"] {
    visibility: visible !important;
    opacity: 1 !important;
  }

  /* Deploy + bozuk ikon — config toolbarMode=viewer birincil; CSS yedek */
  [data-testid="stDeployButton"],
  [data-testid="stMainMenuDeploy"],
  button[aria-label="Deploy"],
  a[aria-label="Deploy"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
  }
  header[data-testid="stHeader"] a[href*="streamlit.io/deploy"],
  header[data-testid="stHeader"] a[href*="share.streamlit.io"],
  header[data-testid="stHeader"] a[href*="/deploy"] {
    display: none !important;
  }

  .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 920px;
  }

  .hero-title {
    font-size: clamp(1.75rem, 4vw, 2.35rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 0.35rem 0;
    color: #ffffff !important;
    background: none !important;
    -webkit-text-fill-color: #ffffff !important;
    background-clip: border-box !important;
  }

  .hero-sub {
    color: #ffffff !important;
    font-size: 1.05rem;
    margin: 0 0 1.75rem 0;
    line-height: 1.55;
  }

  .hero-sub strong {
    color: #ffffff !important;
    font-weight: 600;
  }

  /* Kenar çubuğu (navbar) */
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] > div,
  div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #1a1a1c 0%, #27272a 45%, #2d2d32 100%) !important;
    border-right: 1px solid #3f3f46 !important;
    color: #e4e4e7 !important;
  }

  section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
    margin: 0.75rem 0 !important;
  }

  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] li,
  section[data-testid="stSidebar"] span:not(button span),
  section[data-testid="stSidebar"] .stMarkdown,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
  section[data-testid="stSidebar"] [data-testid="stCaption"],
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #e4e4e7 !important;
  }

  section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #3f3f46 !important;
    border: 1px solid #52525b !important;
    border-radius: 10px !important;
  }

  section[data-testid="stSidebar"] [data-testid="stExpander"] p,
  section[data-testid="stSidebar"] [data-testid="stExpander"] li,
  section[data-testid="stSidebar"] [data-testid="stExpander"] strong,
  section[data-testid="stSidebar"] [data-testid="stExpander"] span,
  section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
  section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
    color: #e4e4e7 !important;
  }

  section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: rgba(63, 63, 70, 0.85) !important;
    border: 1px solid #71717a !important;
  }
  section[data-testid="stSidebar"] [data-testid="stAlert"] p,
  section[data-testid="stSidebar"] [data-testid="stAlert"] div {
    color: #f4f4f5 !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] > div,
  section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background-color: #3f3f46 !important;
    border-color: #71717a !important;
    color: #fafafa !important;
  }

  section[data-testid="stSidebar"] [data-testid="stIconMaterial"],
  section[data-testid="stSidebar"] [data-testid="stIconEmoji"] {
    color: #fafafa !important;
    opacity: 1 !important;
  }
  section[data-testid="stSidebar"] button {
    color: #fafafa !important;
  }
  section[data-testid="stSidebar"] button [data-testid="stIconMaterial"] {
    color: #fafafa !important;
  }
  section[data-testid="stSidebar"] svg {
    color: #fafafa !important;
    opacity: 1 !important;
  }
  section[data-testid="stSidebar"] svg path:not([fill="none"]),
  section[data-testid="stSidebar"] svg rect:not([fill="none"]),
  section[data-testid="stSidebar"] svg circle {
    fill: currentColor !important;
    opacity: 1 !important;
  }
  section[data-testid="stSidebar"] svg path[fill="none"] {
    stroke: currentColor !important;
    opacity: 1 !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] {
    color: #fafafa !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #e4e4e7 !important;
  }

  .verdict-ham {
    border-radius: 16px;
    padding: 1.35rem 1.5rem;
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #6ee7b7;
    margin: 0.5rem 0 1rem 0;
  }

  .verdict-spam {
    border-radius: 16px;
    padding: 1.35rem 1.5rem;
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 1px solid #fda4af;
    margin: 0.5rem 0 1rem 0;
  }

  .verdict-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 0.35rem;
  }

  .verdict-main {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0;
  }

  .verdict-ham .verdict-main { color: #047857; }
  .verdict-spam .verdict-main { color: #be123c; }

  .metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.75rem;
  }

  .pill-hint {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.75rem;
  }

  /* Ana alan expander — çerçeve */
  section[data-testid="stMain"] [data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 12px;
  }

  /* Mesaj metin kutusu */
  section[data-testid="stMain"] [data-testid="stTextArea"] textarea {
    background-color: #1c1917 !important;
    color: #fafaf9 !important;
    border: 1px solid rgba(185, 28, 28, 0.65) !important;
    border-radius: 10px !important;
    caret-color: #fecaca !important;
    box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.25);
  }

  section[data-testid="stMain"] [data-testid="stTextArea"] textarea:focus {
    border-color: #b91c1c !important;
    box-shadow: 0 0 0 1px rgba(185, 28, 28, 0.45);
    outline: none !important;
  }

  section[data-testid="stMain"] [data-testid="stTextArea"] textarea::placeholder {
    color: #a8a29e !important;
    opacity: 1 !important;
  }

  section[data-testid="stMain"] [data-testid="stTextArea"] [data-baseweb="textarea"] {
    background-color: #1c1917 !important;
    border-color: transparent !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _artifact_choices() -> list[tuple[str, str]]:
    base = MODELS_DIR
    opts: list[tuple[str, str]] = []
    if (base / "pipeline_logreg.joblib").is_file():
        opts.append(("Logistic Regression (TF-IDF)", "pipeline_logreg.joblib"))
    if (base / "pipeline_rf.joblib").is_file():
        opts.append(("Random Forest (TF-IDF + SVD)", "pipeline_rf.joblib"))
    if (base / "pipeline_artifacts.joblib").is_file():
        opts.insert(0, ("Varsayılan (en son eğitim)", "pipeline_artifacts.joblib"))
    return opts


@st.cache_resource
def load_model(artifact_file: str) -> SmsClassifierPipeline:
    return SmsClassifierPipeline.load(MODELS_DIR, artifact_file)


def main() -> None:
    st.set_page_config(
        page_title="SMS Güvenlik — Spam Tespiti",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.sidebar.markdown("### Ayarlar")
    st.sidebar.markdown("---")

    choices = _artifact_choices()
    if not choices:
        st.markdown('<p class="hero-title">SMS Spam Tespiti</p>', unsafe_allow_html=True)
        st.error(
            "**Model bulunamadı.** Önce eğitim çalıştırın:\n\n"
            "```bash\npython train.py --data data/spam.csv --text-col v2 --label-col v1\n```\n\n"
            "Çıktı: `models/pipeline_artifacts.joblib`"
        )
        return

    names = [c[0] for c in choices]
    files = [c[1] for c in choices]
    artifact_file = st.sidebar.selectbox("Sınıflandırıcı modeli", options=files, format_func=lambda f: names[files.index(f)])

    try:
        pipe = load_model(artifact_file)
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return

    st.sidebar.markdown("---")
    st.sidebar.caption("Aktif model")
    st.sidebar.info(pipe.model_type)

    with st.sidebar.expander("Bu uygulama ne yapar?"):
        st.markdown(
            """
<div style="color:#d4d4d8;line-height:1.65;font-size:0.92rem;">
<p style="margin:0 0 0.75rem 0;"><strong style="color:#fff;">TF-IDF</strong> ile metin vektörleştirilir;
<strong>Logistic Regression</strong> veya <strong>Random Forest</strong> ile
<strong>ham (0)</strong> / <strong>spam (1)</strong> tahmini üretilir.</p>
<p style="margin:0;">Sonuçtaki yüzde, modelin <strong>spam</strong> sınıfına verdiği olasılıktır;
eğitim verisine bağlıdır.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    # —— Ana içerik ——
    col_main, col_gap = st.columns([1, 0.02])
    with col_main:
        st.markdown('<p class="hero-title">SMS Spam Tespiti</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Mesajınızı aşağıya yazın veya dosya yükleyin. '
            "Tek tıkla <strong>ham</strong> mi <strong>spam</strong> mi olduğunu görün.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("##### Hızlı örnek")
        ec1, ec2, ec3 = st.columns([1, 1, 1])
        with ec1:
            if st.button("Örnek: normal SMS", use_container_width=True):
                st.session_state["sms_input"] = EXAMPLE_HAM
                st.rerun()
        with ec2:
            if st.button("Örnek: şüpheli SMS", use_container_width=True):
                st.session_state["sms_input"] = EXAMPLE_SPAM
                st.rerun()
        with ec3:
            if st.button("Metni temizle", use_container_width=True):
                st.session_state["sms_input"] = ""
                st.rerun()

        if "sms_input" not in st.session_state:
            st.session_state["sms_input"] = ""

        tab1, tab2 = st.tabs(["Yaz", "Dosyadan yükle"])
        text = ""
        with tab1:
            text = st.text_area(
                "Mesaj",
                height=220,
                placeholder="Örn: Free entry in 2 a wkly comp — Text WIN to 87121…",
                label_visibility="collapsed",
                key="sms_input",
            )
        with tab2:
            uploaded = st.file_uploader("`.txt` dosyası seçin", type=["txt"], label_visibility="visible")
            if uploaded is not None:
                text = uploaded.read().decode("utf-8", errors="replace")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("Analiz et", type="primary", use_container_width=True)

        if analyze:
            if not text or not str(text).strip():
                st.warning("Lütfen bir metin girin veya dosya yükleyin.")
            else:
                label, spam_prob = pipe.predict(text)
                st.markdown("---")
                st.markdown("##### Sonuç")

                if label.startswith("Belirsiz"):
                    st.info(label)
                else:
                    is_spam = label == pipe.label_names[1]
                    css_class = "verdict-spam" if is_spam else "verdict-ham"
                    nice = "Spam riski" if is_spam else "Normal mesaj (ham)"
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<div class="verdict-label">Tahmin</div>'
                        f'<p class="verdict-main">{nice}</p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Spam olasılığı", f"{spam_prob * 100:.1f}%", help="Modelin spam sınıfına verdiği olasılık")
                    with m2:
                        st.metric("Ham olasılığı", f"{(1 - spam_prob) * 100:.1f}%")

                    st.progress(min(1.0, max(0.0, spam_prob)), text=f"Spam skoru: {spam_prob * 100:.0f} / 100")

                    with st.expander("Ham metin önizlemesi"):
                        preview = text.strip()[:500] + ("…" if len(text.strip()) > 500 else "")
                        st.code(preview, language=None)

                st.markdown('<p class="pill-hint">Eğitim verisi ve modele göre sonuçlar değişebilir; hukuki / finansal karar için yeterli değildir.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("SMS Spam Tespiti · TF-IDF + makine öğrenmesi · Öğrenci projesi")


if __name__ == "__main__":
    main()
