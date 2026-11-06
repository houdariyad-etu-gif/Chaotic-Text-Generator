"""
Lab 03 : Chaotic Text Generator
Premium Streamlit test bench for POS-HMM, role-split HMM and trigram generation.
Run from the project root:
    python -m streamlit run app.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import (
    HMM,
    NGramGenerator,
    generation_metrics,
    load_tagged,
    split_by_document,
)

# ---------------------------------------------------------------------------
# Page configuration and visual identity
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chaotic Text Generator | NLP Lab 03",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
REAL_RESULTS_PATH = RESULTS_DIR / "ewt_results.json"
FAILURES_PATH = RESULTS_DIR / "failures.txt"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #17211f;
      --muted: #66736f;
      --paper: #f6f7f4;
      --panel: #ffffff;
      --line: #e4e9e4;
      --green: #176b52;
      --green-soft: #e7f2ed;
      --gold: #bd8b43;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(246,247,244,.88); }
    [data-testid="stSidebar"] {
      background: #17211f;
      border-right: 1px solid #263631;
    }
    [data-testid="stSidebar"] * { color: #edf4ef !important; }
    [data-testid="stSidebar"] .stCaption { color: #a9bbb2 !important; }
    .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: #17211f; }
    h1 { font-size: clamp(2.2rem, 4vw, 3.45rem) !important; font-weight: 800 !important; }
    h2 { font-weight: 750 !important; }
    p, li { color: #40504a; }
    
    /* Correction du texte invisible dans les messages d'alerte et st.info */
    [data-testid="stAlert"] {
      background-color: #172b25 !important;
      border: 1px solid #28624d !important;
      border-radius: 10px !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] li,
    [data-testid="stAlert"] div {
      color: #e6f4ee !important;
      font-weight: 500 !important;
    }

    .eyebrow {
      text-transform: uppercase; letter-spacing: .17em; font-size: .68rem;
      font-weight: 700; color: var(--green); margin-bottom: .55rem;
    }
    .hero {
      background: linear-gradient(115deg, #172b25 0%, #1e4739 58%, #28624d 100%);
      color: #f4f7f3; padding: 2.1rem 2.35rem; border-radius: 22px;
      position: relative; overflow: hidden; margin: .2rem 0 1.7rem 0;
      box-shadow: 0 16px 42px rgba(23,43,37,.12);
    }
    .hero:after {
      content: "✳"; position: absolute; right: 4%; top: -48px;
      font-size: 230px; color: rgba(255,255,255,.055); line-height: 1;
    }
    .hero-kicker { color: #b9d5c7; font-size: .72rem; letter-spacing: .19em; text-transform: uppercase; font-weight: 700; }
    .hero-title { color: #fff; font: 800 clamp(2rem,4vw,3.1rem)/1.08 'Manrope',sans-serif; margin: .65rem 0; letter-spacing: -.045em; }
    .hero-copy { color: #d6e4dc; max-width: 760px; font-size: 1rem; line-height: 1.65; }
    .hero-meta { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.2rem; }
    .hero-pill { border:1px solid rgba(255,255,255,.22); color:#eaf4ed; border-radius:99px; padding:.35rem .7rem; font-size:.72rem; }
    .section-kicker { color:var(--green); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:800; margin:1.2rem 0 .35rem; }
    .section-title { font:750 1.45rem 'Manrope',sans-serif; color:#17211f; margin:0 0 .25rem; letter-spacing:-.03em; }
    .section-copy { color:#66736f; margin:0 0 1rem; line-height:1.6; }
    .metric-card {
      background:#fff; border:1px solid var(--line); border-radius:16px;
      padding:1rem 1.1rem; min-height:116px; box-shadow:0 4px 16px rgba(23,43,37,.035);
    }
    .metric-label { color:#73817a; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
    .metric-value { font:800 1.8rem 'Manrope',sans-serif; color:#1c3d31; margin:.28rem 0 .12rem; }
    .metric-note { color:#7a8580; font-size:.72rem; line-height:1.4; }
    .info-card {
      background:#fff; border:1px solid var(--line); border-radius:16px;
      padding:1.1rem 1.2rem; height:100%;
    }
    .info-card h4 { font:750 1rem 'Manrope',sans-serif; color:#1e3c31; margin:0 0 .45rem; }
    .info-card p { color:#63716b; font-size:.88rem; line-height:1.58; margin:0; }
    .tag {
      display:inline-block; font:500 .75rem 'DM Mono',monospace; color:#175b46;
      background:#e8f3ed; border:1px solid #d6e9df; border-radius:6px;
      padding:.22rem .42rem; margin:.12rem .1rem .12rem 0;
    }
    .word-chip {
      display:inline-flex; flex-direction:column; align-items:center; gap:.2rem;
      background:#fff; border:1px solid #dce6df; border-radius:10px;
      padding:.5rem .65rem; margin:.2rem .25rem .2rem 0; min-width:62px;
    }
    .word-chip b { color:#1c3028; font-size:.88rem; }
    .word-chip small { color:#176b52; font:500 .65rem 'DM Mono',monospace; }
    .callout {
      border-left:3px solid #bd8b43; background:#fbf7ef; padding:.85rem 1rem;
      border-radius:0 10px 10px 0; color:#665538; font-size:.88rem; line-height:1.55;
    }

    /* Style des boutons */
    .stButton > button,
    .stDownloadButton > button {
      color:#17352b !important;
      background:#ffffff !important;
      border:1px solid #b9c9bf !important;
      border-radius:10px !important;
      padding:.62rem 1.1rem !important;
      font-weight:700 !important;
      box-shadow:0 4px 12px rgba(23,43,37,.07) !important;
    }
    .stButton > button p,
    .stDownloadButton > button p,
    .stButton > button span,
    .stDownloadButton > button span {
      color:inherit !important;
      font-weight:700 !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:focus,
    .stButton > button[kind="primary"]:active {
      color:#ffffff !important;
      background:#176b52 !important;
      border:1px solid #176b52 !important;
    }
    .stButton > button[kind="secondary"],
    .stDownloadButton > button {
      color:#17352b !important;
      background:#ffffff !important;
      border:1px solid #9aafa3 !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button:hover {
      color:#123f30 !important;
      background:#e7f2ed !important;
      border-color:#176b52 !important;
    }
    .stButton > button[kind="primary"]:hover {
      background:#125640 !important;
      border-color:#125640 !important;
    }
    
    [data-testid="stTabs"] button {
      color:#40504a !important;
      font-weight:700 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
      color:#145b43 !important;
    }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    [data-testid="stCaptionContainer"] p { color:#66736f !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color:#40504a; }
    hr { border-color:#e4e9e4; }
    .small-note { color:#84918b; font-size:.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Estimating model probabilities from the training corpus…")
def load_models():
    docs = load_tagged(str(ROOT / "data" / "corpus.csv"))
    train, test = split_by_document(docs)
    plain = HMM(train)
    role_split = HMM(train, split_states=True)
    trigram = NGramGenerator([words for words, _ in train])
    return train, test, plain, role_split, trigram


try:
    train_sentences, test_sentences, hmm_plain, hmm_role, trigram = load_models()
except Exception as exc:
    st.error("The models could not be loaded. Check that data/corpus.csv and src/pipeline.py are present.")
    st.exception(exc)
    st.stop()

# Exact sentence matches are checked against the training split only.
# This is separate from novelty, which is already calculated by generation_metrics.
TRAIN_SENTENCE_SET = {" ".join(words) for words, _ in train_sentences}


MODEL_LABELS = {
    "Plain HMM": "plain",
    "Role-split HMM": "role",
    "Trigram": "trigram",
}
MODEL_DESCRIPTIONS = {
    "Plain HMM": "A model that first chooses word types, such as noun or verb, then chooses words that match them.",
    "Role-split HMM": "A version of the first model that also keeps track of a word's role in the sentence. This helps it follow simple sentence patterns.",
    "Trigram": "A model that chooses the next word by looking at the two words before it. If it has not seen that pair, it uses a shorter history.",
}


def model_for(label):
    return {"Plain HMM": hmm_plain, "Role-split HMM": hmm_role, "Trigram": trigram}[label]


def as_percent(value):
    return f"{100 * float(value):.1f}%"


def generate_one(label, rng, temperature):
    """Return (words, displayed hidden/predicted tags, metric tags)."""
    if label == "Plain HMM":
        words, states = hmm_plain.sample_with_states(rng, temperature)
        return words, states, [s.split("_")[0] for s in states]
    if label == "Role-split HMM":
        words, states = hmm_role.sample_with_states(rng, temperature)
        return words, states, [s.split("_")[0] for s in states]
    words = trigram.sample_sentence(rng, temperature)
    predicted = hmm_plain.viterbi(words)
    return words, predicted, predicted


def metric_for_batch(label, rows):
    if label == "Plain HMM":
        tagged = [(r["words"], r["metric_tags"]) for r in rows]
        return generation_metrics(tagged, train_sentences, hmm_plain)
    if label == "Role-split HMM":
        tagged = [(r["words"], r["metric_tags"]) for r in rows]
        return generation_metrics(tagged, train_sentences, hmm_plain)
    tagged = [(r["words"], None) for r in rows]
    return generation_metrics(tagged, train_sentences, hmm_plain)


def make_batch(label, count, temperature, seed):
    rng = random.Random(int(seed))
    rows = []
    for index in range(count):
        words, visible_tags, metric_tags = generate_one(label, rng, temperature)
        rows.append({
            "id": index + 1,
            "sentence": " ".join(words),
            "words": words,
            "tags_display": visible_tags,
            "metric_tags": metric_tags,
        })
    return rows


def load_saved_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_real_results():
    if not REAL_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(REAL_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def compute_temperature_curve():
    """Compute a reproducible 500-sentence curve using the lab's evaluation seed."""
    cached = load_saved_metrics()
    temps = [0.5, 1.0, 1.5, 2.5]
    if cached and "generation" in cached:
        records = []
        keys = [("Plain HMM", "hmm"), ("Role-split HMM", "hmm_split"), ("Trigram", "ngram")]
        for label, key in keys:
            for temp in temps:
                item = cached["generation"][key].get(str(temp))
                if item:
                    records.append({
                        "Model": label, "Temperature": temp,
                        "Valid word-type pattern": item["valid_tag_pattern"],
                        "Subject and verb fit": item["plausible_subject_verb"],
                        "New sentences": item["novel_sentences"],
                        "Word-pair variety": item["distinct_2"],
                    })
        if len(records) == 12:
            return pd.DataFrame(records), "results/metrics.json (500 generated sentences per setting)"
    records = []
    for label in MODEL_LABELS:
        for temp in temps:
            rows = make_batch(label, 500, temp, 42)
            m = metric_for_batch(label, rows)
            records.append({
                "Model": label, "Temperature": temp,
                "Valid word-type pattern": m["valid_tag_pattern"],
                "Subject and verb fit": m["plausible_subject_verb"],
                "New sentences": m["novel_sentences"], "Word-pair variety": m["distinct_2"],
            })
    return pd.DataFrame(records), "on-demand evaluation (500 generated sentences per setting)"


def render_metric_card(label, value, note):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: controls shared by the Lab tab
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✦ CHAOTIC TEXT")
    st.caption("NLP LAB 03 · GENERATIVE SEQUENCE MODELS")
    st.markdown("---")
    st.markdown("### Experiment settings")
    selected_model = st.selectbox(
        "Generation model",
        list(MODEL_LABELS.keys()),
        index=1,
        help="Choose which of the three models will create the sentences.",
    )
    temperature = st.slider(
        "Temperature", min_value=0.3, max_value=3.0, value=1.0, step=0.1,
        help="Lower values usually give more predictable sentences. Higher values allow more unusual choices.",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, step=1)
    sentence_count = st.slider("Number of sentences", 10, 500, 100, step=10)
    st.markdown("---")
    st.markdown("### Corpus snapshot")
    st.metric("Training sentences", f"{len(train_sentences):,}")
    st.metric("Test sentences", f"{len(test_sentences):,}")
    st.metric("Known POS tags", len(hmm_plain.tags))
    st.caption("Document-level split · synthetic English corpus")
    st.markdown("---")
    st.caption("Models and algorithms supplied with Lab 03. The interface adds experiment controls, visual analysis and tagging.")


# ---------------------------------------------------------------------------
# Header / overview
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Natural Language Processing · Laboratory 03</div>
      <div class="hero-title">Chaotic Text Generator</div>
      <div class="hero-copy">
        An interactive test bench for probabilistic language generation.
        Compare grammatical structure, lexical diversity and memorisation :
        then reverse the process with Viterbi sequence tagging.
      </div>
      <div class="hero-meta">
        <span class="hero-pill">Hidden Markov Models</span>
        <span class="hero-pill">Role-split states</span>
        <span class="hero-pill">Trigram baseline</span>
        <span class="hero-pill">Viterbi decoding</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Generation lab", "Viterbi tagger", "Model insights", "Real-data evaluation"])


# ---------------------------------------------------------------------------
# TAB 1 : Explain the scientific problem and model mechanisms
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-kicker">01 / The experiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How do we check the results?</div>', unsafe_allow_html=True)
    st.markdown(
        "A generated sentence has no single reference answer. Instead of judging a handful of examples, "
        "this lab evaluates batches of hundreds of sentences using complementary automatic metrics.",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="info-card"><h4>01 · Plain HMM</h4>
            <p>Hidden states are POS tags. The model samples a tag sequence from start and transition
            probabilities, then emits one word from each tag's vocabulary.</p></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="info-card"><h4>02 · Role-split HMM</h4>
            <p>Some states are enriched with a sentence role: S (before the verb), O (after the verb),
            or P (inside a prepositional phrase). This gives a first-order model extra structural memory.</p></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="info-card"><h4>03 · Trigram</h4>
            <p>A word-level baseline. It predicts a token from the two previous words and backs off to
            a bigram or unigram when a context has not been observed.</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### Two directions: generation and tagging")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """<div class="info-card"><h4>Generation · word types → words</h4>
            <p><span class="tag">DET</span> <span class="tag">NOUN</span> <span class="tag">VERB</span>
            <span class="tag">DET</span> <span class="tag">NOUN</span> <span class="tag">PUNCT</span><br><br>
            First choose the type of each word. Then choose the actual words.</p></div>""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """<div class="info-card"><h4>Viterbi · words → most likely tags</h4>
            <p><b>“The student reads a book.”</b><br>
            The decoder searches for the most probable hidden-state path using dynamic programming
            and log-probabilities.</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### The four evaluation lenses")
    metric_cols = st.columns(4)
    metric_info = [
        ("Valid word-type pattern", "How many generated sentences use a word-type pattern that appeared in the training examples."),
        ("Subject and verb fit", "How many sentences begin with a subject and verb combination seen in the training examples."),
        ("New sentences", "How many generated sentences are not exact copies of sentences used for training."),
        ("Word-pair variety", "A diversity score. It checks how many different pairs of neighbouring words appear in the generated text."),
    ]
    for col, (name, desc) in zip(metric_cols, metric_info):
        with col:
            st.markdown(
                f'<div class="info-card"><h4>{name}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.markdown(
        '<div class="callout"><b>Interpretation rule.</b> No single metric defines “good text”. '
        'A model can produce valid grammatical patterns while pairing words with implausible meanings. '
        'Higher diversity can also come with more structural errors.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Checking for memorisation")
    st.markdown(
        "A generated sentence is marked as copied when its complete word sequence is identical "
        "to a sentence in the training set. This is an exact-match check. It does not detect "
        "paraphrases or sentences that reuse only part of a training example. The novelty score "
        "summarises the share of sentences that are not exact copies, while the generation table "
        "lets us inspect each sentence individually."
    )


# ---------------------------------------------------------------------------
# TAB 2 : Interactive generation and batch metrics
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-kicker">02 / Interactive workbench</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Create sentences and compare models</div>', unsafe_allow_html=True)
    st.markdown(
        f"Current model: **{selected_model}** : {MODEL_DESCRIPTIONS[selected_model]}",
        unsafe_allow_html=True,
    )
    a, b, c = st.columns([1, 1, 2])
    a.markdown(f"**Temperature**  `{temperature:.1f}`")
    b.markdown(f"**Seed**  `{int(seed)}`")
    c.markdown(f"**Batch size**  `{sentence_count}` sentences")
    generate_col, compare_col = st.columns([1, 1])
    with generate_col:
        run_generation = st.button("✦ Generate batch", type="primary", use_container_width=True)
    with compare_col:
        run_comparison = st.button("Compare all three models", use_container_width=True)

    if run_generation:
        with st.spinner(f"Generating {sentence_count} sentences with {selected_model}…"):
            st.session_state["generated_rows"] = make_batch(
                selected_model, sentence_count, temperature, seed
            )
            st.session_state["generated_model"] = selected_model
            st.session_state["generated_temp"] = temperature
            st.session_state["generated_seed"] = int(seed)

    if run_comparison:
        comparison = []
        with st.spinner("Generating matched batches for all three models…"):
            for label in MODEL_LABELS:
                batch = make_batch(label, sentence_count, temperature, seed)
                m = metric_for_batch(label, batch)
                comparison.append({
                    "Model": label,
                    "Valid word-type pattern": m["valid_tag_pattern"],
                    "Subject and verb fit": m["plausible_subject_verb"],
                    "New sentences": m["novel_sentences"],
                    "Word-pair variety": m["distinct_2"],
                })
        st.session_state["comparison_rows"] = comparison

    rows = st.session_state.get("generated_rows")
    if rows:
        m = metric_for_batch(st.session_state["generated_model"], rows)
        st.markdown("---")
        st.markdown(
            f"#### Batch results · {st.session_state['generated_model']} · "
            f"T = {st.session_state['generated_temp']:.1f}"
        )
        metric_cols = st.columns(4)
        cards = [
            ("Valid word-type pattern", as_percent(m["valid_tag_pattern"]), "Structural conformity to the synthetic grammar"),
            ("Subject and verb fit", as_percent(m["plausible_subject_verb"]), "Observed first noun-verb pairs"),
            ("New sentences", as_percent(m["novel_sentences"]), "Not an exact training-sentence copy"),
            ("Word-pair variety", f"{m['distinct_2']:.3f}", "Unique bigrams / all bigrams"),
        ]
        for col, (label, value, note) in zip(metric_cols, cards):
            with col:
                render_metric_card(label, value, note)

        train_tag_patterns = {" ".join(tags) for _, tags in train_sentences}
        table_rows = []
        for r in rows:
            is_copy = r["sentence"] in TRAIN_SENTENCE_SET
            table_rows.append({
                "#": r["id"],
                "Generated sentence": r["sentence"],
                "Hidden / predicted tags": "  ".join(r["tags_display"]),
                "Valid pattern": "Yes" if " ".join(r["metric_tags"]) in train_tag_patterns else "No",
                "Copied from training": "Yes" if is_copy else "No",
            })
        copied_count = sum(r["sentence"] in TRAIN_SENTENCE_SET for r in rows)
        if copied_count:
            st.warning(
                f"{copied_count} of {len(rows)} generated sentences are exact copies "
                "of sentences in the training set. This is a sign of memorisation."
            )
        else:
            st.success("No exact copies of training sentences were found in this batch.")

        st.markdown("#### Generated sentences and their word types")
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
            height=430,
            column_config={
                "#": st.column_config.NumberColumn(
                    "#",
                    width="small"
                ),
                "Generated sentence": st.column_config.TextColumn(
                    "Generated sentence",
                    width="medium"
                ),
                "Hidden / predicted tags": st.column_config.TextColumn(
                    "Hidden / predicted tags",
                    width="medium"
                ),
                "Valid pattern": st.column_config.TextColumn(
                    "Valid pattern",
                    width="small"
                ),
                "Copied from training": st.column_config.TextColumn(
                    "Copied from training",
                    width="medium"
                ),
            },
        )

        csv_df = pd.DataFrame([
            {
                "sentence": r["sentence"],
                "tags": " ".join(r["tags_display"]),
                "valid_tag_pattern": "yes" if " ".join(r["metric_tags"]) in train_tag_patterns else "no",
                "copied_from_training": "yes" if r["sentence"] in TRAIN_SENTENCE_SET else "no",
            }
            for r in rows
        ])
        st.download_button(
            "↓ Download generated batch (CSV)",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name=f"chaotic_text_{MODEL_LABELS[st.session_state['generated_model']]}_T{st.session_state['generated_temp']:.1f}.csv",
            mime="text/csv",
        )
        st.caption(
            "For HMMs, the displayed sequence shows sampled hidden states (role suffixes are retained for "
            "the role-split model). For Trigram, tags are inferred afterwards with the plain HMM Viterbi decoder."
        )
    else:
        st.info("Choose the experiment settings in the sidebar, then click **Generate batch** to inspect the output.")

    comparison = st.session_state.get("comparison_rows")
    if comparison:
        st.markdown("---")
        st.markdown("#### Matched model comparison")
        st.caption("Same batch size, temperature and seed. Each model uses its own sampling process.")
        comp_df = pd.DataFrame(comparison)
        display_df = comp_df.copy()
        for metric in ["Valid word-type pattern", "Subject and verb fit", "New sentences"]:
            display_df[metric] = display_df[metric].map(lambda x: f"{x:.1%}")
        display_df["Word-pair variety"] = display_df["Word-pair variety"].map(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.bar_chart(comp_df.set_index("Model")[[
            "Valid word-type pattern", "Subject and verb fit", "New sentences"
        ]], y_label="Score (0-1)")


# ---------------------------------------------------------------------------
# TAB 3 : Viterbi tagger with explicit OOV warning
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-kicker">03 / Sequence labelling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Word tagger (Viterbi)</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter an English sentence. The HMM decoder finds the most probable POS-tag sequence "
        "for the observed words. Words outside the training vocabulary are highlighted as OOV.",
        unsafe_allow_html=True,
    )
    user_sentence = st.text_input(
        "Sentence to analyse",
        value="the student reads a book .",
        placeholder="Type a short English sentence…",
    )
    tag_model_label = st.radio(
        "Tagging model", ["Plain HMM", "Role-split HMM"], horizontal=True,
        help="Both models return the coarse POS labels after decoding.",
    )
    if st.button("Analyse sentence with Viterbi", type="primary"):
        words = user_sentence.lower().split()
        tagger = hmm_plain if tag_model_label == "Plain HMM" else hmm_role
        if not words:
            st.warning("Enter at least one word.")
        else:
            predicted_tags = tagger.viterbi(words)
            oov_words = sorted({w for w in words if w not in tagger.vocab})
            st.markdown("#### Word-by-word prediction")
            chips = ""
            for word, tag in zip(words, predicted_tags):
                is_oov = word not in tagger.vocab
                border = "#d99a55" if is_oov else "#dce6df"
                bg = "#fff5e9" if is_oov else "#ffffff"
                chips += (
                    f'<span class="word-chip" style="border-color:{border};background:{bg}">'
                    f'<b>{word}</b><small>{tag}</small></span>'
                )
            st.markdown(chips, unsafe_allow_html=True)
            if oov_words:
                st.warning(
                    "Words not seen during training: " + ", ".join(oov_words)
                    + ". Add-k smoothing keeps decoding possible, but the predicted tag for an unseen "
                    "word is less reliable."
                )
            else:
                st.success("All words are present in the training vocabulary.")
            st.caption(
                "Viterbi returns the most probable tag path, not a guarantee of linguistic correctness. "
                "The output is a prediction for the whole sentence."
            )
            st.session_state["last_tagging"] = {
                "sentence": user_sentence, "words": words, "tags": predicted_tags,
                "model": tag_model_label, "oov": oov_words,
            }
    previous = st.session_state.get("last_tagging")
    if previous:
        st.markdown("---")
        st.markdown("#### Decoding summary")
        st.write(f"**Input:** {previous['sentence']}")
        st.write(f"**Model:** {previous['model']}")
        st.write("**Predicted path:** " + " → ".join(previous["tags"]))


# ---------------------------------------------------------------------------
# TAB 4 : Temperature curves, transition heatmap, failure analysis
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-kicker">04 / How the models behave</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What changes when the model gets more chaotic?</div>', unsafe_allow_html=True)
    st.markdown(
        "The temperature reshapes sampling weights using count^(1/T). Lower temperatures favour "
        "frequent events; higher temperatures make less frequent events more competitive.",
        unsafe_allow_html=True,
    )
    if st.button("Refresh temperature analysis", key="refresh_curve"):
        st.cache_data.clear()
        st.session_state.pop("temperature_curve", None)
    if "temperature_curve" not in st.session_state:
        with st.spinner("Preparing the 12 temperature/model evaluations…"):
            curve_df, source_label = compute_temperature_curve()
            st.session_state["temperature_curve"] = curve_df
            st.session_state["curve_source"] = source_label
    curve_df = st.session_state["temperature_curve"]
    st.caption(f"Source: {st.session_state['curve_source']}")

    st.markdown("#### Professor's official benchmark at T = 1.0")
    official = load_saved_metrics()
    if official and isinstance(official.get("generation"), dict):
        official_keys = [
            ("Plain HMM", "hmm"),
            ("Role-split HMM", "hmm_split"),
            ("Trigram", "ngram"),
        ]
        official_rows = []
        for model_name, result_key in official_keys:
            result = official["generation"].get(result_key, {}).get("1.0")
            if result is not None:
                official_rows.append({
                    "Model": model_name,
                    "Valid word-type pattern": as_percent(result["valid_tag_pattern"]),
                    "Subject and verb fit": as_percent(result["plausible_subject_verb"]),
                    "New sentences": as_percent(result["novel_sentences"]),
                    "Word-pair variety": f"{result['distinct_2']:.3f}",
                })
        if len(official_rows) == 3:
            st.dataframe(pd.DataFrame(official_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Source: results/metrics.json, generated by the professor's evaluation script. "
                "Each model was evaluated on 500 generated sentences at this temperature."
            )
        else:
            st.warning("The official T = 1.0 results are incomplete in results/metrics.json.")
    else:
        st.warning(
            "Official benchmark results are not available. Run `python -m src.evaluate` "
            "from the project root to create results/metrics.json."
        )

    st.markdown(
        '<div class="callout"><b>Keep the results separate.</b> The table above contains the '
        'professor\'s saved benchmark. The interactive batches below are new experiments '
        'using the settings selected in the sidebar. Their results may differ.</div>',
        unsafe_allow_html=True,
    )

    selected_metric = st.selectbox(
        "Metric to plot",
        ["Valid word-type pattern", "Subject and verb fit", "New sentences", "Word-pair variety"],
        index=0,
    )
    plot_df = curve_df.pivot(index="Temperature", columns="Model", values=selected_metric)
    st.line_chart(plot_df, y_label=selected_metric)
    st.markdown(
        '<div class="callout"><b>How to read this chart.</b> Compare the direction of each curve, '
        'not just its highest point. For example, increased temperature can raise diversity while '
        'reducing the share of structurally valid outputs.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### What word type can come next?")
    st.caption("This table shows which word type is likely to come next. For example, after a DET (a word like 'the'), a NOUN may be common.")
    tags, matrix = hmm_plain.transition_matrix()
    heatmap_df = pd.DataFrame(matrix, index=tags, columns=tags)
    st.dataframe(
        heatmap_df.style.format("{:.2f}").background_gradient(cmap="Greens", axis=None),
        use_container_width=True,
    )
    st.caption("Rows = current tag · columns = next tag. This is a row-normalised transition-count matrix.")

    st.markdown("---")
    st.markdown("#### When can a model make mistakes?")
    if FAILURES_PATH.exists():
        failure_text = FAILURES_PATH.read_text(encoding="utf-8")
        st.code(failure_text, language="text")
    else:
        st.info("Run `python -m src.evaluate` to generate results/failures.txt.")
    st.markdown(
        """
        - **Structure:** the predicted/generated POS sequence does not match one of the training patterns.
        - **Meaning:** the tag pattern is valid, but a word combination is implausible.
        - **Memorisation:** the output is an exact sentence copied from the training corpus.
        """
    )


# ---------------------------------------------------------------------------
# TAB 5 : Real-data evaluation, based on the provided script output
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="section-kicker">05 / Generalisation check</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Practice sentences vs real English</div>', unsafe_allow_html=True)
    st.markdown(
        "The same HMM/Viterbi implementation is estimated on Universal Dependencies English EWT. "
        "This test reveals how performance changes when the vocabulary and grammatical patterns become more varied.",
        unsafe_allow_html=True,
    )
    real = load_real_results()
    if real and "models" in real:
        real_models = real["models"]
        names = ["most-frequent tag", "HMM + Viterbi"]
        rows = []
        for name in names:
            values = real_models[name]
            rows.append({
                "Model": name,
                "Word-label accuracy": as_percent(values["accuracy"]),
                "Words seen in training": as_percent(values["known_words"]),
                "Words not seen in training": as_percent(values["unknown_words"]),
                "Fully correct sentences": as_percent(values["whole_sentence"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        hmm_values = real_models["HMM + Viterbi"]
        baseline_values = real_models["most-frequent tag"]
        delta = 100 * (hmm_values["accuracy"] - baseline_values["accuracy"])
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("HMM accuracy", as_percent(hmm_values["accuracy"]), "Token-level accuracy on EWT")
        with c2:
            render_metric_card("Improvement over baseline", f"+{delta:.1f} pp", "Absolute percentage-point difference")
        with c3:
            render_metric_card("Share of unseen words", as_percent(hmm_values["unknown_share_of_tokens"]), "Test tokens absent from training vocabulary")
        st.markdown(
            '<div class="callout"><b>Key interpretation.</b> The HMM improves overall accuracy over the '
            'most-frequent-tag baseline, but unknown-word accuracy remains low. A model trained on a tiny, '
            'regular synthetic grammar does not automatically generalise to real English. The 16-pattern '
            'validity metric is not meaningful for EWT, which contains thousands of distinct tag sequences.</div>',
            unsafe_allow_html=True,
        )
        if real.get("confusions"):
            st.markdown("#### Frequent HMM tagging confusions")
            conf_df = pd.DataFrame(real["confusions"], columns=["Gold tag", "Predicted tag", "Count"])
            st.dataframe(conf_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Real-data results are not available yet. From the project root, run "
            "`python -m src.evaluate_real`, then refresh this page."
        )
    st.markdown("#### Important limitation")
    st.markdown(
        "The EWT experiment uses lower-cased English web text and a 17-tag Universal Dependencies scheme. "
        "Its results should not be generalised automatically to other languages or domains. "
        "The first-order HMM is an educational baseline, not a state-of-the-art tagger."
    )


# ---------------------------------------------------------------------------
# Footer and required synthetic-data notice
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="padding:.6rem 0 0;color:#718078;font-size:.76rem;line-height:1.6">
      <b style="color:#375b4b">LAB NOTE</b> · The corpus is synthetic and the grammar is tiny.
      Metrics such as valid tag pattern are defined relative to this corpus and do not measure
      the quality of real English.
      <br>A model can produce a sentence that looks correct without understanding what it means.
    </div>
    """,
    unsafe_allow_html=True,
)