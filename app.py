"""
AskAWS — Multi-Agent RAG Assistant
Educational UI for AWS learners.
"""

import time
import streamlit as st
from src.agents.graph import ask_streaming


st.set_page_config(
    page_title="AskAWS — Learn AWS with AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Global CSS ----------

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&display=swap" rel="stylesheet">

    <style>
    :root {
        --bg: #fffbf5;
        --bg-card: #ffffff;
        --bg-soft: #fef6eb;
        --bg-hover: #fdedd6;
        --border: #f0e3d0;
        --border-strong: #e6d4b8;
        --text: #1f1a14;
        --text-dim: #6b5d4c;
        --text-soft: #9b8b76;
        --accent: #f08a2c;
        --accent-dark: #d97316;
        --accent-soft: #ffe4c4;
        --green: #16a34a;
        --green-soft: #dcfce7;
        --red: #dc2626;
        --red-soft: #fee2e2;
        --font-body: 'Plus Jakarta Sans', system-ui, sans-serif;
        --font-display: 'Fraunces', Georgia, serif;
    }

    .stApp { background: var(--bg); font-family: var(--font-body); color: var(--text); }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

    /* Prevent horizontal scrolling */
    html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* ========== Welcome splash ========== */
    .splash-wrap {
        max-width: 720px;
        margin: 2rem auto;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 12px 40px rgba(240, 138, 44, 0.08);
    }
    .splash-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: var(--accent-dark);
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .splash-title {
        font-family: var(--font-display);
        font-size: 2.4rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin: 0 0 0.8rem 0;
    }
    .splash-title em { font-style: italic; color: var(--accent); font-weight: 500; }
    .splash-lede {
        font-size: 1rem;
        color: var(--text-dim);
        line-height: 1.6;
        margin: 0 0 1.5rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
        margin: 1.5rem 0 2rem 0;
    }
    .feature-card {
        background: var(--bg-soft);
        border-radius: 14px;
        padding: 1.1rem 0.9rem;
    }
    .feature-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
    .feature-title { font-weight: 700; font-size: 0.9rem; color: var(--text); margin-bottom: 0.25rem; }
    .feature-body { font-size: 0.78rem; color: var(--text-dim); line-height: 1.45; }
    @media (max-width: 700px) {
        .feature-grid { grid-template-columns: 1fr; }
        .splash-wrap { padding: 1.5rem; margin: 1rem; }
        .splash-title { font-size: 1.8rem; }
    }

    /* ========== Header ========== */
    .main-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding-bottom: 1.2rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
        gap: 1rem;
    }
    .main-header h1 {
        font-family: var(--font-display);
        font-size: 2.6rem;
        font-weight: 600;
        margin: 0;
        color: var(--text);
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .main-header h1 em { color: var(--accent); font-style: italic; font-weight: 500; }
    .main-header .tag {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        color: var(--text-soft);
        text-transform: uppercase;
        font-weight: 600;
        padding-bottom: 0.4rem;
        white-space: nowrap;
    }

    /* ========== Chat messages ========== */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
    }
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        display: none !important;
    }
    [data-testid="stChatMessageContent"] { background: transparent !important; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: var(--text); line-height: 1.7; font-size: 0.97rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--bg-soft) !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {
        font-weight: 600; font-size: 1rem; margin: 0;
    }

    /* ========== Citations ========== */
    .citation-block {
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px dashed var(--border-strong);
    }
    .citation-label {
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        color: var(--text-soft);
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .citation {
        background: var(--bg-card);
        border-left: 3px solid var(--accent);
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.86rem;
    }
    .citation-num {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 1.1rem;
        color: var(--accent);
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .citation-service {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent-dark);
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .citation-title { color: var(--text); font-weight: 600; margin-top: 0.3rem; margin-bottom: 0.35rem; }
    .citation-link a {
        color: var(--text-dim);
        text-decoration: none;
        font-size: 0.76rem;
        word-break: break-all;
    }
    .citation-link a:hover { color: var(--accent-dark); text-decoration: underline; }

    /* ========== Verdict ========== */
    .verdict-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.76rem; font-weight: 700;
        padding: 0.3rem 0.8rem; border-radius: 999px;
        margin-top: 0.9rem;
    }
    .verdict-pill.pass { background: var(--green-soft); color: var(--green); }
    .verdict-pill.fail { background: var(--red-soft); color: var(--red); }
    .verdict-feedback {
        font-size: 0.83rem; color: var(--text-dim); margin-top: 0.5rem;
        line-height: 1.5; font-style: italic;
    }

    /* ========== Agent flow ========== */
    .agent-flow-wrap {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem 1.1rem;
    }
    .flow-header {
        font-size: 0.68rem; letter-spacing: 0.18em;
        color: var(--text-soft); text-transform: uppercase;
        font-weight: 700; margin-bottom: 0.3rem;
    }
    .flow-subtitle {
        font-family: var(--font-display);
        font-size: 1rem; font-weight: 600;
        color: var(--text); margin-bottom: 1.1rem; letter-spacing: -0.01em;
    }
    .agent-flow { display: flex; flex-direction: column; gap: 0; }
    .agent-node {
        background: var(--bg-soft);
        border: 1.5px solid var(--border);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        display: flex; align-items: center; gap: 0.6rem;
        transition: all 0.35s ease;
    }
    .agent-node.active {
        background: var(--accent-soft);
        border-color: var(--accent);
        box-shadow: 0 0 0 4px rgba(240, 138, 44, 0.15);
        animation: gentlePulse 1.5s ease-in-out infinite;
    }
    .agent-node.done { background: var(--green-soft); border-color: var(--green); }
    .agent-node.skipped { opacity: 0.35; }
    @keyframes gentlePulse {
        0%, 100% { box-shadow: 0 0 0 4px rgba(240, 138, 44, 0.15); }
        50%      { box-shadow: 0 0 0 8px rgba(240, 138, 44, 0.10); }
    }
    .agent-icon {
        width: 30px; height: 30px;
        border-radius: 8px;
        background: var(--bg-card);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.95rem; flex-shrink: 0;
    }
    .agent-info { flex: 1; min-width: 0; }
    .agent-name { font-weight: 700; font-size: 0.86rem; color: var(--text); margin: 0; }
    .agent-role { font-size: 0.72rem; color: var(--text-dim); margin-top: 0.08rem; }
    .agent-status {
        font-size: 0.66rem; font-weight: 700;
        letter-spacing: 0.08em; color: var(--text-soft); text-transform: uppercase;
    }
    .agent-node.active .agent-status { color: var(--accent-dark); }
    .agent-node.done   .agent-status { color: var(--green); }
    .flow-connector { width: 2px; height: 12px; background: var(--border-strong); margin: 0 auto; }
    .flow-footer {
        margin-top: 1rem; padding-top: 0.9rem;
        border-top: 1px solid var(--border);
        font-size: 0.7rem; color: var(--text-soft); line-height: 1.5;
    }

    /* ========== HIDE the default chat_input completely ========== */
    [data-testid="stChatInput"], [data-testid="stBottom"], [data-testid="stChatInputContainer"] {
        display: none !important;
    }

    /* ========== Custom in-column input form ========== */
    .stForm {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    /* Text input wrapper */
    .stTextInput > div > div {
        background: var(--bg-card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: border-color 0.2s ease;
    }
    .stTextInput > div > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(240, 138, 44, 0.15), 0 4px 16px rgba(240, 138, 44, 0.12) !important;
    }
    .stTextInput input {
        background: transparent !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1rem !important;
    }
    .stTextInput input::placeholder { color: var(--text-soft) !important; }

    /* Form submit button — orange */
    .stFormSubmitButton button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-family: var(--font-body) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1.5rem !important;
        transition: all 0.2s ease;
        width: 100% !important;
    }
    .stFormSubmitButton button:hover {
        background: var(--accent-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(240, 138, 44, 0.35);
    }

    /* Primary CTA for splash */
    .stButton button[kind="primary"] {
        background: var(--accent) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 0.9rem 2rem !important;
        font-size: 1rem !important;
        border-radius: 14px !important;
        transition: all 0.2s ease;
    }
    .stButton button[kind="primary"]:hover {
        background: var(--accent-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(240, 138, 44, 0.35);
    }

    /* Mobile */
    @media (max-width: 900px) {
        .main-header h1 { font-size: 2rem; }
        .main-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
        .main-header .tag { padding-bottom: 0; }
    }

    .input-wrap-label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        color: var(--text-soft);
        text-transform: uppercase;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------

AGENTS_META = [
    ("router",     "🧭", "Router",     "Decides if your question is in-scope"),
    ("retrieval",  "🔍", "Retrieval",  "Searches AWS docs for relevant context"),
    ("generator",  "✍️", "Generator",  "Writes a grounded answer with citations"),
    ("critic",     "✅", "Critic",     "Fact-checks the answer against sources"),
]


def agent_flow_html(done: set, active: str | None, refused: bool = False) -> str:
    parts = [
        '<div class="agent-flow-wrap">',
        '<div class="flow-header">multi-agent pipeline</div>',
        '<div class="flow-subtitle">How your question is answered</div>',
        '<div class="agent-flow">',
    ]
    for i, (key, icon, name, role) in enumerate(AGENTS_META):
        if key in done:
            cls, status = "done", "done"
        elif key == active:
            cls, status = "active", "running"
        elif refused and key != "router":
            cls, status = "skipped", "skipped"
        else:
            cls, status = "", "idle"
        parts.append(
            f'<div class="agent-node {cls}">'
            f'<div class="agent-icon">{icon}</div>'
            f'<div class="agent-info"><div class="agent-name">{name}</div>'
            f'<div class="agent-role">{role}</div></div>'
            f'<div class="agent-status">{status}</div></div>'
        )
        if i < len(AGENTS_META) - 1:
            parts.append('<div class="flow-connector"></div>')

    parts.append('</div>')
    parts.append(
        '<div class="flow-footer">'
        'Each agent has one job. Together they keep answers grounded in real AWS docs.'
        '</div>'
    )
    parts.append('</div>')
    return "".join(parts)


def citation_html(s: dict) -> str:
    return (
        f"<div class='citation'>"
        f"<span class='citation-num'>[{s['idx']}]</span>"
        f"<span class='citation-service'>{s['service']}</span>"
        f"<div class='citation-title'>{s['title']}</div>"
        f"<div class='citation-link'><a href='{s['url']}' target='_blank'>{s['url']}</a></div>"
        f"</div>"
    )


def render_assistant_message(msg: dict):
    st.markdown(msg["content"])
    if msg.get("sources"):
        st.markdown("<div class='citation-block'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='citation-label'>Sources · {len(msg['sources'])} references</div>",
            unsafe_allow_html=True,
        )
        for s in msg["sources"]:
            st.markdown(citation_html(s), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    verdict = msg.get("verdict")
    if verdict:
        cls = "pass" if verdict == "pass" else "fail"
        label = "Answer verified" if verdict == "pass" else "Needs review"
        st.markdown(
            f"<div class='verdict-pill {cls}'>{label}</div>"
            f"<div class='verdict-feedback'>{msg.get('critic_feedback', '')}</div>",
            unsafe_allow_html=True,
        )


# ---------- Session state ----------

if "splash_dismissed" not in st.session_state:
    st.session_state.splash_dismissed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------- SPLASH ----------

if not st.session_state.splash_dismissed:
    st.markdown(
        """
        <div class="splash-wrap">
            <div class="splash-eyebrow">a learning tool · powered by ai</div>
            <h1 class="splash-title">Learn AWS, <em>one question at a time.</em></h1>
            <p class="splash-lede">
                AskAWS is a friendly assistant that helps you understand AWS services.
                Type a question, and four specialized AI agents work together to give you
                an accurate, grounded answer — with sources you can verify.
            </p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <div class="feature-title">Ask anything</div>
                    <div class="feature-body">Questions about S3, Lambda, EC2, IAM, or SageMaker.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">Four AI agents</div>
                    <div class="feature-body">A router, retriever, generator, and fact-checker collaborate.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📚</div>
                    <div class="feature-title">Always cited</div>
                    <div class="feature-body">Every answer points to official AWS documentation.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 1, 1])
    with col_c:
        if st.button("Start exploring →", key="splash_start", type="primary", use_container_width=True):
            st.session_state.splash_dismissed = True
            st.rerun()
    st.stop()


# ---------- MAIN ----------

st.markdown(
    """
    <div class="main-header">
        <h1>Ask<em>AWS</em></h1>
        <div class="tag">grounded · cited · multi-agent</div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_col, flow_col = st.columns([1.55, 1], gap="large")

# Right panel
with flow_col:
    flow_placeholder = st.empty()
    flow_placeholder.markdown(
        agent_flow_html(done=set(), active=None), unsafe_allow_html=True
    )


# Left column — chat + input
with chat_col:
    # Render past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant_message(msg)

    # Placeholder for the in-flight streaming response
    streaming_container = st.container()

    # Custom input form INSIDE the chat column
    st.markdown("<div class='input-wrap-label'>Your question</div>", unsafe_allow_html=True)
    with st.form(key="question_form", clear_on_submit=True):
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input(
                "question",
                placeholder="Ask anything about AWS S3, Lambda, EC2, IAM, or SageMaker...",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Ask", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.pending_question = user_input.strip()


# ---------- Handle a pending question ----------

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question})

    with chat_col:
        with streaming_container:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                sources_placeholder = st.empty()
                verdict_placeholder = st.empty()

                done = set()
                active = "router"
                flow_placeholder.markdown(
                    agent_flow_html(done, active), unsafe_allow_html=True
                )

                final_state = None
                try:
                    for node_name, state in ask_streaming(question):
                        if node_name == "final":
                            final_state = state
                            continue
                        done.add(node_name)
                        refused = state.get("route_decision") == "refuse"
                        if refused:
                            active = None
                        else:
                            remaining = [a for a, _, _, _ in AGENTS_META if a not in done]
                            active = remaining[0] if remaining else None
                        flow_placeholder.markdown(
                            agent_flow_html(done, active, refused=refused),
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.05)

                    if final_state is None:
                        answer_placeholder.error("No response generated.")
                    else:
                        answer = final_state.get("final_answer", "(no answer)")
                        refused = final_state.get("route_decision") == "refuse"
                        flow_placeholder.markdown(
                            agent_flow_html(done, None, refused=refused),
                            unsafe_allow_html=True,
                        )
                        answer_placeholder.markdown(answer)

                        msg_data = {"role": "assistant", "content": answer}

                        chunks = final_state.get("retrieved_chunks", [])
                        distances = final_state.get("retrieved_distances", [])
                        if chunks:
                            sources = [
                                {
                                    "idx": i,
                                    "service": doc.metadata.get("service", "?").upper(),
                                    "title": doc.metadata.get("title", "?"),
                                    "url": doc.metadata.get("source_url", "?"),
                                    "distance": dist,
                                }
                                for i, (doc, dist) in enumerate(zip(chunks, distances), start=1)
                            ]
                            msg_data["sources"] = sources

                            with sources_placeholder.container():
                                st.markdown("<div class='citation-block'>", unsafe_allow_html=True)
                                st.markdown(
                                    f"<div class='citation-label'>Sources · {len(sources)} references</div>",
                                    unsafe_allow_html=True,
                                )
                                for s in sources:
                                    st.markdown(citation_html(s), unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                        verdict = final_state.get("critic_verdict")
                        if verdict:
                            msg_data["verdict"] = verdict
                            msg_data["critic_feedback"] = final_state.get("critic_feedback", "")
                            cls = "pass" if verdict == "pass" else "fail"
                            label = "Answer verified" if verdict == "pass" else "Needs review"
                            verdict_placeholder.markdown(
                                f"<div class='verdict-pill {cls}'>{label}</div>"
                                f"<div class='verdict-feedback'>{msg_data['critic_feedback']}</div>",
                                unsafe_allow_html=True,
                            )

                        st.session_state.messages.append(msg_data)

                except Exception as e:
                    answer_placeholder.error(f"Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                    })