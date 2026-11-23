


import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dotenv import load_dotenv

# ---------------- ENVIRONMENT ----------------

load_dotenv(ROOT_DIR / ".env")

# Streamlit Cloud Secrets support
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from src.rag_pipeline import create_rag_pipeline


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Finexa | AI Financial Operations",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}

[data-testid="stSidebar"] {
    background-color: #101d35;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 30px;
    font-weight: 750;
    color: #142747;
}

.subtitle {
    color: #718096;
    font-size: 14px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e5eaf2;
    margin-bottom: 12px;
}

.metric-label {
    color: #718096;
    font-size: 14px;
}

.metric-value {
    color: #142747;
    font-size: 27px;
    font-weight: 700;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---------------- NAVIGATION CALLBACK ----------------

def navigate_to(page_name):
    st.session_state.current_page = page_name


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("# 💰 FINEXA")
    st.caption("AI FINANCIAL OPERATIONS PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "AI Policy Assistant",
            "Policy Library",
            "Chat History"
        ],
        key="current_page"
    )

    st.divider()

    st.markdown("### System Status")
    st.success("Application Running")
    st.caption("Finexa AI Assistant")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Financial Operations Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Welcome to Finexa AI Financial Operations Platform</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    metrics = [
        (col1, "Total Policies", "22"),
        (col2, "Source Records", "9,805"),
        (col3, "Policy Coverage", "POL001–POL022")
    ]

    for column, label, value in metrics:
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("AI Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🤖 AI Policy Assistant")

        st.write(
            "Ask questions about financial policies "
            "and retrieve answers from the handbook."
        )

        st.button(
            "Open AI Assistant",
            on_click=navigate_to,
            args=("AI Policy Assistant",),
            key="open_ai_assistant"
        )

    with col2:

        st.markdown("### 📚 Policy Library")

        st.write(
            "Explore the 22 financial policies "
            "included in the policy handbook."
        )

        st.button(
            "Explore Policies",
            on_click=navigate_to,
            args=("Policy Library",),
            key="explore_policies"
        )


# ---------------- AI POLICY ASSISTANT ----------------

elif page == "AI Policy Assistant":

    st.markdown(
        '<div class="main-title">🤖 AI Policy Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Ask questions about your financial policies</div>',
        unsafe_allow_html=True
    )

    # Check API key
    if not os.getenv("GROQ_API_KEY"):

        st.error(
            "Groq API key not found. "
            "Configure GROQ_API_KEY in your .env file "
            "or Streamlit Cloud Secrets."
        )

        st.stop()

    # Initialize RAG pipeline
    if st.session_state.rag_pipeline is None:

        with st.spinner("Initializing Finexa AI..."):

            try:
                st.session_state.rag_pipeline = create_rag_pipeline()

            except Exception as error:
                st.error(f"Initialization error: {error}")
                st.stop()

    # Display previous conversations
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input(
        "Ask your financial question..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching financial policies..."):

                try:

                    answer = st.session_state.rag_pipeline(question)

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.error(f"Error: {error}")


# ---------------- POLICY LIBRARY ----------------

elif page == "Policy Library":

    st.markdown(
        '<div class="main-title">📚 Financial Policy Library</div>',
        unsafe_allow_html=True
    )

    st.caption("Policies listed in Financial Policies Handbook")

    policies = [
        ("POL001", "Failed Payment Policy", "Payment"),
        ("POL002", "Loan EMI Auto-Debit Failure", "Lending"),
        ("POL003", "Chargeback and Dispute Policy", "Cards"),
        ("POL004", "Document Verification SLA", "Operations"),
        ("POL005", "Immediate Auto-Refund Policy", "UPI"),
        ("POL006", "High-Value Transaction Freeze", "Compliance"),
        ("POL007", "Insufficient Balance Penalty Waiver", "Lending"),
        ("POL008", "Corporate NetBanking Settlement", "Corporate"),
        ("POL009", "Beneficiary Bank Offline Policy", "Switching"),
        ("POL010", "Chargeback Evidence Standard", "Disputes"),
        ("POL011", "Priority Escalation Protocol", "Support"),
        ("POL012", "Merchant Settlement Reversal", "Merchant Services"),
        ("POL013", "Duplicate Transaction Policy", "Payment & Clearing"),
        ("POL014", "Transaction Reversal Policy", "Core Banking Operations"),
        ("POL015", "UPI Dispute Policy", "Digital Payments"),
        ("POL016", "KYC Verification Policy", "Compliance & Risk"),
        ("POL017", "Account Closure Policy", "Account Servicing"),
        ("POL018", "Fund Transfer Policy", "Funds Transfer & Remittance"),
        ("POL019", "Transaction Fee Policy", "Billing & Tariffs"),
        ("POL020", "Credit Card Payment Policy", "Credit Cards"),
        ("POL021", "Refund Processing Policy", "Merchant Services"),
        ("POL022", "Suspicious Transaction Policy", "AML & Financial Crime")
    ]

    search = st.text_input(
        "Search policies",
        placeholder="Enter policy name or ID"
    )

    filtered = [
        policy for policy in policies
        if search.lower() in " ".join(policy).lower()
    ]

    for policy_id, name, category in filtered:

        with st.container(border=True):

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown(f"**{policy_id}**")

            with col2:
                st.markdown(f"**{name}**")
                st.caption(category)


# ---------------- CHAT HISTORY ----------------

elif page == "Chat History":

    st.markdown(
        '<div class="main-title">💬 Chat History</div>',
        unsafe_allow_html=True
    )

    st.caption("Your current session conversations")

    if st.session_state.messages:

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.button("Clear Chat History"):

            st.session_state.messages = []
            st.rerun()

    else:

        st.info(
            "No conversations yet. Start chatting with Finexa AI."
        )
