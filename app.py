import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Use the content generator pipeline for the second tab
from content_generator import generate_content


# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="LLM Content Generator & Chat", layout="centered")


@st.cache_resource(show_spinner=False)
def get_llm():
    """Create and cache the chat model once per session."""
    # Prefer the larger, more capable model if available
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    return ChatGroq(model=model_name, temperature=temperature)


def ensure_api_key():
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it to your .env file.")
        return False
    return True


# ---------- UI ----------
st.title("🧠 LLM Content Generator & Chat")
tabs = st.tabs(["Chat", "Content Generator"])


# ---------- Chat Tab ----------
with tabs[0]:
    st.subheader("Chat Assistant")
    st.caption("Backed by Groq Llama 3.3. Your messages are ephemeral and stored only in this session.")

    if ensure_api_key():
        llm = get_llm()

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hi! I'm your AI assistant. How can I help today?")
            ]

        # Render history
        for msg in st.session_state.chat_history:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            with st.chat_message(role):
                st.markdown(msg.content)

        # Chat input
        user_input = st.chat_input("Type your message…")
        if user_input:
            user_msg = HumanMessage(content=user_input)
            st.session_state.chat_history.append(user_msg)

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    # Optional system prompt for behavior
                    system_msg = SystemMessage(content=os.getenv(
                        "SYSTEM_PROMPT",
                        "You are a helpful and concise assistant."
                    ))
                    messages = [system_msg] + st.session_state.chat_history
                    try:
                        ai_msg = llm.invoke(messages)
                        st.session_state.chat_history.append(ai_msg)
                        st.markdown(ai_msg.content)
                    except Exception as e:
                        st.error(f"Model error: {e}")


# ---------- Content Generator Tab ----------
with tabs[1]:
    st.subheader("Generate Marketing Content")
    st.caption("Fill in the fields and generate ready-to-publish content.")

    with st.form("content_form"):
        topic = st.text_input("Topic", placeholder="e.g., The benefits of virtual reality for education")
        platform = st.selectbox(
            "Platform",
            ["Blog Post", "Twitter/X", "Instagram Caption", "LinkedIn Post"],
            index=0,
        )
        audience = st.text_input("Audience", placeholder="e.g., School Administrators and Educators")
        tone = st.selectbox("Tone", ["Informative", "Professional", "Friendly", "Playful", "Persuasive"], index=0)
        submitted = st.form_submit_button("Generate")

    if submitted:
        if not all([topic.strip(), audience.strip(), tone.strip(), platform.strip()]):
            st.warning("Please fill in all fields.")
        elif not ensure_api_key():
            pass
        else:
            with st.spinner("Generating content…"):
                try:
                    output = generate_content(topic=topic, platform=platform, audience=audience, tone=tone)
                    st.markdown("---")
                    st.markdown(output)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
