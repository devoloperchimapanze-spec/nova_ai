import streamlit as st
import os
from huggingface_hub import InferenceClient

# 1. Glassmorphism UI Styling
st.set_page_config(page_title="NOVA AI", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    h1 { color: #ffffff; font-family: 'Inter', sans-serif; font-weight: 800; letter-spacing: -1px; }
    .stTextInput>div>div>input { background: rgba(255,255,255,0.1); color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar for Features
with st.sidebar:
    st.title("NOVA 🛰️")
    st.markdown("---")
    mode = st.radio("Student Tools:", ["Search Agent", "CoT Reasoning", "LTR (Memory)", "PDF/Image Scan"])
    st.info("Built for Students by Pratyush")

# 3. Brain Setup
try:
    client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=st.secrets["HF_TOKEN"])
except:
    st.error("Please add HF_TOKEN to Streamlit Secrets!")

# 4. Main UI
st.title("NOVA AI")
st.caption(f"Active Mode: {mode}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Feature Logic
if prompt := st.chat_input("How can I help your studies today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.status(f"Nova {mode} processing...", expanded=True) as status:
            
            if mode == "CoT Reasoning":
                st.write("🔍 Breaking down the problem step-by-step...")
                final_prompt = f"Explain step-by-step: {prompt}"
            elif mode == "LTR (Memory)":
                st.write("📂 Searching personal study notes...")
                # Add your FAISS logic here
                final_prompt = prompt
            else:
                final_prompt = prompt

            response = client.chat_completion(messages=[{"role": "user", "content": final_prompt}], max_tokens=1024)
            full_response = response.choices[0].message.content
            status.update(label="Response Generated!", state="complete")

        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
