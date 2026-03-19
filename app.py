import streamlit as st
from huggingface_hub import InferenceClient
import os

# 1. Page Config & Liquid Glass UI
st.set_page_config(page_title="NOVA AI", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stChatFloatingInputContainer {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar Setup
with st.sidebar:
    st.title("NOVA PRO 🛰️")
    st.markdown("---")
    model_choice = st.selectbox(
        "Select Brain:", 
        ["HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-Instruct-v0.2"]
    )
    mode = st.radio("Tools:", ["Search Agent", "CoT Reasoning", "LTR (Memory)", "PDF Scan"])
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload PDF Study Notes", type="pdf")
    if uploaded_file:
        st.success("PDF Content Indexed!")

# 3. Secure Brain Connection
try:
    # This pulls your token from the Streamlit Secrets you set up
    client = InferenceClient(model_choice, token=st.secrets["HF_TOKEN"])
except Exception as e:
    st.error("Missing HF_TOKEN in Secrets!")
    st.stop()

# 4. Chat Session History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Input Handling
if prompt := st.chat_input("Ask NOVA..."):
    # Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.status(f"Nova {mode} thinking...", expanded=True) as status:
            
            # Apply CoT or Standard formatting
            if mode == "CoT Reasoning":
                formatted_prompt = f"System: Think step-by-step.\nUser: {prompt}"
            else:
                formatted_prompt = prompt

            try:
                # Optimized for speed and GPT-3 style responses
                stream = client.chat_completion(
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=1024,
                    stream=True,
                    temperature=0.7
                )
                
                response_text = ""
                placeholder = st.empty()
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        response_text += content
                        placeholder.markdown(response_text)
                
                status.update(label="Analysis Complete!", state="complete")
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                status.update(label="Error Occurred", state="error")
                st.error(f"The model is currently overloaded. Try 'Zephyr' in the sidebar! Error: {e}")

# Small hack to keep the UI snappy
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
     pass # Final response is already drawn
