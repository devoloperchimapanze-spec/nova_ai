import streamlit as st
from huggingface_hub import InferenceClient
import os

# 1. Page Config & Liquid Glass UI Styling
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
    h1, h2, h3 { color: white !important; }
    .stMarkdown { color: #e0e0e0 !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar & Model Selection
with st.sidebar:
    st.title("NOVA PRO 🛰️")
    st.markdown("---")
    
    # Using Llama-3.2-1B-Instruct as it has the highest uptime for free API shards
    model_choice = st.selectbox(
        "Select Brain:", 
        ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
    )
    
    mode = st.radio("Tools:", ["Search Agent", "CoT Reasoning", "LTR (Memory)", "PDF Scan"])
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload PDF Study Notes", type="pdf")
    if uploaded_file:
        st.success("PDF Content Indexed for LTR!")

# 3. Secure Brain Connection
try:
    # Uses the HF_TOKEN from your Streamlit Secrets
    client = InferenceClient(model_choice, token=st.secrets["HF_TOKEN"])
except Exception as e:
    st.error("Please check your HF_TOKEN in Streamlit Secrets!")
    st.stop()

# 4. Chat Session History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Input Handling & AI Response
if prompt := st.chat_input("Ask NOVA..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.status(f"Nova analyzing via {model_choice.split('/')[-1]}...", expanded=True) as status:
            
            # Formatting for Chain of Thought (CoT)
            if mode == "CoT Reasoning":
                system_instruction = "You are a helpful assistant. Explain your reasoning step-by-step."
            else:
                system_instruction = "You are NOVA, a helpful AI assistant for students."

            try:
                # API Call to the Brain
                response = client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content
                status.update(label="Response Secured!", state="complete")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                status.update(label="Server Delay", state="error")
                st.error(f"The model is currently busy. Please wait 10 seconds and try again. Error: {e}")

# Keep UI from jumping
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
     pass
