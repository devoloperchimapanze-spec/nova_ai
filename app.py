import streamlit as st
from huggingface_hub import InferenceClient
import os

# 1. Page & Liquid Glass CSS
st.set_page_config(page_title="NOVA AI", page_icon="🛰️", layout="wide")
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .stChatFloatingInputContainer { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar & Model Selection
with st.sidebar:
    st.title("NOVA PRO 🛰️")
    model_choice = st.selectbox("Select Brain:", ["HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-Instruct-v0.2"])
    mode = st.radio("Tools:", ["Search Agent", "CoT Reasoning", "LTR (Memory)", "PDF Scan"])
    
    uploaded_file = st.file_uploader("Upload PDF Study Notes", type="pdf")
    if uploaded_file:
        st.success("PDF Loaded into LTR!")

# 3. Connect to the Brain
# Make sure your HF_TOKEN is saved in Streamlit Secrets!
client = InferenceClient(model_choice, token=st.secrets["HF_TOKEN"])

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask NOVA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.status(f"Nova {mode} thinking via {model_choice.split('/')[-1]}...", expanded=True) as status:
            
            # Feature Logic
            if mode == "CoT Reasoning":
                formatted_prompt = f"<|system|>\nYou are a helpful assistant that thinks step-by-step.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n<thought>"
            else:
                formatted_prompt = prompt

try:
                # Use chat_completion for better "GPT-3" style conversation
                response = client.chat_completion(
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=512,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
                status.update(label="Analysis Complete!", state="complete")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Brain is tired (API Error): {e}")
                st.error(f"Server Busy: {e}")
