import gradio as gr
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Initialize AI Client safely
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=os.getenv("HF_TOKEN"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Memory Loading Logic
try:
    if os.path.exists("nova_index.faiss"):
        index = faiss.read_index("nova_index.faiss")
        chunks = np.load("chunks.npy", allow_pickle=True)
        status = "Nova's Memory: Online ✅"
    else:
        index, chunks = None, None
        status = "Nova's Memory: Offline ⚠️"
except Exception as e:
    index, chunks = None, None
    status = f"Memory Error: {e}"

def nova_engine(msg, history):
    # RAG Search
    context = ""
    if index:
        query_vec = embed_model.encode([msg])
        _, I = index.search(np.array(query_vec).astype('float32'), k=2)
        context = " ".join([chunks[i] for i in I[0]])

    system_msg = f"You are Nova, a male study assistant built by Pratyush Anant in 2026. Use this context: {context}"
    
    response = ""
    for message in client.chat_completion(
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": msg}],
        max_tokens=800, stream=True
    ):
        token = message.choices[0].delta.content
        if token:
            response += token
            yield response

# The "Jealousy-Inducing" UI
theme = gr.themes.Soft(primary_hue="cyan", neutral_hue="zinc")
with gr.Blocks(theme=theme, title="NOVA AI") as demo:
    gr.Markdown(f"# 🪐 NOVA AI\n**Status:** {status} | **Developer:** Pratyush Anant")
    gr.ChatInterface(fn=nova_engine)

demo.launch()
