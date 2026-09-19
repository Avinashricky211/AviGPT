"""
AViGPT v2 Studio - Autonomous Hardware Memory Bus Interface
-----------------------------------------------------------
Interactive Desktop UI for AViGPT v2:
- 183M Parameter Core Neural Network
- Sub-Millisecond SQLite FTS5 NVMe SSD Memory Engine
- Autonomous Python Calculation Sandbox
- Live Hardware Bus Telemetry & Dynamic Memory Writer

Creator & Owner: Avinash Ricky Yadlapalli
"""

import os
import re
import time
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from memory_engine import SSDMemoryEngine
from autonomous_bus import AutonomousHardwareBus

# Page Configuration
st.set_page_config(
    page_title="AViGPT - Neural Hardware Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stChatMessage { border-radius: 12px; margin-bottom: 12px; }
    .creator-badge {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold;
        display: inline-block; margin-bottom: 15px; font-size: 13px;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px; padding: 12px; margin-bottom: 10px;
    }
    .tag-intent { color: #38bdf8; font-weight: bold; }
    .tag-mem { color: #4ade80; font-weight: bold; }
    .tag-calc { color: #fbbf24; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Priority search paths for AViGPT models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(base_dir, "AViGPT_v2_HF"),
        os.path.join(base_dir, "model"),
        "D:/coding/Newgpt/AViGPT_v2_HF",
        "D:/coding/Newgpt/ckpt_v2_hardware_bus/AViGPT_v2_HF",
        "D:/coding/Newgpt/AViGPT_Master_DPO_v2",
        "D:/coding/Newgpt/AViGPT_v4_Coder",
        "D:/coding/Newgpt/ckpt_sft/AViGPT",
        "D:/coding/Newgpt/"
    ]
    
    model_dir = None
    for p in search_paths:
        if os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(os.path.join(p, "pytorch_model.bin")):
            model_dir = p
            break
            
    if model_dir is None:
        model_dir = "D:/coding/Newgpt/"

    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    except Exception:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure hardware memory bus tokens are registered
    special_tokens = [
        "<|intent_start|>", "<|intent_end|>",
        "<|mem_query|>", "<|mem_query_end|>",
        "<|mem_payload|>", "<|mem_payload_end|>",
        "<|calc|>", "<|calc_end|>",
        "<|synthesize|>"
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Load model
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    except Exception:
        ckpt_path = "D:/coding/Newgpt/ckpt_real/ckpt.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        else:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()

    # Initialize Hardware Bus Controller
    bus = AutonomousHardwareBus(model=model, tokenizer=tokenizer, device=device)
    return tokenizer, model, bus, device, os.path.basename(model_dir.rstrip("/\\"))


tokenizer, model, bus, device, model_name = load_system()

# ─── SIDEBAR: HARDWARE BUS MONITOR & MEMORY WRITER ───
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=110)
    st.markdown('<div class="creator-badge">Creator & Owner: Avinash Ricky Yadlapalli</div>', unsafe_allow_html=True)
    
    st.title("⚙️ Hardware Bus Status")
    st.caption(f"Active Model: **{model_name}**")
    st.caption(f"Compute Device: **{device.upper()}** (183M Core)")

    st.markdown("---")
    enable_bus = st.toggle("⚡ Autonomous SSD Memory Bus", value=True, help="Intercepts <|mem_query|> & <|calc|> tokens in real-time.")
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
    max_tokens = st.slider("Max New Tokens", min_value=50, max_value=350, value=180, step=10)

    st.markdown("---")
    st.subheader("💾 SSD Dynamic Memory Writer")
    st.caption("Teach AViGPT new facts in real-time without retraining!")

    with st.expander("➕ Store Fact into SSD", expanded=False):
        fact_title = st.text_input("Fact Title / Keyword", placeholder="e.g., Project Hyperion")
        fact_content = st.text_area("Ground Truth Payload", placeholder="e.g., The secret code is 8842.")
        fact_domain = st.selectbox("Knowledge Domain", ["General", "Confidential", "Science", "Tech", "History"])

        if st.button("Write to SSD Storage", use_container_width=True):
            if fact_title and fact_content:
                t_w0 = time.perf_counter()
                ok = bus.memory.store(fact_title, fact_content, fact_domain)
                w_lat = (time.perf_counter() - t_w0) * 1000.0
                if ok:
                    st.success(f"Stored to SSD in {w_lat:.2f} ms!")
                else:
                    st.error("Failed to write to SSD.")
            else:
                st.warning("Please provide both title and content.")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── MAIN CHAT INTERFACE ───
st.title("🧠 AViGPT Studio")
st.markdown(
    "**The First SSD-Augmented Autonomous Language Model** — Designed, pretrained from scratch, "
    "and equipped with an external hardware memory bus by **Avinash Ricky Yadlapalli**."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **AViGPT**, owned and created by **Avinash Ricky Yadlapalli**. My neural core (183M) is connected to a sub-millisecond local SSD memory bus. How can I assist you today?"
        }
    ]

# Render Message History
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "telemetry" in msg:
            tel = msg["telemetry"]
            with st.expander("⚡ Hardware Bus Execution Telemetry", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("SSD Hits", tel.get("ssd_memory_hits", 0))
                col2.metric("SSD Latency", f"{tel.get('ssd_latency_ms', 0):.2f} ms")
                col3.metric("Total Latency", f"{tel.get('total_latency_s', 0):.2f} s")
                if "raw_trajectory" in tel:
                    st.markdown("**Raw Trajectory:**")
                    st.code(tel["raw_trajectory"])

# User Input
if prompt := st.chat_input("Ask AViGPT anything (e.g. 'Who is your owner?' or 'When was Apollo 11?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("AViGPT is routing through neural core and SSD bus..."):
            if enable_bus:
                raw_resp, metrics = bus.generate_autonomous_response(
                    prompt,
                    max_total_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Extract clean synthesis for display
                if "<|synthesize|>" in raw_resp:
                    clean_answer = raw_resp.split("<|synthesize|>")[-1].replace("<|endoftext|>", "").strip()
                else:
                    clean_answer = raw_resp.replace("<|endoftext|>", "").strip()

                metrics["raw_trajectory"] = raw_resp
                st.markdown(clean_answer)

                with st.expander("⚡ Hardware Bus Execution Telemetry", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("SSD Memory Hits", metrics.get("ssd_memory_hits", 0))
                    col2.metric("SSD Retrieval Latency", f"{metrics.get('ssd_latency_ms', 0):.2f} ms")
                    col3.metric("Total Latency", f"{metrics.get('total_latency_s', 0):.2f} s")
                    st.markdown("**Autonomous Hardware Trajectory:**")
                    st.code(raw_resp)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clean_answer,
                    "telemetry": metrics
                })
            else:
                # Direct generation without bus interception
                full_prompt = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{prompt}\n\n### Response:\n"
                )
                inp = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(inp, max_new_tokens=max_tokens, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
                res = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
