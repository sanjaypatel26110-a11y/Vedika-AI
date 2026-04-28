import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# Page Configuration (English)
st.set_page_config(page_title="Vedika AI", page_icon="🙏")
st.title("🙏 Vedika AI (Live Stream)")

@st.cache_resource
def load_model():
    model_id = "Vedika35/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    return tokenizer, model

# Loading Spinner (English)
with st.spinner("Loading Vedika AI... Please wait."):
    tokenizer, model = load_model()

# System Prompt (Instructing the AI to speak in Hindi)
system_prompt = "You are Vedika AI, a smart and polite Indian assistant. You must always reply in pure Hindi with respect. your creator is Divy Patel. You have to analyse deeply everything then answer fastly"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input (English)
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        response = st.write_stream(streamer)
        st.session_state.messages.append({"role": "assistant", "content": response})
