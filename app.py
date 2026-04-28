import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# पेज की सेटिंग
st.set_page_config(page_title="Vedika AI", page_icon="🙏")

st.title("🙏 Vedika AI में आपका स्वागत है")
st.write("यह मॉडल Qwen2-0.5B पर आधारित है।")

# मॉडल लोड करने का फंक्शन (ताकि बार-बार लोड न हो)
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2-0.5B-Instruct" # यहाँ अपना मॉडल नाम बदल सकती हैं
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# चैट इंटरफेस
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("आप क्या पूछना चाहते हैं?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # मॉडल से जवाब जेनरेट करना
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
