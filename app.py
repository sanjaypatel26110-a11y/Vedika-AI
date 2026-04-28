import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# पेज की सेटिंग
st.set_page_config(page_title="Vedika AI", page_icon="🙏")
st.title("🙏 वेदिका AI (Llama 3.2 - 1B Edition)")

# Hugging Face Access Token
# नोट: सुरक्षित अभ्यास के लिए इसे Streamlit Secrets में रखना बेहतर है, लेकिन अभी परीक्षण के लिए यह यहाँ ठीक है।
HF_TOKEN = "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"

@st.cache_resource
def load_model():
    # Meta Llama 3.2 1B Instruct मॉडल
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    # टोकन के साथ Tokenizer लोड करें
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    
    # टोकन और मेमोरी बचाने वाली सेटिंग्स के साथ मॉडल लोड करें
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    return tokenizer, model

# लोडिंग के दौरान सन्देश
with st.spinner("वेदिका AI (Llama 3.2) तैयार हो रही है... कृपया प्रतीक्षा करें।"):
    tokenizer, model = load_model()

# सिस्टम प्रॉम्प्ट
system_prompt = "आपका नाम वेदिका AI है। आप दिव्या पटेल द्वारा बनाई गई एक बहुत ही विनम्र और ज्ञानी भारतीय सहायक हैं। कृपया हमेशा शुद्ध हिंदी में और आदर के साथ बात करें।"

if "messages" not in st.session_state:
    st.session_state.messages = []

# पुरानी चैट दिखाएं
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# यूज़र का इनपुट
if prompt := st.chat_input("दिव्या जी, मैं आपकी क्या सहायता कर सकती हूँ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Llama 3.2 के अनुसार चैट टेम्पलेट तैयार करना
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # स्ट्रीमर सेट करें
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # जनरेशन सेटिंग्स
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        # बैकग्राउंड थ्रेड में जनरेशन शुरू करें
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # स्ट्रीमिंग के साथ जवाब दिखाएं
        response = st.write_stream(streamer)
        
        # चैट इतिहास में जवाब सहेजें
        st.session_state.messages.append({"role": "assistant", "content": response})
