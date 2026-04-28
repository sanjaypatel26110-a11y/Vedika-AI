import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="Vedika AI", page_icon="🙏")
st.title("🙏 वेदिका AI (आपका अपना मॉडल)")

@st.cache_resource
def load_model():
    # यह रहा आपका खुद का मॉडल
    model_id = "Vedika35/Qwen2.5-0.5B-Instruct"
    
    # pipeline का उपयोग और सही dtype ताकि कोई वॉर्निंग न आए
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, # यह एरर को हटा देगा
        device_map="cpu" # फ्री सर्वर के लिए सबसे सुरक्षित
    )
    return pipe

with st.spinner("आपकी अपनी वेदिका AI लोड हो रही है..."):
    generator = load_model()

system_prompt = "आपका नाम वेदिका AI है। आप एक बहुत ही समझदार भारतीय सहायक हैं। कृपया हमेशा शुद्ध हिंदी में और आदर के साथ बात करें।"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("दिव्य जी, पूछिये क्या पूछना है?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # मॉडल से सीधा जवाब उत्पन्न करना
        output = generator(messages, max_new_tokens=512, temperature=0.7, do_sample=True)
        
        # जवाब को सही तरीके से निकालना
        response = output[0]['generated_text'][-1]['content']
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
