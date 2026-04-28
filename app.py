import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="Vedika AI", page_icon="🙏")
st.title("🙏 वेदika AI (Direct Transformers)")

# मॉडल लोड करने का सबसे आसान तरीका 'pipeline' है
@st.cache_resource
def load_assistant():
    # यहाँ आप Qwen/Qwen2.5-1.5B-Instruct या अपना पसंदीदा मॉडल डाल सकती हैं
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    return pipe

generator = load_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("दिव्या जी, पूछिये क्या पूछना है?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # चैट के लिए मैसेज तैयार करना
        messages = [
            {"role": "system", "content": "आपका नाम वेदिका AI है। आप दिव्या पटेल द्वारा बनाई गई हैं। हमेशा हिंदी में बात करें।"},
            {"role": "user", "content": prompt},
        ]
        
        # डायरेक्ट जनरेशन
        output = generator(messages, max_new_tokens=512, temperature=0.7, do_sample=True)
        response = output[0]['generated_text'][-1]['content']
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
