import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Vedika AI 3B", page_icon="🔥")
st.title("🔥 वेदिका AI (3B Power Edition)")

@st.cache_resource
def load_model():
    # 3 बिलियन पैरामीटर वाला मॉडल
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 16-bit (float16) कॉन्फ़िगरेशन और मेमोरी मैनेजमेंट
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, # 16-bit का उपयोग
        device_map="auto",
        low_cpu_mem_usage=True, # कम RAM में फिट होने की कोशिश
        trust_remote_code=True
    )
    return tokenizer, model

# मॉडल लोड होने के दौरान मैसेज दिखाएँ
try:
    with st.spinner("3B शक्तिशाली दिमाग लोड हो रहा है... इसमें थोड़ा समय लग सकता है।"):
        tokenizer, model = load_model()
except Exception as e:
    st.error(f"मेमोरी कम होने के कारण मॉडल लोड नहीं हो सका: {e}")
    st.info("सुझाव: 3B मॉडल के लिए अधिक RAM चाहिए। अगर यह फेल हो, तो हमें 1.5B पर वापस जाना होगा।")

# सिस्टम मैसेज
system_prompt = "आपका नाम वेदिका AI है। आप दिव्या पटेल द्वारा बनाई गई एक अत्यंत बुद्धिमान भारतीय AI हैं। हमेशा शुद्ध हिंदी और आदर के साथ उत्तर दें।"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("दिव्या जी, इस 3B दिमाग से कुछ भी पूछिए..."):
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

        # सटीक जवाब के लिए Parameters
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        
        response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
