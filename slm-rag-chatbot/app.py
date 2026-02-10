import streamlit as st
import requests
import json

st.title("🤖 SLM RAG Chatbot")
st.caption("Powered by Llama 3.2 on Modal")

# Modal endpoint
MODAL_URL = "https://avipalsai--llm-rag-generator-generate.modal.run"

# Input
question = st.text_input("Ask a question:", placeholder="What is your refund policy?")
max_results = st.slider("Max results to retrieve:", 1, 10, 3)

if st.button("Ask", type="primary"):
    if question:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    MODAL_URL,
                    json={"question": question, "max_results": max_results},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Answer:")
                    st.write(data.get("answer", "No answer"))
                    
                    with st.expander("📚 Sources used"):
                        for i, source in enumerate(data.get("sources", []), 1):
                            st.markdown(f"**{i}.** {source}")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")
    else:
        st.warning("⚠️ Please enter a question")
