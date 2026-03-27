import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="SaaS Support Assistant", page_icon="💬", layout="centered")

st.title("💬 SaaS Support Assistant")
st.write("Welcome! Ask me any technical question based on our documentation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I reset my password?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={"question": prompt}
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No answer found.")
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"{i+1}. {source}")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to backend: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
