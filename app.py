import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# app config
st.set_page_config(page_title="Local LLM Chatbot", layout="wide")
st.title("Mini Assignment C2: Open-Source Chatbot")
st.caption("Chat with local models using Ollama and LangChain.")

# sidebar
with st.sidebar:
    st.header("Controls")
    
    model_option = st.selectbox(
        "Choose an LLM:",
        ("llama3:8b", "gemma:7b")
    )
    st.markdown("---")

    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 4096, 512, 100)
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)

    st.markdown("---")
    
    if st.button("Summarize Conversation"):
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.messages])
            summarization_prompt = f"Please provide a concise summary of the following conversation:\n\n---\n{conversation_history}\n---\n\nSummary:"
            
            with st.spinner("Summarizing..."):
                try:
                    summarizer_llm = ChatOllama(model=model_option, temperature=0.5)
                    summary = summarizer_llm.invoke(summarization_prompt).content
                    st.text_area("Conversation Summary", summary, height=200)
                except Exception as e:
                    st.error(f"Summarization error: {e}")
        else:
            st.warning("Not enough conversation to summarize.")

# chat
llm = ChatOllama(
    model=model_option,
    temperature=temperature,
    num_predict=max_tokens, 
    top_p=top_p
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, SystemMessage):
        pass
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(st.session_state.messages)
                ai_response_content = response.content
                st.markdown(ai_response_content)
                st.session_state.messages.append(response)
            except Exception as e:
                st.error(f"An error occurred")