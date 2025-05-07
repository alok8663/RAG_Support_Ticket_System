import streamlit as st
from retriever import load_tickets, build_index, retrieve
from generator import generate_response

st.title("RAG-Based Support Ticket System")

query = st.text_input("Enter your support query:")

if query:
    tickets = load_tickets()
    index, _, texts = build_index(tickets)
    retrieved = retrieve(query, index, texts)
    st.subheader("Top Relevant Tickets")
    for text in retrieved:
        st.markdown(f"- {text}")

    response = generate_response(query, retrieved)
    st.subheader("Generated Response")
    st.write(response)