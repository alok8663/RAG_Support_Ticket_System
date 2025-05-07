# RAG Support Ticket System

A Retrieval-Augmented Generation (RAG) based support ticket system that leverages custom knowledge to answer user queries using an interactive Streamlit interface.

## 📁 Project Structure
##### app.py
##### retriever.py
##### generator.py
##### tickets.json
##### requirements.txt
##### README.md
##### .gitignore


## 🔧 Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/alok8663/RAG_Support_Ticket_System.git
cd RAG_Support_Ticket_System

### 2. Create and activate a virtual environment
conda create -p venv python==3.8 -y
conda activate venv/ 


### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Application
streamlit run app.py

