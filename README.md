# 📚 SmartLibrarian

## 1. Project Overview
**SmartLibrarian** is an AI-powered chatbot that recommends books based on natural-language prompts.  
It combines **RAG (Retrieval-Augmented Generation)** with **OpenAI GPT models** and a curated dataset of book summaries to deliver grounded, helpful recommendations.  

The chatbot is built with **Streamlit** for an interactive web interface and integrates an OpenAI **function-calling tool** to provide complete, authoritative summaries of recommended books.

---

## 2. Features
- 🔎 **Semantic Search**: uses OpenAI embeddings + ChromaDB to find the most relevant book summaries.  
- 💬 **Conversational Chatbot**: built with Streamlit for an interactive, user-friendly chat UI.  
- 🛠️ **Tool-Calling**: GPT uses the `get_summary_by_title` tool to fetch full summaries for chosen titles.  
- 📖 **Curated Dataset**: contains 20 diverse books spanning fantasy, thrillers, romance, history and non-fiction.  
- ⚡ **Fallbacks & Thresholds**: prevents nonsense queries from producing random recommendations.  

---

## 3. Installation

### Prerequisites
- Python **3.10+**  
- Git  
- An OpenAI API key (you must set in the environment variables on the local machine the variable `OPENAI_API_KEY` with your key value)

### Setup
```bash
# Clone the repository
git clone https://github.com/Dumitru-Daniel-Antoniu/SmartLibrarian.git
cd SmartLibrarian

# Create and activate a virtual environment
python -m venv .venv
source .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution

### Step 1: Build the index
Embed the dataset and build the ChromaDB vector store:
```bash
python -m scripts.build_index
```

### Step 2: Launch the chatbot
Run the Streamlit app:
```bash
python -m streamlit run src/ui/app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.