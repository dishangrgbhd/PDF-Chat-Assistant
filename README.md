 # 📄💬 PDF Chat Assistant using Agentic RAG
**Conversational Chatbot with PDF Assistance capabilities.**


This project is an intelligent chatbot assistant that allows users to **chat naturally**, **ask questions about a PDF paper(it can be a research paper or PDF related to laws or an academic material)**, also providing the functionality to **generate summaries related to that PDF**, all in a single interface. It's built with:

- **LangGraph** (to orchestrate memory and tool-based flows)
- **LangChain** (LCEL for retrieval + summarization)
- **LangSmith** (for tracing tool calls, LLM prompts, and debugging) 
- **Groq Llama3** (for blazing-fast LLM responses)
- **Streamlit** (for frontend deployment)

---

##  Features 🧠

-  **Conversational Chatbot** with memory keeping track of past conversations
-  Upload any kind of PDF and **ask questions about it**
-  Get **summaries** of the full paper content
-  LangGraph flow to route queries smartly:
    - Normal chat
    - QA from PDF(agent tool)
    - Summarization(agent tool)
-  Uses **embeddings + vector search** (FAISS) for document retrieval
-  Powered by [Groq API](https://groq.com) (Llama3)

## 📽️ Demo

Watch a quick walkthrough of the PDF Chat Assistant in action:

👉 [Demo video](https://www.dropbox.com/scl/fi/rd7x19e7ufb6r9zmerd6y/demo-assistant.mp4?rlkey=zopfetq0yl6grz8zezeokr5ig&st=8l6tqi31&dl=0)

---

##  Installation & Setup 💻

Follow the steps below to get the app running locally:

### 1. Clone the Repository

```bash
git clone https://github.com/dishangrgbhd/PDF-Chat-Assistant.git
cd PDF-Chat-Assistant
```

### 2. Set Up a Virtual Environment 

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

Create a .env file in the root of your project and add your Groq API and Langchain API Key:
```bash
# .env.example
GROQ_API_KEY = your_groq_key_here
LANGCHAIN_API_KEY = your_langchain_key_here
LANGCHAIN_TRACING_V2 = true
LANGSMITH_PROJECT = your_project_name
```
### 5. Run the Assistant

```bash
streamlit run app.py
```

---

## Contributions 🤝
💡 I’m open to feedback, issues, and contributions — feel free to fork, clone, or open a pull request! Contributions for improvements are welcomed!
