langgraph
langchain
langchain-community
langchain-groq
faiss-cpu
sentence-transformers


from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# --------------------------------------------------
# 1. Load & split document (ONLY source of truth)
# --------------------------------------------------

loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)


# --------------------------------------------------
# 2. Embeddings (Groq does not provide embeddings)
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

# MMR retriever for better diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15,
        "lambda_mult": 0.7
    }
)


# --------------------------------------------------
# 3. Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0
)


# --------------------------------------------------
# 4. LangGraph State
# --------------------------------------------------

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    relevance_score: float
    compressed_context: str
    answer: str


# --------------------------------------------------
# 5. Retrieve Node
# --------------------------------------------------

def retrieve(state: RAGState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


# --------------------------------------------------
# 6. Relevance Scoring Node
# --------------------------------------------------

def score_relevance(state: RAGState):
    if not state["documents"]:
        return {"relevance_score": 0.0}

    context = "\n\n".join(d.page_content for d in state["documents"])

    prompt = f"""
Score how relevant the following context is for answering the question.
Return ONLY a number between 0 and 1.

Context:
{context}

Question:
{state["question"]}
"""

    response = llm.invoke(prompt).content.strip()

    try:
        score = float(response)
    except:
        score = 0.0

    return {"relevance_score": score}


# --------------------------------------------------
# 7. Relevance Router
# --------------------------------------------------

def relevance_route(state: RAGState):
    if state["relevance_score"] < 0.6:
        return "no_docs"
    return "compress"


# --------------------------------------------------
# 8. Context Compression Node
# --------------------------------------------------

def compress_context(state: RAGState):
    text = "\n\n".join(d.page_content for d in state["documents"])

    prompt = f"""
Extract ONLY the information strictly necessary to answer the question.
Remove repetition, examples, and unrelated details.

Question:
{state["question"]}

Text:
{text}
"""

    compressed = llm.invoke(prompt).content
    return {"compressed_context": compressed}


# --------------------------------------------------
# 9. Answer Generation Node (STRICT)
# --------------------------------------------------

def generate(state: RAGState):
    prompt = f"""
You are a document-restricted assistant.

Rules:
- Answer ONLY using the context below
- If the answer is missing, say exactly:
  "I don't know based on the provided document."
- Be concise, precise, and factual
- Do NOT add assumptions or external knowledge

Context:
{state["compressed_context"]}

Question:
{state["question"]}
"""

    response = llm.invoke(prompt).content
    return {"answer": response}


# --------------------------------------------------
# 10. No-docs Fallback Node
# --------------------------------------------------

def no_docs(state: RAGState):
    return {
        "answer": "I don't know based on the provided document."
    }


# --------------------------------------------------
# 11. Build LangGraph
# --------------------------------------------------

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("score", score_relevance)
graph.add_node("compress", compress_context)
graph.add_node("generate", generate)
graph.add_node("no_docs", no_docs)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "score")

graph.add_conditional_edges(
    "score",
    relevance_route,
    {
        "compress": "compress",
        "no_docs": "no_docs"
    }
)

graph.add_edge("compress", "generate")
graph.add_edge("generate", END)
graph.add_edge("no_docs", END)

rag_app = graph.compile()


# --------------------------------------------------
# 12. Run
# --------------------------------------------------

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        result = rag_app.invoke({"question": question})
        print("\nAnswer:\n", result["answer"])
