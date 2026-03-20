import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"   # free, runs locally — no API needed
GROQ_MODEL  = "llama-3.1-8b-instant"     # fast & free on Groq


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def ingest_documents(file_paths: list[str]) -> int:
    """Load, chunk, embed and store documents. Returns chunk count."""
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return len(chunks)


def build_qa_chain(k: int = 4, api_key: str = ""):
    """Build and return a RetrievalQA chain powered by Groq."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    # No need to persist on every load

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in environment or passed as api_key")
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use ONLY the context below to answer
the question. If the answer is not in the context, say "I could not find that in
the uploaded documents."

Context: {context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "retriever": retriever}


def has_index() -> bool:
    """True if a valid ChromaDB index already exists."""
    if not os.path.isdir(CHROMA_DIR):
        return False
    chroma_files = ["chroma.sqlite3", "index"]
    return any(os.path.exists(os.path.join(CHROMA_DIR, f)) for f in chroma_files)
