import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------
# 1. Data Ingestion
# ---------------------------------------------------------
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# ---------------------------------------------------------
# 2. Text Splitter
# ---------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

# ---------------------------------------------------------
# 3. Embeddings
# Free local Hugging Face embedding model.
# The model is downloaded once and then reused locally.
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------
# 4. Vector Store
# ---------------------------------------------------------
vector_store = FAISS.from_documents(chunks, embeddings)

# ---------------------------------------------------------
# 5. Retriever
# ---------------------------------------------------------
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------------------------------------------------
# 6. Chain
# ---------------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the user's question using only the supplied context.
If the context does not contain the answer, say that the answer is not
available in the provided document.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return response.content, docs


# ---------------------------------------------------------
# Run three questions.
#
# This also satisfies the original Week 2 Day 3 requirement:
# at least three prompts in one run, with LangSmith tracing
# enabled entirely through environment variables.
# ---------------------------------------------------------
questions = [
    "What is the main purpose of LangChain?",
    "What does a vector store do?",
    "Why is a retriever useful in a LangChain application?",
]

print(f"Loaded documents : {len(documents)}")
print(f"Created chunks   : {len(chunks)}")
print("Vector store     : FAISS")
print("Embeddings       : all-MiniLM-L6-v2 (local)")
print()

for index, question in enumerate(questions, start=1):
    answer, source_docs = answer_question(question)

    print(f"--- Run {index} ---")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Retrieved chunks: {len(source_docs)}")
    print()
