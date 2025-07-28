import os
import re
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def connect_to_vector_db(persist_directory, embedding_model_name):
    """
    Connect to an existing vector database.

    Args:
        persist_directory: Directory where the vector database is stored
        embedding_model_name: Name of the embedding model to use

    Returns:
        The loaded vector database
    """
    print(f"Connecting to vector database at {persist_directory}")
    embeddings = OllamaEmbeddings(model=embedding_model_name)

    # Load the persisted database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(f"Successfully connected to database with {db._collection.count()} documents")
    return db


def setup_rag_chain(db, llm_model_name):
    """
    Set up a RAG chain for answering questions.

    Args:
        db: The vector database
        llm_model_name: Name of the LLM to use

    Returns:
        A QA chain
    """
    # Create a retriever from the vector database
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Initialize the LLM
    llm = OllamaLLM(model=llm_model_name)

    # Create a custom prompt template
    template = """
    You are an assistant that helps users find information in their personal notes.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep your answers concise and to the point.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def format_source_documents(source_docs):
    """Format source documents for display."""
    formatted_sources = []
    seen_sources = set()

    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        # Only include each source once
        if source not in seen_sources:
            seen_sources.add(source)
            file_name = os.path.basename(source)
            formatted_sources.append(f"- {file_name}")

    return "\n".join(formatted_sources)


def ask_question(qa_chain, question):
    """
    Ask a question and get an answer using the RAG chain.

    Args:
        qa_chain: The QA chain
        question: The question to ask

    Returns:
        The answer
    """
    print(f"\nQuestion: {question}")
    print("Thinking...")

    # Get the answer
    result = qa_chain({"query": question})


    print("\n" + "=" * 50)
    print("Answer:")
    print(result["result"])
    print("\nSources:")
    print(format_source_documents(result["source_documents"]))
    print("=" * 50)

    return result


def interactive_qa_session(qa_chain):
    """Run an interactive Q&A session."""
    print("\nWelcome to your Org Notes Assistant!")
    print("Ask questions about your notes or type 'exit' to quit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        ask_question(qa_chain, question)


if __name__ == "__main__":
    # Configuration
    persist_directory = "./db"
    embedding_model_name = "mxbai-embed-large"
    llm_model_name = "llama3"  # or any other model you have in Ollama

    # Connect to the database
    db = connect_to_vector_db(persist_directory, embedding_model_name)

    # Set up the RAG chain
    qa_chain = setup_rag_chain(db, llm_model_name)

    # test: What are the main topics in my notes?

    # Check if a question was provided as a command-line argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_question(qa_chain, question)
    else:
        # Start interactive session
        interactive_qa_session(qa_chain)