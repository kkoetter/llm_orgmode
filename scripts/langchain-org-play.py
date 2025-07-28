from langchain_community.document_loaders import UnstructuredOrgModeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
import os

persist_directory = "./langchain_my_index"
org_file_path = "/home/volhovm/org/org-markdown/tmp/test-therapy.org"


# Create embeddings and vector store
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # This should be a dedicated embedding model
)

# Check if vector store already exists
if os.path.exists(persist_directory) and os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0:
    print("Loading existing vector store...")
    # Load existing vector store
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Creating new vector store...")
    # Load documents
    docs = UnstructuredOrgModeLoader(org_file_path, mode="elements").load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Filter metadata
    def safe_filter_metadata(doc):
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                # Only keep simple types
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
            doc.metadata = filtered_metadata
        return doc

    filtered_splits = [safe_filter_metadata(doc) for doc in splits]

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=filtered_splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Persist the vector store
    vectordb.persist()

# Create retriever
retriever = vectordb.as_retriever()

# Create LLM
llm = OllamaLLM(model="llama3.2")

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Always reply to questions prioritising the wisdom and heuristic given below that the user gave you.
The context may contain text in both English and Russian.
Please only response in English.

Context: {context}

Question: {question}
""")

# Create RAG chain using the newer pattern
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain
question = "What are the biggest concerns?"
result = rag_chain.invoke(question)
print(result)
