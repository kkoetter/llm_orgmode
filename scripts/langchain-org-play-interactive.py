from langchain_community.document_loaders import UnstructuredOrgModeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import sys
import argparse
from rich.console import Console
from rich.markdown import Markdown

# Initialize rich console for better formatting
console = Console()

# Add command-line arguments
parser = argparse.ArgumentParser(description='RAG system for org files')
parser.add_argument('--regenerate', action='store_true', help='Force regeneration of vector store')
args = parser.parse_args()

# Define paths
persist_directory = "./langchain_my_index"
org_file_path = "/home/volhovm/org/org-markdown/tmp/test-therapy.org"

# Create embeddings model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

# Check if vector store already exists
if not args.regenerate and os.path.exists(persist_directory) and os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0:
    console.print("[bold green]Loading existing vector store...[/bold green]")
    # Load existing vector store
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    console.print("[bold yellow]Creating new vector store...[/bold yellow]")
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

# Create a chat-style CLI interface
def chat_interface():
    console.print("[bold blue]Welcome to OrgChat![/bold blue]")
    console.print("Ask questions about your org files. Type 'exit' or 'quit' to end the session.")
    console.print("Type 'debug on' to see retrieved documents, 'debug off' to hide them.")

    debug_mode = False
    chat_history = []

    while True:
        # Get user input
        console.print("\n[bold cyan]You:[/bold cyan] ", end="")
        question = input()

        # Check for exit commands
        if question.lower() in ['exit', 'quit', 'q']:
            console.print("[bold blue]Goodbye![/bold blue]")
            break

        # Check for debug commands
        if question.lower() == 'debug on':
            debug_mode = True
            console.print("[italic]Debug mode enabled. Retrieved documents will be shown.[/italic]")
            continue
        elif question.lower() == 'debug off':
            debug_mode = False
            console.print("[italic]Debug mode disabled.[/italic]")
            continue

        # Add to chat history
        chat_history.append({"role": "user", "content": question})

        try:
            # Retrieve documents
            docs = retriever.invoke(question)

            # Show retrieved documents in debug mode
            if debug_mode:
                console.print("\n[bold yellow]Retrieved Documents:[/bold yellow]")
                for i, doc in enumerate(docs):
                    console.print(f"[bold]Document {i+1}:[/bold]")
                    console.print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    console.print()

            # Generate response
            console.print("[bold green]Assistant:[/bold green] ", end="")
            sys.stdout.flush()  # Ensure the prompt is displayed

            result = rag_chain.invoke(question)

            # Print the result as markdown for better formatting
            console.print(Markdown(result))

            # Add to chat history
            chat_history.append({"role": "assistant", "content": result})

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

# Run the chat interface
if __name__ == "__main__":
    chat_interface()
