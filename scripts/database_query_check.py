import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def query_vector_db(query_text, persist_directory, embedding_model_name, k=5):
    """
    Query the vector database for similar documents.

    Args:
        query_text: The query text
        persist_directory: Directory where the vector database is stored
        embedding_model_name: Name of the embedding model to use
        k: Number of results to return

    Returns:
        List of documents and their similarity scores
    """
    print(f"Loading vector database from {persist_directory}")
    embeddings = OllamaEmbeddings(model=embedding_model_name)

    # Load the persisted database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(f"Querying database with: '{query_text}'")
    results = db.similarity_search_with_score(query_text, k=k)

    return results


def display_results(results):
    """Display the query results in a readable format."""
    print("\n===== QUERY RESULTS =====\n")

    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Similarity: {score:.4f})")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)
        print("-" * 50)


if __name__ == "__main__":
    # Configuration
    persist_directory = "./db"
    embedding_model_name = "mxbai-embed-large"

    # Example query
    query = "What are the main topics in my notes?"

    # Query the database
    results = query_vector_db(
        query_text=query,
        persist_directory=persist_directory,
        embedding_model_name=embedding_model_name,
        k=5  # Return top 3 results

    )

    # Display results
    display_results(results)
