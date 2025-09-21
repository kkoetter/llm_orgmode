
#!/usr/bin/env python3
"""
create_vector_db.py - Script to create a vector database from Org mode files.

This script loads Org mode files, cleans them using the OrgModeLoader,
splits them into chunks, and creates a vector database for semantic search.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Import the OrgModeLoader from our module
from orgmode_syntax_cleaner import OrgModeLoader


def create_vector_db(
    files_directory: str, 
    persist_directory: str, 
    embedding_model_name: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    glob_pattern: str = "**/*.org",
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Chroma:
    """
    Load Org mode files from a directory, split them into chunks, and create a vector database.

    Args:
        files_directory: Directory containing Org mode files
        persist_directory: Directory to save the vector database
        embedding_model_name: Name of the embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        glob_pattern: Pattern to match files
        verbose: Whether to print detailed information
        config: Additional configuration for the embedding model

    Returns:
        The created vector database
    """
    start_time = time.time()
    
    # Ensure directories exist
    files_path = Path(files_directory)
    if not files_path.exists():
        raise FileNotFoundError(f"Directory not found: {files_directory}")
    
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # Load documents
    print(f"Loading Org mode documents from {files_directory}")
    loader = DirectoryLoader(
        files_directory,
        glob=glob_pattern,
        loader_cls=lambda file_path: OrgModeLoader(file_path, verbose=verbose)
    )
    
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        raise
    
    if not documents:
        print(f"No documents found in {files_directory} matching pattern {glob_pattern}")
        return None
    
    # Split documents into chunks
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n* ", "\n** ", "\n*** ", "\n", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Print a sample chunk to verify content
    if chunks and verbose:
        print("\nSample chunk content:")
        print(f"Source: {chunks[0].metadata.get('source')}")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Chunk sizes: min={min(len(c.page_content) for c in chunks)}, "
              f"max={max(len(c.page_content) for c in chunks)}, "
              f"avg={sum(len(c.page_content) for c in chunks) / len(chunks):.1f}")

    # Create embeddings
    print(f"Creating embeddings using {embedding_model_name}")
    embedding_config = config or {}
    
    try:
        embeddings = OllamaEmbeddings(model=embedding_model_name, **embedding_config)
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        raise

    # Create and persist vector database
    print(f"Creating vector database in {persist_directory}")
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        db.persist()
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        raise

    elapsed_time = time.time() - start_time
    print(f"Vector database created successfully in {elapsed_time:.2f} seconds")
    print(f"Total documents: {len(documents)}, Total chunks: {len(chunks)}")
    
    return db


def main():
    """Parse command line arguments and create the vector database."""
    parser = argparse.ArgumentParser(
        description="Create a vector database from Org mode files"
    )
    
    parser.add_argument(
        "--files-dir", "-f", 
        required=True,
        help="Directory containing Org mode files"
    )
    
    parser.add_argument(
        "--db-dir", "-d", 
        required=True,
        help="Directory to save the vector database"
    )
    
    parser.add_argument(
        "--model", "-m", 
        default="mxbai-embed-large",
        help="Embedding model name (default: mxbai-embed-large)"
    )
    
    parser.add_argument(
        "--chunk-size", "-c", 
        type=int, 
        default=800,
        help="Size of text chunks (default: 800)"
    )
    
    parser.add_argument(
        "--overlap", "-o", 
        type=int, 
        default=100,
        help="Overlap between chunks (default: 100)"
    )
    
    parser.add_argument(
        "--glob", "-g", 
        default="**/*.org",
        help="Glob pattern for finding files (default: **/*.org)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information during processing"
    )
    
    args = parser.parse_args()
    
    # Create vector database with parsed arguments
    db = create_vector_db(
        files_directory=args.files_dir,
        persist_directory=args.db_dir,
        embedding_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        glob_pattern=args.glob,
        verbose=args.verbose
    )
    
    return db


if __name__ == "__main__":
    # If running as a script, call the main function
    main()