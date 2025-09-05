#!/usr/bin/env python3
"""
A utility script to clean Org mode syntax from files.

This script provides functionality to clean Org mode specific syntax elements
from .org files, making them more suitable for processing with LLMs or other
text analysis tools.

Example: Clean a single Org mode file:
BASH
python org_cleaner.py file input.org --output cleaned.txt

Clean all Org mode files in a directory:
BASH
python org_cleaner.py directory /path/to/org/files /path/to/output/directory

"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


class OrgModeLoader(TextLoader):
    """Custom loader for Org mode files with cleaning functionality."""

    def __init__(self, file_path, encoding=None):
        super().__init__(file_path, encoding=encoding)

    def load(self):
        """Load and return documents from Org mode file with Org syntax cleaned."""
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()

        # Clean Org mode specific syntax
        cleaned_text = self._clean_org_syntax(text)

        # Extract title from the file name
        file_name = os.path.basename(self.file_path)
        title = os.path.splitext(file_name)[0]

        metadata = {
            "source": self.file_path,
            "title": title
        }

        return [Document(page_content=cleaned_text, metadata=metadata)]

    def _clean_org_syntax(self, text):
        """Remove Org mode specific syntax elements."""
        # Remove property drawers
        text = re.sub(r':PROPERTIES:\n(.+?)\n:END:', '', text, flags=re.DOTALL)

        # Remove logbook drawers
        text = re.sub(r':LOGBOOK:\n(.+?)\n:END:', '', text, flags=re.DOTALL)

        # Remove other drawers
        text = re.sub(r':(CLOCK|RESULTS|NOTES):\n(.+?)\n:END:', '', text, flags=re.DOTALL)

        # Remove org tags at end of headings
        text = re.sub(r'\s*:[A-Za-z0-9_@:]+:\s*$', '', text, flags=re.MULTILINE)

        # Remove org priorities [#A], [#B], etc.
        text = re.sub(r'\s*\[#[A-Z]\]\s*', ' ', text)

        # Remove org links, keeping the description or link target
        text = re.sub(r'\[\[([^\]]+)\]\[([^\]]+)\]\]', r'\2', text)  # [[link][description]] -> description
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]] -> link

        # Remove org formatting markers
        text = re.sub(r'(\*\*|\/\/|==|~~|_)(.+?)\1', r'\2', text)  # Bold, italic, underline, etc.

        # Remove org comments
        text = re.sub(r'^\s*#\+.*$', '', text, flags=re.MULTILINE)

        # Remove TODO/DONE states
        text = re.sub(r'^\s*\*+\s+(TODO|DONE|WAITING|CANCELLED|IN-PROGRESS)\s+', r'* ', text, flags=re.MULTILINE)

        # Remove timestamps
        text = re.sub(
            r'<\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?>(?:--<\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?>)?',
            '', text)
        text = re.sub(
            r'\[\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?\](?:--\[\d{4}-\d{2}-\d{2}(?: [A-Za-z]+)?(?: \d{2}:\d{2})?(?: [+-]\d+[dwmy])?(?: \d{2}:\d{2})?\])?',
            '', text)

        # Remove checkbox markers
        text = re.sub(r'^\s*- \[([ X])\]\s*', '- ', text, flags=re.MULTILINE)

        # Clean up multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


def clean_org_file(input_file: str, output_file: Optional[str] = None, encoding: str = 'utf-8') -> str:
    """
    Clean Org mode syntax from a single file.
    
    Args:
        input_file: Path to the input Org mode file
        output_file: Path to save the cleaned file (if None, doesn't save to file)
        encoding: File encoding to use
        
    Returns:
        The cleaned text content
    """
    loader = OrgModeLoader(input_file, encoding=encoding)
    documents = loader.load()
    cleaned_text = documents[0].page_content
    
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w', encoding=encoding) as f:
            f.write(cleaned_text)
    
    return cleaned_text


def clean_org_directory(input_dir: str, output_dir: str, encoding: str = 'utf-8') -> List[str]:
    """
    Clean Org mode syntax from all .org files in a directory.
    
    Args:
        input_dir: Directory containing Org mode files
        output_dir: Directory to save cleaned files
        encoding: File encoding to use
        
    Returns:
        List of paths to the cleaned files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    cleaned_files = []
    
    for org_file in input_path.glob('**/*.org'):
        # Create relative path to maintain directory structure
        rel_path = org_file.relative_to(input_path)
        output_file = output_path / rel_path
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean and save the file
        clean_org_file(str(org_file), str(output_file), encoding)
        cleaned_files.append(str(output_file))
        
    return cleaned_files


def create_vector_db(files_directory, persist_directory, embedding_model_name, chunk_size=800, chunk_overlap=100):
    """
    Load files from a directory, split them into chunks, and create a vector database.

    Args:
        files_directory: Directory containing text files
        persist_directory: Directory to save the vector database
        embedding_model_name: Name of the embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        The created vector database
    """
    # Load documents
    print(f"Loading documents from {files_directory}")
    loader = DirectoryLoader(files_directory, glob="**/*.org", loader_cls=OrgModeLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Split documents into chunks
    print("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    print(f"Creating embeddings using {embedding_model_name}")
    embeddings = OllamaEmbeddings(model=embedding_model_name)

    # Create and persist vector database
    print(f"Creating vector database in {persist_directory}")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("Vector database created successfully")
    return db


def main():
    parser = argparse.ArgumentParser(description='Clean Org mode syntax from files')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Clean file command
    file_parser = subparsers.add_parser('file', help='Clean a single Org mode file')
    file_parser.add_argument('input_file', help='Path to the input Org mode file')
    file_parser.add_argument('--output', '-o', help='Path to save the cleaned file (if not specified, prints to stdout)')
    file_parser.add_argument('--encoding', '-e', default='utf-8', help='File encoding (default: utf-8)')
    
    # Clean directory command
    dir_parser = subparsers.add_parser('directory', help='Clean all Org mode files in a directory')
    dir_parser.add_argument('input_dir', help='Directory containing Org mode files')
    dir_parser.add_argument('output_dir', help='Directory to save cleaned files')
    dir_parser.add_argument('--encoding', '-e', default='utf-8', help='File encoding (default: utf-8)')
    
    # Vector DB command
    db_parser = subparsers.add_parser('vectordb', help='Create a vector database from Org mode files')
    db_parser.add_argument('files_dir', help='Directory containing Org mode files')
    db_parser.add_argument('db_dir', help='Directory to save the vector database')
    db_parser.add_argument('--model', '-m', default='mxbai-embed-large', 
                          help='Embedding model name (default: mxbai-embed-large)')
    db_parser.add_argument('--chunk-size', '-c', type=int, default=800, 
                          help='Size of text chunks (default: 800)')
    db_parser.add_argument('--overlap', '-v', type=int, default=100, 
                          help='Overlap between chunks (default: 100)')
    
    args = parser.parse_args()
    
    if args.command == 'file':
        cleaned_text = clean_org_file(args.input_file, args.output, args.encoding)
        if not args.output:
            print(cleaned_text)
        else:
            print(f"Cleaned file saved to: {args.output}")
            
    elif args.command == 'directory':
        cleaned_files = clean_org_directory(args.input_dir, args.output_dir, args.encoding)
        print(f"Cleaned {len(cleaned_files)} files and saved to {args.output_dir}")
        
    elif args.command == 'vectordb':
        db = create_vector_db(
            files_directory=args.files_dir,
            persist_directory=args.db_dir,
            embedding_model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap
        )
        print(f"Vector database created in {args.db_dir}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()