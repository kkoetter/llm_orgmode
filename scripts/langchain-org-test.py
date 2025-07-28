from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
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


if __name__ == "__main__":
    print("import os")

    chunk_size = 800
    chunk_overlap = 100
    top_k = 6

    # Configuration
    persist_directory = "./db"
    org_files_directory = "/Users/katharinakotter/python_code/llm_orgmode/Examples/testnotes/"
    embedding_model_name = "mxbai-embed-large"

    # Create vector database
    db = create_vector_db(
        files_directory=org_files_directory,
        persist_directory=persist_directory,
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )