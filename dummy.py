import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_environment_variables():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    return pinecone_api_key

def get_documents_directory(documents_directory_name):
    # Get the current directory of the Python file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # If your Python file is in a subdirectory, you may want to move up to the parent directory
    # Uncomment the following lines if needed:
    # parent_dir = os.path.join(current_dir, "..")
    # documents_dir = os.path.join(parent_dir, directory_name)

    # If your Python file is in the current(parent) directory, use the following:
    documents_dir = os.path.join(current_dir, documents_directory_name)

    # Check if the data directory exists and is a directory
    if not os.path.isdir(documents_dir):
        raise NotADirectoryError(f"The directory '{documents_directory_name}' does not exist or is not a directory.")

    return documents_dir

def split_documents_into_chunks(documents_dir):
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    split_documents = []

    # Iterate over all files in the directory
    for file_name in os.listdir(documents_dir):
        if file_name.endswith(".pdf"):  # Check if the file is a PDF
            file_path = os.path.join(documents_dir, file_name)

            # Read the PDF file and extract text
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            # Create a Document object with the extracted text and metadata
            document = Document(page_content=text, metadata={"source": file_path})

            # Split the document into chunks and add them to the list
            split_documents.extend(text_splitter.split_documents([document]))

    return split_documents

def main():
    documents_dir = get_documents_directory("data")

    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    index_name = "document-embeddings"



if __name__ == "__main__":
    main()