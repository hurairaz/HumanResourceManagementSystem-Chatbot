import logging
import os
import warnings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, GPTVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, SimpleDirectoryReader

from langchain_g4f import G4FLLM
from g4f import models

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def resolve_documents_path(directory_name):
    """
    Resolve the path to the documents directory.

    Args:
        directory_name (str): Name of the directory containing documents.

    Returns:
        str: Absolute path to the documents directory.

    Raises:
        NotADirectoryError: If the directory does not exist or is not a directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # If your Python file is in a subdirectory, you may want to move up to the parent directory
    # Uncomment the following lines if needed:
    # parent_dir = os.path.join(current_dir, "..")
    # documents_dir = os.path.join(parent_dir, directory_name)

    # If your Python file is in the current(parent) directory, use the following:
    documents_dir = os.path.join(current_dir, directory_name)


    if not os.path.isdir(documents_dir) and os.listdir(documents_dir):
        raise NotADirectoryError(f"The directory '{directory_name}' does not exist or is not a directory.")

    return documents_dir

def initialize_embedding_model(model_name):
    """
    Initialize the Hugging Face embedding model.

    Args:
        model_name (str): Name of the Hugging Face model to use.

    Returns:
        HuggingFaceEmbedding: Initialized embedding model.
    """
    try:
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        return embed_model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise

def load_documents(directory):
    """
    Load documents from the specified directory.

    Args:
        directory (str): Directory containing documents.

    Returns:
        SimpleDirectoryReader: Reader object for accessing documents.
    """
    try:
        documents = SimpleDirectoryReader(directory).load_data()
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise


def insert_documents_into_index(index, documents):
    """
    Splits documents into chunks and inserts them into the index.

    Args:
        index (GPTVectorStoreIndex): The vector store index to insert documents into.
        documents: List of documents to process.

    Returns:
        GPTVectorStoreIndex: The updated index after inserting the documents.
    """
    sentence_splitter = SentenceSplitter(chunk_size=150, chunk_overlap=20)
    chunks = sentence_splitter.get_nodes_from_documents(documents=documents)
    index.insert_nodes(chunks)
    return index


def create_pinecone_vector_store():
    """
    Creates or retrieves a Pinecone vector store.

    Returns:
        PineconeVectorStore: The Pinecone vector store instance.
    """
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("Pinecone API key is not set.")
        raise ValueError("Pinecone API key is missing.")

    pinecone_instance = Pinecone(api_key=pinecone_api_key)
    pinecone_index_name = "hrms-index"

    if pinecone_index_name not in pinecone_instance.list_indexes().names():
        logger.debug(f"Creating a new index: {pinecone_index_name}")
        pinecone_instance.create_index(
            name=pinecone_index_name,
            dimension=768,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logger.debug(f"Index '{pinecone_index_name}' already exists.")

    pinecone_index = pinecone_instance.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


def configure_storage_context(vector_store):
    """
    Configures the storage context using the provided vector store.

    Args:
        vector_store (PineconeVectorStore): The vector store to use for storage context.

    Returns:
        StorageContext: The configured storage context.
    """
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
    except Exception as e:
        logger.error(f"Failed to configure storage context: {e}")
        raise


def create_vector_index(llm, vector_store, embed_model):
    """
    Creates a vector store index from documents.

    Args:
        llm (G4FLLM): The language model to use.
        vector_store (PineconeVectorStore): The vector store to use for the index.
        embed_model: The embedding model to use.

    Returns:
        GPTVectorStoreIndex: The created vector store index.
    """
    try:
        index = GPTVectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            llm=llm
        )
        return index
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        raise


def generate_query_prompt(question, context):
    """
    Generate a query prompt for the system.

    Args:
        question (str): User's question.
        context (str): Document context to use for the query.

    Returns:
        str: Generated prompt for the query.
    """
    return f"""
        You are an AI assistant for a Human Resource Management System (HRMS), responsible for delivering clear and precise answers about company policies based on the provided context. Your responses should be thorough, professional, and maintain an approachable tone.

        **Guidelines:**
        1. Context-Based Responses: Always base your answer strictly on the provided context to ensure relevance.
        2. Completeness: Make sure your answers are comprehensive and coherent. Avoid including incomplete or broken sentences.
        3. Accuracy: Ensure that the information provided is accurate and aligns with the context.
        4. Relevance: If the context doesn’t cover the question, provide the most relevant information available or suggest asking a more specific question.
        5. Clarity: Maintain clear and concise language throughout your response to avoid any ambiguity.
        6. Tone: Keep your tone professional yet approachable, ensuring a human-like interaction.
        7. Handling Greetings: Respond appropriately and helpfully to simple greetings or casual inquiries without deviating from the context.
        8. Word Limit: Keep your response concise, with a maximum of 400 characters, unless the question requires a more detailed explanation.
        9. Format: Structure your response in a single, well-organized paragraph, making it easy to read.
        10. Language: Always respond in clear and correct English, avoiding jargon unless necessary.

        Context: {context}
        User Input: {question}
    """


def query_vector_index(documents, index, query):
    """
    Query the vector store index and return the response.

    Args:
        documents: Documents to use for generating context.
        index (GPTVectorStoreIndex): Vector store index to query.
        query (str): Query string.

    Returns:
        str: Response from the index.
    """
    try:
        query_engine = index.as_query_engine()
        context = "\n".join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])
        prompt = generate_query_prompt(query, context)
        response = query_engine.query(prompt)
        return response
    except Exception as e:
        logger.error(f"Failed to query index: {e}")
        raise

def main():
    # Load documents from the specified directory
    documents_dir = resolve_documents_path(directory_name="data")
    documents = load_documents(directory=documents_dir)

    # Create Pinecone vector store and storage context
    vector_store = create_pinecone_vector_store()
    storage_context = configure_storage_context(vector_store=vector_store)

    # Initialize embedding model and language model
    model_name = "bert-base-nli-mean-tokens"
    embed_model = initialize_embedding_model(model_name=model_name)

    llm = G4FLLM(model=models.default)
    Settings.llm = llm

    # Create vector index and insert documents
    index = create_vector_index(llm=llm, vector_store=vector_store, embed_model=embed_model)
    index = insert_documents_into_index(index=index, documents=documents)

    response = query_vector_index(documents, index, "Tell me about the ownership of the assets")
    if response:
        print("__________________________________________")
        print("LlamaIndex response:", response)
    else:
        print("No response from LlamaIndex.")


if __name__ == "__main__":
    main()
