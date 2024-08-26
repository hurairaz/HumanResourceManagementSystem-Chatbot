import logging
import warnings
import os
from llama_index.legacy import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.legacy import StorageContext, load_index_from_storage, ServiceContext, set_global_service_context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_g4f import G4FLLM
from g4f import models

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, message=r'`clean_up_tokenization_spaces` was not set')
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def configure_service_context(embed_model, llm):
    """
    Configure the ServiceContext with the embedding model and LLM.

    Args:
        embed_model (HuggingFaceEmbedding): Embedding model to use.
        llm: Language model (LLM) to integrate.

    Returns:
        ServiceContext: Configured service context.
    """
    try:
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            chunk_size=800,
            chunk_overlap=20,
            llm=llm,
        )
        set_global_service_context(service_context)
        return service_context
    except Exception as e:
        logger.error(f"Failed to configure service context: {e}")
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


def create_vector_index(documents, service_context):
    """
    Create a vector store index from documents.

    Args:
        documents: Documents to index.
        service_context (ServiceContext): Context to use for the index.

    Returns:
        GPTVectorStoreIndex: Created vector store index.
    """
    try:
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        return index
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        raise


def load_vector_index(storage_directory, service_context):
    """
    Load a vector store index from storage.

    Args:
        storage_directory (str): Directory to load the index from.
        service_context (ServiceContext): Context to use for the index.

    Returns:
        GPTVectorStoreIndex: Loaded vector store index.
    """
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_directory)
        index = load_index_from_storage(
            storage_context=storage_context, service_context=service_context
        )
        return index
    except Exception as e:
        logger.error(f"Failed to load index from storage: {e}")
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
    documents_dir = resolve_documents_path("data")
    documents = load_documents(documents_dir)

    model_name = "bert-base-nli-mean-tokens"
    embed_model = initialize_embedding_model(model_name)

    llm = G4FLLM(model=models.default)  # Assuming G4FLLM is defined elsewhere
    service_context = configure_service_context(embed_model, llm)

    storage_dir = "./storage"

    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        logger.debug("Index exists. Loading from storage.")
        index = load_vector_index(storage_dir, service_context)
    else:
        logger.debug("No existing index found. Creating and persisting a new index.")
        index = create_vector_index(documents, service_context)
        index.storage_context.persist(persist_dir=storage_dir)

    response = query_vector_index(documents, index, "Tell me about the maternity leave")
    if response:
        print("__________________________________________")
        print("LlamaIndex response:", response)
    else:
        print("No response from LlamaIndex.")


if __name__ == "__main__":
    main()