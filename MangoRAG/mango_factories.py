# Stdlib imports
from typing import Callable, Any, List

# 3rd party imports
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings

class VectorStoreWrapper(object):
    """
    Wrapper class to unify initialization logic and persistence
    methods of different vector store classes.
    """
    def __init__(self, db: Any, persist_fn: Callable[[], None]):
        self.db = db
        self.persist = persist_fn
        
def get_vector_store(db_type: str, docs: List[Document], embedder, path2db: str) -> VectorStoreWrapper:
    """
    Returns the vector store specified by db_type.
    
    This function imports the necessary package and
    instantiates and returns the vector store based 
    on the parameter db_type.

    Parameter
    ---------
    db_type : ['weaviate', 'faiss', 'chroma', 'qdrant', 'milvus']
        Name of the vector store
        
    Raises
    ------
    ValueError
        If db_type is not in ['weaviate', 'faiss', 'chroma', 'qdrant', 'milvus'].
    """
    db_type = db_type.lower()
    
    if db_type == "faiss":
        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(docs, embedding=embedder)
        return VectorStoreWrapper(db, persist_fn=lambda: db.save_local(path2db))

    elif db_type == "chroma":
        from langchain.vectorstores.chroma import Chroma
        db = Chroma.from_documents(docs, embedder, persist_directory=path2db)
        return VectorStoreWrapper(db, persist_fn=db.persist)

    elif db_type == "weaviate":
        raise ValueError("Weaviate is currently not supported.")

    elif db_type.lower() == "qdrant":
            raise ValueError("Qdrant is currently not supported.")
            
    elif db_type.lower() == "milvus":
        raise ValueError("Milvus is currently not supported.")

    else:
        raise ValueError("Invalid database type. Must be 'faiss', "
                         + "'weaviate', 'chroma', 'qdrant', or 'milvus'."
                        )

def get_embedding_model(embedding_name: str) -> Embeddings:
    """
    Returns the embedding model specified by embedding_name.
    
    This function imports the necessary package and
    instantiates and returns the embedding model based 
    on the parameter embedding_name.

    Parameter
    ---------
    embedding_name : ['spacy', 'openai', 'nomic']
        Name of the embedding model
        
    Raises
    ------
    ValueError
        If embedding model name is not in ['spacy', 'openai', 'nomic'].
    """
    if embedding_name.lower() == "spacy":
        from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
        return SpacyEmbeddings(model_name="de_core_news_sm")

    elif embedding_name.lower() == "openai":
        raise ValueError("Embedding with OpenAI is currently not supported.")
        #from langchain.embeddings import OpenAIEmbeddings

    elif embedding_name.lower() == "nomic":
        raise ValueError("Embedding with nomic is currently not supported.")

    else:
        raise ValueError("Invalid embedding model. Must be 'spacy', 'openai', or 'nomic'.")
            
def get_llm(llm_name: str) -> BaseChatModel:
    """
    Returns the chat model specified by llm_name.
    
    This function imports the necessary package and
    instantiates and returns the LLM based on the
    parameter llm_name.

    Parameter
    ---------
    llm_name : ['openai', 'ollama']
        Name of the chat model
        
    Raises
    ------
    ValueError
        If chat model name is not in ['openai', 'ollama'].
    """
    if llm_name == "openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI()
    elif llm_name == "ollama":
        from langchain.chat_models import ChatOllama
        return ChatOllama()
    else:
        raise ValueError("Invalid chat model name. Must be 'openai' or 'ollama'.")