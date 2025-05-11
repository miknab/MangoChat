# Stdlib imports
from typing import Callable, Any, List
import os

# 3rd party imports
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStore

class VectorStoreAdapter(object):
    """
    Adapter class to unify initialization logic and persistence
    methods of different vector store classes.
    
    Parameter:
    ----------
    store : Any
        Vector store instance
    
    persist_fn : Callable[[], None]
        Persistence function specific to the vector store in self.store.
    
    add_fn : Callable[[list], None]
        Document adding function specific to the vector store in self.store.
    
    Methods:
    --------
    self.add_documents(docs: List[Document]) -> None
    self.persist() -> None
    """
    def __init__(self, store: Any, persist_fn: Callable[[], None]|None, add_fn: Callable[[list], None]) -> None:
        self.store = store
        self._persist = persist_fn
        self._add = add_fn

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add documents to vector store.
        
        Parameter:
        ----------
        docs : List[Document]
            A list of langchain documents.
        """
        self._add(docs)

    def persist(self) -> None:
        """
        Write vector store to disc.
        
        Parameter:
        ----------
        None
        """
        if self._persist:
            self._persist()
        
    def __getattr__(self, name: str) -> Any:
        """
        Forwards attributes of self.store to VectorStoreAdapter.
        
        Parameter:
        ----------
        name : str
            Name of self.store attribute
            
        Returns:
        --------
        Any
            Attribute of self.store
        """
        # TODO: make this type safe
        return getattr(self.store, name)
        
def get_vector_store(db_type: str, embedding: Embeddings, path2db: str, load: bool = False) -> VectorStoreAdapter:
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
        if load:
            db = FAISS.load_local(path2db, embeddings=embedding, allow_dangerous_deserialization=True)
        else:
            db = FAISS(embedding_function=embedding)
        
        return VectorStoreAdapter(store=db,
                                  persist_fn=lambda: db.save_local(path2db),
                                  add_fn=db.add_documents
                                 )

    elif db_type == "chroma":
        from langchain_chroma import Chroma
        db = Chroma(embedding_function=embedding, persist_directory=path2db)
        
        return VectorStoreAdapter(store=db,
                                  persist_fn=None,
                                  add_fn=db.add_documents
                                 )

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
            
def get_llm(llm_name: str, temperature: float=1) -> BaseChatModel:
    """
    Returns the chat model specified by llm_name.
    
    This function imports the necessary package and
    instantiates and returns the LLM based on the
    parameter llm_name.

    Parameter
    ---------
    llm_name : ['llama[version], 'gpt-[version]']
        Name of the chat model
        
    Optional Parameter
    ------------------
    temperature : float
        Temperature of the LLM.
        
    Raises
    ------
    ValueError
        If chat model name is not in ['openai', 'ollama'].
    """
    if "gpt" in llm_name:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=llm_name, 
                          temperature = temperature,
                          openai_api_key=os.environ['OPENAI_API_KEY']
                         )
    elif "llama" in llm_name:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=llm_name, 
                          temperature=temperature)
    elif "fake" in llm_name:
        from langchain.llms.fake import FakeListLLM
        return FakeListLLM(responses=["Rewritten Query"])
    else:
        raise ValueError("Invalid chat model name. Must be 'openai' or 'ollama'.")