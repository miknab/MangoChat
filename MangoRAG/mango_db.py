# Stdlib imports
import os
import shutil

# 3rd party imports
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema import Document


class MangoDB(object):
    """
    The class MangoDB creates the vector store containing the embedded
    chunks extracted from documents.
    
    MangoDB reads in documents stored in a specified location (path2docs),
    chunks them, embeds the chunks, and stores the resulting embeddings
    in a vector store.
    
    Methods
    -------
    self.read_docs()
    self.chunk_docs()
    self.create_db()
    """
    def __init__(self):
        self.raw_docs: List[Document] = None
        self.chunked_docs: List[Document] = None
    
    def read_docs(self, path2docs: str) -> None:
        """
        Reads in the documents found in path2docs and stores
        the resulting list of documents into the attribute
        self.raw_docs.
        
        Parameters
        ----------
        path2docs : str
            path to the documents to be embedded
        
        Returns
        -------
        None
        """
        # Initialize PDF loader with specified directory
        document_loader = PyPDFDirectoryLoader(path2docs) 
        # Load PDF documents and store them as a list into self.raw_docs
        self.raw_docs = document_loader.load() 
    
    def chunk_docs(self, strategy: str, size: int, overlap: int) -> None:
        """
        Chunks the documents in self.raw_docs and stores the
        result into self.chunked_docs.
        
        Parameters
        ----------
        strategy : ['token', 'sentence', 'semantics']
            Strategy used to chunk the documents.
            
        size : int
            number of tokens per chunk (chunk_size)
            
        overlap : int
            number of tokens used in chunk overlap
            between consecutive chunks
            
        Returns
        -------
        None
        """
        if self.raw_docs is None:
            raise ValueError("No raw documents found.")
    
        if strategy.lower() == "token":
            from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTSplitter
            # Initialize text splitter with specified parameters
            chunker = RCTSplitter(chunk_size=size,
                                  chunk_overlap=overlap,
                                  length_function=len,  # Function to compute the length of the text
                                  add_start_index=True,  # Flag to add start index to each chunk
                                  )
            
        elif strategy.lower() == "sentence":
            raise ValueError("Sentence-based chunking is currently not supported.")
            
        elif strategy.lower() == "semantics":
            raise ValueError("Semantics-based chunking is currently not supported.")
            
        else:
            raise ValueError("Invalid chunking strategy. Must be 'token', 'sentence', or 'semantics'.")

        # Split documents into smaller chunks using text splitter
        chunks = chunker.split_documents(self.raw_docs)
        print(f"Split {len(self.raw_docs)} documents into {len(chunks)} chunks.")

        # Print example of page content and metadata for a chunk
        first_page = chunks[0]
        print(first_page.page_content)
        print(first_page.metadata)

        self.chunked_docs = chunks # Return the list of split text chunks
    
    def create_db(self, path2db: str, embedding_model: str, db_type: str) -> None:
        """
        Embeds the chunked documents using the specified embedding model
        and stores the chunk embeddings in a vector store.
        
        Parameters
        ----------    
        path2db : str
            path to where the resulting vector store
            shall be written
            
        embedding_model : ['spacy', 'openai', 'nomic']
            Model used to embed the chunks
        
        db_type: ['weaviate', 'faiss', 'chroma', 'qdrant', 'milvus']
            Type of database used to create the vector store
        """
        # Sanity check (status)
        if self.chunked_docs is None:
            raise ValueError("No chunked documents found.")
            
        # Clear out the existing database directory if it exists
        if os.path.exists(path2db):
            shutil.rmtree(path2db)
         
        # Specify embedding model
        if embedding_model.lower() == "spacy":
            from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
            embedder = SpacyEmbeddings(model_name="de_core_web_sm")

        elif embedding_model.lower() == "openai":
            raise ValueError("Embedding with OpenAI is currently not supported.")
            from langchain.embeddings import OpenAIEmbeddings
            
        elif embedding_model.lower() == "nomic":
            raise ValueError("Embedding with nomic is currently not supported.")
            
        else:
            raise ValueError("Invalid embedding model. Must be 'spacy', 'openai', or 'nomic'.")
        
        # Specify database type
        if db_type.lower() == "faiss":
            from langchain_community.vectorstores import FAISS
            
            db = FAISS.from_documents(self.chunked_docs, embedding=embedder)
            db.save_local(path2db)
            
        elif db_type.lower() == "weaviate":
            raise ValueError("Weaviate is currently not supported.")
            
        elif db_type.lower() == "chroma":
            from langchain.vectorstores.chroma import Chroma
            # Create a new Chroma database from the documents using OpenAI embeddings
            db = Chroma.from_documents(self.chunked_docs, embedder, persist_directory=path2db)
            # Persist the database to disk
            db.persist()

        elif db_type.lower() == "qdrant":
            raise ValueError("Qdrant is currently not supported.")
            
        elif db_type.lower() == "milvus":
            raise ValueError("Milvus is currently not supported.")
            
        else:
            raise ValueError("Invalid database type. Must be 'faiss', "
                             + "'weaviate', 'chroma', 'qdrant', or 'milvus'."
                            )
        
        # Output message
        print(f"Saved {len(self.chunked_docs)} chunks to {path2db}.")
