# Stdlib imports
import os
import shutil

# 3rd party imports
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain_core.embeddings.embeddings import Embeddings

# Local imports
from MangoRAG import mango_factories as factories


class MangoDB(object):
    """
    The class MangoDB creates the vector store containing the embedded
    chunks extracted from documents.
    
    MangoDB reads in documents stored in a specified location (path2docs),
    chunks them, embeds the chunks, and stores the resulting embeddings
    in a vector store.
    
    Attributes
    ----------
    self.raw_docs : List[Document]
        List of documents from the text corpus used by the RAG as context.
        
    self.chunked_docs : List[Document]
        List of chunked documents from the text corpus used by the RAG as
        context.
        
    self.embedder : langchain_core.embeddings.embeddings.Embeddings
        Embedding model used to create the vector store.
        
    Methods
    -------
    self.read_docs()
    self.chunk_docs()
    self.create_db()
    """
    def __init__(self, embedding_model: str):
        self.raw_docs: List[Document] = None
        self.chunked_docs: List[Document] = None
        
        # Specify embedding model
        self.embedder: Embeddings = factories.get_embedding_model(embedding_model)
    
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
    
    def chunk_docs(self, 
                   strategy: str, 
                   size: int|None = None, 
                   overlap: int|None = None
                  ) -> None:
        """
        Chunks the documents in self.raw_docs and stores the
        result into self.chunked_docs.
        
        Parameters
        ----------
        strategy : ['token', 'sentence', 'semantics']
            Strategy used to chunk the documents.
            
        size : int | None
            number of tokens per chunk (chunk_size). 
            Required if strategy == 'token'.
            default: None
            
        overlap : int | None
            number of tokens used in chunk overlap
            between consecutive chunks. Required if 
            strategy == 'token'.
            default: None
            
        Returns
        -------
        None
        """
        if self.raw_docs is None:
            raise ValueError("No raw documents found.")
    
        # Note: In principle, the following if-elif-else block could also be
        # encapsulated in a factory function. This would enhance consistency
        # with the rest of the code. However, the different chunkers are so 
        # wildly different in their initialization logic that the code would
        # become much harder to read and understand if that logic would be 
        # unified by using a factory function and a wrapper.
        if strategy.lower() == "token":
            from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTSplitter
            # Initialize text splitter with specified parameters
            if size is None or overlap is None:
                raise ValueError("When strategy = 'token', both size and overlap must be specified.")
            chunker = RCTSplitter(chunk_size=size,
                                  chunk_overlap=overlap,
                                  length_function=len,  # Function to compute the length of the text
                                  add_start_index=True,  # Flag to add start index to each chunk
                                  )
            
        elif strategy.lower() == "sentence":
            raise ValueError("Sentence-based chunking is currently not supported.")
            
        elif strategy.lower() == "semantics":
            from langchain_experimental.text_splitter import SemanticChunker
            chunker = SemanticChunker(self.embedder)
            
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
         
        # Specify vector store through factory function
        vector_store = factories.get_vector_store(db_type, 
                                                  self.embedder, 
                                                  path2db, 
                                                  load=False)
        vector_store.add_documents(self.chunked_docs)
        vector_store.persist()
        
        # Output message
        print(f"Saved {len(self.chunked_docs)} chunks to {path2db}.")
