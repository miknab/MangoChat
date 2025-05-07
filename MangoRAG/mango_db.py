from langchain.schema import Document

class MangoDB(object):
    """
    The class MangoDB creates the vector store containing the embedded
    chunks extracted from documents.
    
    MangoDB reads in documents stored in a specified location (path2docs),
    chunks them, embeds the chunks, and stores the resulting embeddings
    in a vector store.
    
    Parameters
    ----------
    path2docs : str
        path to the documents to be embedded
        
    path2db : str
        path to where the resulting vector store
        shall be written

    Methods
    -------
    self.read_docs()
    self.chunk_docs()
    self.create_db()
    """
    def __init__(self, path2docs: str, path2db: str):
        self.path2docs = path2docs
        self.path2db = path2db
        self.raw_docs: List[Document] = None
        self.chunked_docs: List[Document] = None
    
    def read_docs(self) -> None:
        """
        Reads in the documents found in path2docs and stores
        the resulting list of documents into the attribute
        self.raw_docs.
        
        Returns
        -------
        None
        """
        pass
    
    def chunk_docs(self, chunk_size: int, overlap: int) -> None:
        """
        Chunks the documents in self.raw_docs and stores the
        result into self.chunked_docs.
        
        Parameters
        ----------
        chunk_size : int
            number of tokens per chunk
            
        overlap : int
            number of tokens used in chunk overlap
            
        Returns
        -------
        None
        """
        if self.raw_docs is None:
            raise ValueError("No raw documents found.")
        pass
    
    def create_db(self, embedding_model) -> None:
        """
        Embeds the chunked documents using the specified embedding model
        and stores the chunk embeddings in a vector store.
        
        Parameters
        ----------
        embedding_model : Model
        """
        
        pass
    