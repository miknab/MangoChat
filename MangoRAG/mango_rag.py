# Stdlib imports
from typing import Tuple

# Local imports
import mango_db as db

# 3rd party imports
from langchain.schema import Document

    
class MangoQueryTransformer(object):
    """
    Offers several query transformation techniques in order to modify
    query such that the quality of the resulting answer improves.
    
    Methods
    -------
    self.transform(user_query: str, method: str) -> str
    """
    def __init__(self):
        pass
    
    def _rewrite(self, query: str) -> str:
        """
        Query rewriting refines queries to better match relevant documents. 
        Prompt an LLM to rewrite queries to enhance performance.
        
        Parameters
        ----------
        query : str
            user query to be transformed
            
        Returns
        -------
        str
            rewritten query
        """
        pass
        
    def _decompose(self, query: str) -> str:
        """
        Retrieves documents based on sub-questions derived from the original
        query, which is more complex to comprehend and handle.
        
        Parameters
        ----------
        query : str
            user query to be transformed
            
        Returns
        -------
        str
            decomposed query
        """
        pass
        
    def _generate_pseudodoc(self, query: str) -> str:
        """
        Follows HyDE (hypothetical document extraction), i.e. a hypothetical
        answer is hallucinated and relevant chunks are retrieved based on
        the similarity to the hypothetical answer.
        
        Parameters
        ----------
        query : str
            user query to be transformed
            
        Returns
        -------
        str
            hypothetical document
        """
        pass
        
    def transform(self, user_query: str, method: str) -> str:
        """
        Transforms user_query according to method.
        
        Parameters
        ----------
        user_query : str
            Query entered by user
            
        method : str
            Method used to transform user_query
            
        Returns
        -------
        trf_query : str
            Transformed query
        """
        if method.lower() = "rewrite":
            trf_query = self._rewrite(user_query)
        elif method.lower() = "decompose":
            trf_query = self._decompose(user_query)
        elif self.method.lower() = "hyde":
            trf_query = self._generate_pseudodoc(user_query)
        else:
            raise ValuerError("Invalid query transformation method. Must be 'rewrite', 'decompose' or 'hyde'.")
            
        return trf_query
    
class MangoRetriever(object):
    """
    Retriever class to retrieve relevant chunks for a given query.
    """
    def __init__(self, path2db: str, embedding_model: str, db_type: str):
        # Prepare the database
        if db_type.lower() == "faiss":
            from langchain_community.vectorstores import FAISS
            self.db = FAISS.load_local(path2db, embeddings=embedding_model)
            
        elif db_type.lower() == "weaviate":
            raise ValueError("Weaviate is currently not supported.")
            
        elif db_type.lower() == "chroma":
            from langchain.vectorstores.chroma import Chroma
            self.db = Chroma(persist_directory=path2db, embedding_function=embedding_model)
            
        elif db_type.lower() == "qdrant":
            raise ValueError("Qdrant is currently not supported.")
            
        elif db_type.lower() == "milvus":
            raise ValueError("Milvus is currently not supported.")
            
        else:
            raise ValueError("Invalid database type. Must be 'faiss', "
                             + "'weaviate', 'chroma', 'qdrant', or 'milvus'."
                            )
        
    def retrieve_chunks(self, user_query: str, top_k: int) -> Tuple[List[Document], List[float]]:
        """
        Retrieves chunks from vector store based on parsed user query.
        
        Parameters
        ----------
        user_query : str
            The query passed to the RAG by the user.
            
        top_k : int
            Size of the retrieved chunk set.
            
        Returns
        -------
        results
            Set of retrieved chunks
        """
        # Retrieving the context from the DB using similarity search
        retrieved_chunks_with_scores = self.db.similarity_search_with_relevance_scores(user_query, k=top_k)
        [retrieved_chunks, scores] = list(zip(*retrieved_chunks_with_scores))
        return retrieved_chunks, scores
    
class MangoReranker(object):
    """
    Reranks retrieved chunks in order to improve answer quality.
    """
    def __init__(self, rerank_model: str, top_k):
        self.rerank_model = rerank_model
        self.top_k = top_k
         
    def rerank(self, docs: List[Document], user_query: str) -> Tuple[List[Document], List[float]]:
        from langchain.retrievers.document_compressors import RerankersRerank
        from rerankers.models import load_model
        
        if self.rerank_model.lower() in []
        # Load reranker model from rerankers package
        if self.rerank_model.lower() == "bge":
            model = load_model("BAAI/bge-reranker-base")
        elif self.rerank_model.lower() == "ms-marco":
            model = load_model("cross-encoder/ms-marco-MiniLM-L-6-v2")
        elif self.rerank_model.lower() == "colbert":
            model = load_model("naver/colbertv2.0")
        elif self.rerank_model.lower() == "jina":
            model = load_model("jina-reranker")
        else:
            raise ValueError("Invalid reranker model name. Must be 'bge', 'ms-marco', 'colbert', or 'jina').")
        
        # Wrap in LangChain-compatible reranker
        reranker = RerankersRerank(model=model, top_n=self.top_k)

        # Compress documents (rerank) manually
        reranked_docs = reranker.compress_documents(docs, query=user_query)
        
        [reranked_chunks, reranking_scores] = list(zip(*retrieved_chunks_with_scores))
        
        return reranked_chunks, reranking_scores

    
class MangoRAG(object):
    """
    Bundles the different classes above as well as the LLM into a single
    class.
    
    This class takes in a user query and goes through the following steps:
    1. transform the query
    2. embed the transformed query and retrieve relevant chunks
    3. rerank retrieved chunks
    4. use LLM to generate an answer based on reranked chunks
    
    Parameters
    ----------
    trf_method : ['rewrite', 'decompose', 'hyde']
        Method used to transform user_query
    
    path2db : str
        path from where the resulting vector store
        shall be loaded.
    
    db_type : ['weaviate', 'faiss', 'chroma', 'qdrant', 'milvus']
        Type of database used to create the vector store
    
    embedding_model : ['spacy', 'openai', 'nomic']
        Model used to embed the chunks. YOU MUST USE THE
        SAME EMBEDDING MODEL AS IN THE VECTOR STORE GENERATION!
    
    top_k : int
        Size of the retrieved chunk set. High top_k leads to better recall and slower reranking
        while low top_k is faster, but might miss relevant docs.
        
    top_n : int
        Size of the chunk set after reranking. This parameter governs the amount of context for
        the LLM during answer generation (higher top_n = more context).
         
    rerank_model : ['bge', 'ms-marco', 'colbert', 'jina']
        Model used for chunk reranking.
        
    Methods
    -------
    self.answer_query(user_query: str)
    """
    def __init__(self, trf_method: str, path2db: str, db_type: str, embedding_model: str, top_k: int, top_n: int, rerank_model: str):
        self.user_query = None
        self.trf_method = trf_method
        self.path2db = path2db  
        self.db_type = db_type
        self.embedding_model = embedding_model  # TODO: make sure it is enforced that this parameter
                                                # has the same value as during vector store generation
        self.top_k = top_k
        self.top_n = top_n
        self.rerank_model = rerank_model
        
        if self.top_k > self.top_n:
            raise ValueError("top_k (reranker) must be smaller or equal to top_n (retriever).")
    
    def answer_query(self, user_query: str):
        # Query transformation
        if self.trf_method.lower() in ['rewrite', 'decompose', 'hyde']:
            trafo = MangoQueryTransformer()
            trf_query = trafo.transform(user_query, self.trf_method)
        else:
            raise ValueError("Invalid trf_method. Must be 'rewrite', 'decompose', or 'hyde'.")
        
        # Query embedding & chunk retrieval
        if self.db_type.lower() in ['weaviate', 'faiss', 'chroma', 'qdrant', 'milvus']:
            retriever = MangoRetriever(self.path2db, self.embedding_model, self.db_type)
            top_k_chunks, similarity_scores = retriever.retrieve_chunks(user_query, self.top_k)
        else:
            raise ValueError("Invalid db_type. Must be 'weaviate', 'faiss', 'chroma', 'qdrant', or 'milvus'.")
            
        # Chunk reranking
        if self.rerank_model.lower() in ['bge', 'ms-marco', 'colbert', 'jina']:
            reranker = MangoReranker(self.rerank_model, self.top_n)
            reranked_chunks, reranking_scores = reranker.rerank(self, top_k_chunks, user_query: str)
        else:
            raise ValueError("Invalid reranker model name. Must be 'bge', 'ms-marco', 'colbert', or 'jina'.")
        
        # Create context
        context_text = "\n\n -- \n\n".join([doc.page_content for doc in reranked_chunks])
 
        # Answer generation
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=user_query)

        if self.chat_model_name.lower() == "openai":
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI()
        elif self.chat_model_name.lower() == "ollama":
            from langchain.chat_models import ChatOllama
            llm = ChatOllama()
        else:
            raise ValueError("Invalid chat model name. Must be 'openai'.")
        
        response_text = llm.predict(prompt)

        # Get sources of the matching documents
        sources = [doc.metadata.get("source", None) for doc, reranking_score in zip(reranked_chunks, reranking_scores)]

        # Format and return response including generated text and sources
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        
        return formatted_response, response_text