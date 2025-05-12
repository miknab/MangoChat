# Stdlib imports
from typing import Tuple, List
import psutil, os

# Local imports
from MangoRAG import mango_db as db
from MangoRAG import mango_factories as factories

# 3rd party imports
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages.ai import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings

# REMARK: Further imports can be found embedded in the 
# code below. This is done in order to avoid importing
# packages that are not being used. The code below is
# highly configurable and it depends on the configurations
# which packages need to be important and which ones
# are not needed.
    
class MangoQueryTransformer(object):
    """
    Offers several query transformation techniques in order to modify
    query such that the quality of the resulting answer improves.
    
    Methods
    -------
    self.transform(user_query: str, method: str) -> str
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
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
        REWRITE_PROMPT_TEMPLATE = (
            "Schreibe die folgende Anfrage so um, dass sie spezifischer für die Dokumentensuche "
            "(document retrieval) wird. Gib lediglich die umgeschriebene Anfrage zurück und "
            "verzichte auf jegliche Zusatzinformationen:\n"
            "Anfrage: {query}\n"
            "Umgeschriebene Anfrage:"
        )

        rewrite_prompt = PromptTemplate(input_variables=["query"],
                                        template=REWRITE_PROMPT_TEMPLATE
                                       )
        
        # Compose the prompt and the LLM using the Runnable interface
        rewrite_chain = rewrite_prompt | self.llm
        
        # Call the chain using invoke instead of run
        rewritten_query = rewrite_chain.invoke({"query": query})
        
        if isinstance(rewritten_query, AIMessage):
            print("Umgeschriebene Anfrage:", rewritten_query.content)
            return rewritten_query.content
        elif isinstance(rewritten_query, str):
            print("Umgeschriebene Anfrage:", rewritten_query)
            return rewritten_query
        else:
            # The code should never make it here
            raise TypeError(f"rewritten_query should be of type Document or str, but is {type(rewritten_query)}.")
        
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
        DECOMPOSITION_PROMPT_TEMPLATE = (
            "Zerlege die nachfolgende, komplexe Frage in einfachere Teilfragen:\n"
            "Frage: {query}\n"
            "Teilfragen:"
            
        )    
        
        decompose_prompt = PromptTemplate(input_variables=["query"], 
                                          template=DECOMPOSITION_PROMPT_TEMPLATE
                                         )
        decomposer_chain = decompose_prompt | self.llm

        decomposed_text = decomposer_chain.invoke({"query": query})
        print("Teilfragen:", decomposed_text.content)
        #subquestions = [q.strip("- ").strip() for q in decomposed_text.split("\n") if q.strip()]
        return decomposed_text
        
    def _generate_pseudodoc(self, query: str, base_embedding_name: str) -> str:
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
        from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
        base_embedder = factories.get_embedding_model(base_embedding_name)
        hyde_embedder = HypotheticalDocumentEmbedder(llm=self.llm,
                                                     base_embedder=base_embedder
                                                    )
        
        hypo_doc = hyde_embedder.embed_query(query)
        return hypo_doc
        
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
        if method.lower() == "rewrite":
            trf_query = self._rewrite(user_query)
        elif method.lower() == "decompose":
            trf_query = self._decompose(user_query)
        elif method.lower() == "hyde":
            trf_query = self._generate_pseudodoc(user_query)
        else:
            raise ValuerError("Invalid query transformation method. Must be 'rewrite', 'decompose' or 'hyde'.")
            
        return trf_query
    
class MangoRetriever(object):
    """
    Retriever class to retrieve relevant chunks for a given query.
    """
    def __init__(self, vector_store: factories.VectorStoreAdapter):
        self.db = vector_store
        
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
    def __init__(self, 
                 path2db: str, 
                 db_type: str, 
                 embedding_model: str, 
                 top_k: int,
                 chat_model_name: str | None = None,
                 chat_model: BaseChatModel | None = None,
                 trf_method: str | None = None,  
                 rerank_model: str | None = None,
                 top_n: int | None = None,
                ):
        
        self.user_query = None
        self.trf_method = trf_method
        self.embedding = factories.get_embedding_model(embedding_model)
        self.db_type = db_type
        self.vector_store = factories.get_vector_store(db_type, self.embedding, path2db, load=True)
        self.top_k = top_k
        self.top_n = top_n if top_n else -1 
        self.rerank_model = rerank_model
        self.chat_model = chat_model if chat_model else factories.get_llm(chat_model_name)
        
        if self.top_n > self.top_k:
            print(self.top_k, self.top_n)
            raise ValueError("top_n (reranker) must be smaller or equal to top_k (retriever).")
    
    def answer_query(self, user_query: str):
        self.user_query = user_query  # keep a copy of the original query
        query = self.user_query
        
        # Query transformation
        if self.trf_method is not None:
            print("Query transformation:")
            print("---------------------")
            if self.trf_method.lower() in ['rewrite', 'decompose', 'hyde']:
                trafo = MangoQueryTransformer(llm=self.chat_model)
                query = trafo.transform(query, self.trf_method)
                print("DONE")
            else:
                raise ValueError("Invalid trf_method. Must be 'rewrite', 'decompose', or 'hyde'.")
        
        # Query embedding & chunk retrieval
        print("\nQuery embedding & chunk retrieval:")
        print("----------------------------------")
        if self.db_type.lower() in ['weaviate', 'faiss', 'chroma', 'qdrant', 'milvus']:
            print("\tCreating retriever...", end="")
            retriever = MangoRetriever(self.vector_store)
            print("done")
            print("\tretrieve...", end="")
            relevant_chunks, relevance_scores = retriever.retrieve_chunks(query, self.top_k)
            print("done")
        else:
            raise ValueError("Invalid db_type. Must be 'weaviate', 'faiss', 'chroma', 'qdrant', or 'milvus'.")
        print("DONE")
        
        # Chunk reranking
        if self.rerank_model is not None:
            print("\nChunk reranking:")
            print("----------------")
            if self.rerank_model.lower() in ['bge', 'ms-marco', 'colbert', 'jina']:
                reranker = MangoReranker(self.rerank_model, self.top_n)
                relevant_chunks, relevance_scores = reranker.rerank(self, relevant_chunks, query)
            else:
                raise ValueError("Invalid reranker model name. Must be 'bge', 'ms-marco', 'colbert', or 'jina'.")
            print("DONE")
        
        # Create context
        print("\nContext concatenation:")
        print("----------------------")
        context_text = "\n\n -- \n\n".join([doc.page_content for doc in relevant_chunks])
        print("DONE")
        
        # Answer generation
        print("\nAnswer generation:")
        print("------------------")
        PROMPT_TEMPLATE = (
            "Beantworte die Frage basierend auf dem nachfolgenden Kontext.\n"
            "Kontext: {context}\n"
            "Frage: {question}\n"
            "Falls du keine Antwort auf meine Frage im zur Verfügung gestellten "
            "Kontext finden kannst, antworte mit 'Ich kann die Frage leider nicht "
            "beantworten, da mir die nötigen Informationen dafür fehlen.'"
        )
        
        prompt_template = PromptTemplate(input_variables=["query"], 
                                         template=PROMPT_TEMPLATE
                                         )
        
        llm_chain = prompt_template | self.chat_model
        
        response = llm_chain.invoke({"context": context_text, "question": query})
        print("DONE")
        
        # Get sources of the matching documents
        print("\nSource consolidation:")
        print("---------------------")
        sources = [doc.metadata.get("source", None) for doc, score in zip(relevant_chunks, relevance_scores)]
        print("DONE")

        # Format and return response including generated text and sources
        formatted_response = f"Antwort: {response.content}\n\nQuellen: {sources}"
        print(formatted_response)
        
        return formatted_response, sources, response