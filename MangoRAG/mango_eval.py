# 3rd party imports
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# local imports
from MangoRAG import mango_rag

class RAGEvaluator(object):
    """
    Class facilitating RAG evaluation.
    
    Parameters
    ----------
    rag : mango_rag.MangoRAG
        The RAG system to be evaluated.
        
    Attributes
    ----------
    self.sample_queries : List[str]
        A list of manually defined, hard-coded queries on which
        self.rag is tested.
        
    self.expected_responses : List[str]
        Ground-truth answers to the queries defined in self.sample_queries.
    
    self.rag : mango_rag.MangoRAG
        The RAG system to be evaluated.
        
    Methods
    -------
    self.evaluate() -> dict[str, float]
    """
    def __init__(self, rag: mango_rag.MangoRAG):
        """
        Initializer function for RAGEvaluator class
        
        Parameters
        ----------
        rag : mango_rag.MangoRAG
            The RAG system to be evaluated.
        """
        self.sample_queries = [
            "Wer sind die Los Manguitos?",
            "Was ist der Vereinszweck der Los Manguitos?",
            "Wann wurde der Verein 'Los Manguitos' gegründet?",
            "Wie heissen die Aktivmitglieder der Los Manguitos am 20. September 2024?"
        ]

        self.expected_responses = [
            "Die Los Manguitos sind ein Tanzverein.",
            ("Zweck des Vereins ist es, den Paar- und Gruppentanz, insbesondere karibische und "
             "südamerikanische Tänze, zu pflegen, zu fördern und zu trainieren. Dabei ist die oberste  "
             "Priorität, dass die Vereinsmitglieder die Freude an diesem wertvollen Kulturgut aufrecht "
             "erhalten. Dazu werden gemeinsam Tanzfiguren und Choreographien erlernt bzw. erarbeitet. "
             "Der Gruppen- und Teamaspekt wird bei den Los Manguitos sehr grossgeschrieben. Der Verein "
             "ist politisch und konfessionell neutral."),
            "Am 26. März 2013.",
            ("Die Aktivmitglieder am 20. September 2024 heissen Alexandra, Iliana, Nicole, María, Julian, "
             "Bruno und Mischa.")
        ]
        
        self.rag = rag
        
    def evaluate(self) -> dict[str, float]:
        """
        Performs the actual evaluation of self.rag.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict[str, float]
            A dictionary with (metric_name, metric_value)-pairs
            for the thre metrics context recall, faithfulness and
            factual correctness.
        """
        # Run the RAG on the test examples
        dataset = []
        counter = 1
        for query, ground_truth in zip(self.sample_queries, self.expected_responses):
            print("")
            response, sources, _ = self.rag.answer_query(query)
            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": sources,
                    "response": response,
                    "reference": ground_truth
                }
            )
        
        # 
        evaluation_dataset = EvaluationDataset.from_list(dataset)
                  
        evaluator_llm = LangchainLLMWrapper(self.rag.chat_model)

        # Compute performance metrics
        metrics_list = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
        result = evaluate(dataset=evaluation_dataset,metrics=metrics_list,llm=evaluator_llm)
        return result, dataset