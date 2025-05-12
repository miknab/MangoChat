# MangoChat

## Summary
MangoChat is a RAG system that can be used to extract information from minutes created at meetings from the "Los Manguitos", a Zurich-based dance team specializing in Cuban dances.

## Goal
The goal of this project is to come up with a highly modular, clearly structured code for a RAG model that can configured by the user and that can be evaluated with RAGAS. In principle, I want this code to allow a grid search over all supported hyperparameters with RAGAS evaluating each hyperparameter configuration such that the best performing RAG model can be constructed in a structured, data-driven way.

## Structure
At this point in time the package MangoRAG contains the following code files:
- mango_chat.py: contains the user interface that starts and ends MangoChat sessions
- mango_db.py: contains the code to create and persist the vector store
- mango_eval.py: contains the code to evaluate a RAG model with RAGAS
- mango_factories.py: contains a number of factory functions and adapter classes in order to organize conditional imports
- mango_rag.py: contains the actual RAG model code (i.e. query transformer, document retriever, reranker, and LLM call)

## Remarks
- The current code base is work in progress
- A few configurations are not supported yet
- Some configurations seem to be tricky to test with the available hardware at this time (example: using FAISS vector stores seems to blow up my RAM leading to Kernel crashes).
