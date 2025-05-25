"""
LangChain RetrievalQA Engine Module

This module creates a LangChain RetrievalQA chain using OpenAI GPT-4 and a provided
retriever from a vector store. It provides a simple interface to build question-answering
systems over document collections.

Dependencies:
    pip install langchain openai
"""
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file variables into the environment

openai_api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")  # optional, can be None if not set

print("OpenAI API Key:", openai_api_key)
print("News API Key:", news_api_key)

import os
from typing import Optional, Dict, Any, List
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import Document


def build_query_engine(
    retriever: BaseRetriever,
    model_name: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    custom_prompt: Optional[PromptTemplate] = None,
    chain_type: str = "stuff",
    return_source_documents: bool = True,
    openai_api_key: Optional[str] = None
) -> RetrievalQA:
    """
    Build a RetrievalQA chain using OpenAI GPT-4 and a provided retriever.
    
    Args:
        retriever: LangChain BaseRetriever instance from a vector store
        model_name: OpenAI model name (default: "gpt-4")
        temperature: Sampling temperature for the model (0.0 to 1.0)
        max_tokens: Maximum tokens in the response
        custom_prompt: Custom prompt template for the QA chain
        chain_type: Type of chain to use ("stuff", "map_reduce", "refine", "map_rerank")
        return_source_documents: Whether to return source documents with answers
        openai_api_key: OpenAI API key (if not set as environment variable)
        
    Returns:
        Configured RetrievalQA chain ready for querying
        
    Raises:
        ValueError: If OpenAI API key is not provided or found
        
    Example:
        >>> retriever = vector_store.as_retriever()
        >>> qa_chain = build_query_engine(retriever)
        >>> result = qa_chain({"query": "What are the latest tech trends?"})
    """
    # Set OpenAI API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please provide it via the openai_api_key parameter "
            "or set the OPENAI_API_KEY environment variable."
        )
    
    # Initialize the language model
    llm_kwargs = {
        "model_name": model_name,
        "temperature": temperature,
    }
    
    if max_tokens:
        llm_kwargs["max_tokens"] = max_tokens
    
    # Use ChatOpenAI for GPT-4 models
    if model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5-turbo"):
        llm = ChatOpenAI(**llm_kwargs)
    else:
        llm = OpenAI(**llm_kwargs)
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=return_source_documents,
        chain_type_kwargs={
            "prompt": custom_prompt
        } if custom_prompt else {}
    )
    
    return qa_chain


class QAEngineBuilder:
    """
    Builder class for creating and configuring RetrievalQA chains with various options.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the QA Engine Builder.
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key
        self.model_name = "gpt-4"
        self.temperature = 0.0
        self.max_tokens = None
        self.chain_type = "stuff"
        self.return_source_documents = True
        self.custom_prompt = None
    
    def with_model(self, model_name: str) -> "QAEngineBuilder":
        """Set the OpenAI model to use."""
        self.model_name = model_name
        return self
    
    def with_temperature(self, temperature: float) -> "QAEngineBuilder":
        """Set the sampling temperature."""
        self.temperature = temperature
        return self
    
    def with_max_tokens(self, max_tokens: int) -> "QAEngineBuilder":
        """Set the maximum tokens for responses."""
        self.max_tokens = max_tokens
        return self
    
    def with_chain_type(self, chain_type: str) -> "QAEngineBuilder":
        """Set the chain type (stuff, map_reduce, refine, map_rerank)."""
        self.chain_type = chain_type
        return self
    
    def with_source_documents(self, return_sources: bool) -> "QAEngineBuilder":
        """Set whether to return source documents."""
        self.return_source_documents = return_sources
        return self
    
    def with_custom_prompt(self, prompt: PromptTemplate) -> "QAEngineBuilder":
        """Set a custom prompt template."""
        self.custom_prompt = prompt
        return self
    
    def build(self, retriever: BaseRetriever) -> RetrievalQA:
        """
        Build the RetrievalQA chain with the configured options.
        
        Args:
            retriever: LangChain BaseRetriever instance
            
        Returns:
            Configured RetrievalQA chain
        """
        return build_query_engine(
            retriever=retriever,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            custom_prompt=self.custom_prompt,
            chain_type=self.chain_type,
            return_source_documents=self.return_source_documents,
            openai_api_key=self.openai_api_key
        )


# Predefined prompt templates for different use cases
FINANCIAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a financial analyst assistant. Use the following financial news and data to answer the question.
If you don't know the answer based on the provided context, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear and concise answer based on the financial information above. Include relevant ticker symbols, dates, and key metrics when applicable."""
)

GENERAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
)

DETAILED_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert assistant. Use the following context to provide a comprehensive answer to the question.
Include relevant details and explain your reasoning when possible.

Context:
{context}

Question: {question}

Detailed Answer:"""
)


def create_financial_qa_engine(
    retriever: BaseRetriever,
    model_name: str = "gpt-4",
    temperature: float = 0.1
) -> RetrievalQA:
    """
    Create a specialized QA engine for financial queries.
    
    Args:
        retriever: Retriever for financial documents
        model_name: OpenAI model to use
        temperature: Sampling temperature
        
    Returns:
        RetrievalQA chain optimized for financial queries
    """
    return build_query_engine(
        retriever=retriever,
        model_name=model_name,
        temperature=temperature,
        custom_prompt=FINANCIAL_QA_PROMPT,
        return_source_documents=True
    )


def create_conversational_qa_engine(
    retriever: BaseRetriever,
    model_name: str = "gpt-4",
    temperature: float = 0.3
) -> RetrievalQA:
    """
    Create a QA engine optimized for conversational interactions.
    
    Args:
        retriever: Document retriever
        model_name: OpenAI model to use
        temperature: Higher temperature for more creative responses
        
    Returns:
        RetrievalQA chain optimized for conversations
    """
    return build_query_engine(
        retriever=retriever,
        model_name=model_name,
        temperature=temperature,
        custom_prompt=DETAILED_QA_PROMPT,
        return_source_documents=True
    )


class QAEngineWrapper:
    """
    Wrapper class that provides additional functionality around a RetrievalQA chain.
    """
    
    def __init__(self, qa_chain: RetrievalQA):
        """
        Initialize the wrapper.
        
        Args:
            qa_chain: The RetrievalQA chain to wrap
        """
        self.qa_chain = qa_chain
        self.query_history = []
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Query the QA chain and store the interaction.
        
        Args:
            question: The question to ask
            **kwargs: Additional arguments for the chain
            
        Returns:
            Dictionary containing the answer and metadata
        """
        result = self.qa_chain({"query": question}, **kwargs)
        
        # Store in history
        self.query_history.append({
            "question": question,
            "answer": result.get("result", ""),
            "sources": result.get("source_documents", []),
            "timestamp": __import__("datetime").datetime.now().isoformat()
        })
        
        return result
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the query history."""
        return self.query_history.copy()
    
    def clear_history(self):
        """Clear the query history."""
        self.query_history.clear()
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of results for each question
        """
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        return results


# Utility functions
def validate_retriever(retriever: BaseRetriever) -> bool:
    """
    Validate that the provided retriever is properly configured.
    
    Args:
        retriever: The retriever to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Test the retriever with a simple query
        test_docs = retriever.get_relevant_documents("test query")
        return isinstance(test_docs, list)
    except Exception as e:
        print(f"Retriever validation failed: {e}")
        return False


def get_available_models() -> List[str]:
    """
    Get list of available OpenAI models for the QA engine.
    
    Returns:
        List of model names
    """
    return [
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "text-davinci-003",
        "text-davinci-002"
    ]


def get_qa_response(question: str, stock_symbol: str) -> str:
    """
    Get a response for a question about a specific stock.
    
    Args:
        question: The question to ask
        stock_symbol: The stock symbol to query about
        
    Returns:
        String response to the question
    """
    try:
        # Create a mock retriever for now - in a real app, this would use actual data
        class StockRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                # For now, return a simple document with stock info
                return [Document(
                    page_content=f"Information about {stock_symbol}: This is a placeholder response. In a real implementation, this would contain actual stock data and news about {stock_symbol}.",
                    metadata={"source": "stock_data", "ticker": stock_symbol}
                )]
        
        # Create retriever and QA engine
        retriever = StockRetriever()
        qa_engine = create_financial_qa_engine(retriever)
        
        # Get response
        result = qa_engine({"query": question})
        return result.get("result", "Sorry, I couldn't generate a response at this time.")
        
    except Exception as e:
        return f"Error processing question: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Example with a mock retriever (replace with your actual retriever)
    class MockRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return [
                Document(
                    page_content=f"Sample document content related to: {query}",
                    metadata={"source": "mock", "score": 0.9}
                )
            ]
    
    # Create a mock retriever for demonstration
    mock_retriever = MockRetriever()
    
    # Build QA engine using the main function
    print("Building QA engine...")
    try:
        # Note: This will fail without a valid OpenAI API key
        # qa_engine = build_query_engine(mock_retriever)
        print("QA engine would be built successfully with valid API key")
    except ValueError as e:
        print(f"Expected error (no API key): {e}")
    
    # Demonstrate builder pattern
    print("\nUsing builder pattern...")
    builder = QAEngineBuilder()
    configured_builder = (builder
                         .with_model("gpt-4")
                         .with_temperature(0.2)
                         .with_max_tokens(500)
                         .with_custom_prompt(FINANCIAL_QA_PROMPT))
    
    print("Builder configured successfully")
    
    # Show available models
    print(f"\nAvailable models: {get_available_models()}")