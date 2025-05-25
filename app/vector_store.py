"""
Pathway TextVectorStore module for headline data indexing.

This module processes Pathway tables containing headline, ticker, and timestamp fields
and creates a vector store using Pathway's TextVectorStore class for efficient
similarity search and retrieval.

Dependencies:
    pip install pathway sentence-transformers
"""

from typing import Optional, Dict, Any, Union, List
import pathway as pw
from pathway.xpacks.llm.vector_store import TextVectorStore
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder


class HeadlineSchema(pw.Schema):
    """Schema for headline data tables."""
    headline: str
    ticker: str
    timestamp: str


def create_vector_index(
    table: pw.Table,
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    dimensions: int = 384,
    metadata_columns: Optional[List[str]] = None
) -> TextVectorStore:
    """
    Create a TextVectorStore from a Pathway table with headline, ticker, and timestamp fields.
    
    Args:
        table: Pathway table with headline, ticker, and timestamp fields
        embedder_model: Name of the sentence transformer model to use
        dimensions: Dimension of the embedding vectors
        metadata_columns: Additional columns to include as metadata
        
    Returns:
        TextVectorStore instance ready for similarity search
        
    Raises:
        ValueError: If required columns are missing from the table
        
    Example:
        >>> import pathway as pw
        >>> data = [
        ...     {"headline": "Apple stock surges on earnings", "ticker": "AAPL", "timestamp": "2024-01-15T10:30:00"},
        ...     {"headline": "Tech rally continues", "ticker": "NVDA", "timestamp": "2024-01-15T11:00:00"}
        ... ]
        >>> table = pw.debug.table_from_rows(schema=HeadlineSchema, rows=data)
        >>> vector_store = create_vector_index(table)
    """
    # Validate required columns
    required_columns = {'headline', 'ticker', 'timestamp'}
    table_columns = set(table.column_names())
    
    if not required_columns.issubset(table_columns):
        missing = required_columns - table_columns
        raise ValueError(f"Table missing required columns: {missing}")
    
    # Set default metadata columns if not specified
    if metadata_columns is None:
        metadata_columns = ['ticker', 'timestamp']
    
    # Create embedder
    embedder = SentenceTransformerEmbedder(
        model=embedder_model,
        call_kwargs={"show_progress_bar": False}
    )
    
    # Create TextVectorStore
    vector_store = TextVectorStore(
        data=table,
        text_column='headline',  # Use headline as the text to embed
        embedder=embedder,
        metadata_columns=metadata_columns,
        dimensions=dimensions
    )
    
    return vector_store


class HeadlineVectorStoreManager:
    """
    Manager class for creating and working with headline vector stores.
    Provides additional utilities and configuration options.
    """
    
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int = 384
    ):
        """
        Initialize the HeadlineVectorStoreManager.
        
        Args:
            embedder_model: Default sentence transformer model
            dimensions: Default embedding dimensions
        """
        self.embedder_model = embedder_model
        self.dimensions = dimensions
        self.embedder = None
    
    def create_vector_store(
        self,
        table: pw.Table,
        metadata_columns: Optional[List[str]] = None,
        custom_embedder: Optional[SentenceTransformerEmbedder] = None
    ) -> TextVectorStore:
        """
        Create a vector store with the manager's configuration.
        
        Args:
            table: Input table with headline data
            metadata_columns: Columns to include as metadata
            custom_embedder: Optional custom embedder instance
            
        Returns:
            Configured TextVectorStore
        """
        if custom_embedder is None:
            if self.embedder is None:
                self.embedder = SentenceTransformerEmbedder(
                    model=self.embedder_model,
                    call_kwargs={"show_progress_bar": False}
                )
            embedder = self.embedder
        else:
            embedder = custom_embedder
        
        if metadata_columns is None:
            metadata_columns = ['ticker', 'timestamp']
        
        return TextVectorStore(
            data=table,
            text_column='headline',
            embedder=embedder,
            metadata_columns=metadata_columns,
            dimensions=self.dimensions
        )
    
    def create_filtered_vector_store(
        self,
        table: pw.Table,
        ticker_filter: Optional[Union[str, List[str]]] = None,
        timestamp_range: Optional[tuple] = None,
        metadata_columns: Optional[List[str]] = None
    ) -> TextVectorStore:
        """
        Create a vector store with filtering applied.
        
        Args:
            table: Input table with headline data
            ticker_filter: Single ticker or list of tickers to include
            timestamp_range: Tuple of (start, end) timestamps for filtering
            metadata_columns: Columns to include as metadata
            
        Returns:
            Filtered TextVectorStore
        """
        filtered_table = table
        
        # Apply ticker filter
        if ticker_filter is not None:
            if isinstance(ticker_filter, str):
                filtered_table = filtered_table.filter(pw.this.ticker == ticker_filter)
            elif isinstance(ticker_filter, list):
                filtered_table = filtered_table.filter(pw.this.ticker.is_in(ticker_filter))
        
        # Apply timestamp range filter
        if timestamp_range is not None:
            start_time, end_time = timestamp_range
            filtered_table = filtered_table.filter(
                (pw.this.timestamp >= start_time) & (pw.this.timestamp <= end_time)
            )
        
        return self.create_vector_store(filtered_table, metadata_columns)


def create_sample_headline_table() -> pw.Table:
    """
    Create a sample headline table for testing purposes.
    
    Returns:
        Pathway table with sample headline data
    """
    sample_data = [
        {
            "headline": "Apple reports record quarterly earnings, stock price jumps 5% in after-hours trading",
            "ticker": "AAPL",
            "timestamp": "2024-01-15T16:30:00Z"
        },
        {
            "headline": "NVIDIA unveils next-generation AI chip architecture with 50% performance improvement",
            "ticker": "NVDA", 
            "timestamp": "2024-01-15T14:15:00Z"
        },
        {
            "headline": "Tesla announces significant price cuts across all vehicle models to boost demand",
            "ticker": "TSLA",
            "timestamp": "2024-01-15T11:00:00Z"
        },
        {
            "headline": "Microsoft Azure cloud revenue surges 30% year-over-year, beating analyst expectations",
            "ticker": "MSFT",
            "timestamp": "2024-01-15T15:45:00Z"
        },
        {
            "headline": "Amazon Web Services launches new data centers in three major markets",
            "ticker": "AMZN",
            "timestamp": "2024-01-15T13:20:00Z"
        },
        {
            "headline": "Google parent Alphabet achieves quantum computing breakthrough with new processor",
            "ticker": "GOOGL",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        {
            "headline": "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
            "ticker": "SPY",
            "timestamp": "2024-01-15T14:00:00Z"
        },
        {
            "headline": "Oil prices surge 3% on geopolitical tensions, energy stocks rally",
            "ticker": "XOM",
            "timestamp": "2024-01-15T09:45:00Z"
        },
        {
            "headline": "Cryptocurrency market rebounds with Bitcoin crossing $45,000 threshold",
            "ticker": "BTC",
            "timestamp": "2024-01-15T12:15:00Z"
        },
        {
            "headline": "Pharmaceutical giant announces breakthrough cancer treatment in Phase III trials",
            "ticker": "PFE",
            "timestamp": "2024-01-15T11:30:00Z"
        }
    ]
    
    return pw.debug.table_from_rows(
        schema=HeadlineSchema,
        rows=sample_data
    )


def search_similar_headlines(
    vector_store: TextVectorStore,
    query: str,
    k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> pw.Table:
    """
    Search for similar headlines in the vector store.
    
    Args:
        vector_store: The TextVectorStore instance
        query: Search query string
        k: Number of similar headlines to return
        metadata_filter: Optional filter for metadata fields
        
    Returns:
        Table with similar headlines and their scores
    """
    # Use the vector store's similarity search functionality
    results = vector_store.similarity_search(
        query=query,
        k=k,
        metadata_filter=metadata_filter
    )
    
    return results


def create_tech_headlines_vector_store(table: pw.Table) -> TextVectorStore:
    """
    Create a specialized vector store for technology-related headlines.
    
    Args:
        table: Table with headline data
        
    Returns:
        TextVectorStore optimized for tech headlines
    """
    # Filter for tech-related tickers
    tech_tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META"]
    tech_table = table.filter(pw.this.ticker.is_in(tech_tickers))
    
    # Use a model that might be better for technical content
    return create_vector_index(
        tech_table,
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        metadata_columns=['ticker', 'timestamp', 'headline']
    )


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    print("Creating sample headline table...")
    sample_table = create_sample_headline_table()
    print(f"Created table with {len(sample_table)} headlines")
    
    # Create basic vector index
    print("\nCreating vector index...")
    vector_store = create_vector_index(sample_table)
    print("Vector store created successfully!")
    
    # Create filtered vector store using manager
    print("\nCreating filtered vector store...")
    manager = HeadlineVectorStoreManager()
    filtered_store = manager.create_filtered_vector_store(
        sample_table,
        ticker_filter=["AAPL", "NVDA", "TSLA"],
        timestamp_range=("2024-01-15T10:00:00Z", "2024-01-15T16:00:00Z")
    )
    print("Filtered vector store created successfully!")
    
    # Create tech-focused vector store
    print("\nCreating tech headlines vector store...")
    tech_store = create_tech_headlines_vector_store(sample_table)
    print("Tech vector store created successfully!")
    
    print("\nAll vector stores are ready for similarity search operations.")
    print("Example usage:")
    print("  results = search_similar_headlines(vector_store, 'Apple earnings report', k=3)")