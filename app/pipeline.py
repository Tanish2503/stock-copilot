"""
Real-time Pathway pipeline for indexing JSON headline files and providing LangChain retrieval.

This module creates a Pathway pipeline that:
1. Monitors a directory for JSON files containing headlines
2. Indexes the content using Pathway's vector store
3. Provides a LangChain-compatible retriever interface

Dependencies:
    pip install pathway langchain sentence-transformers
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

import pathway as pw
from pathway.stdlib.ml.index import KNNIndex
from pathway.xpacks.llm import embedders, parsers
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class PathwayHeadlinesRetriever(BaseRetriever):
    """LangChain-compatible retriever that uses Pathway's vector index."""
    
    def __init__(self, pathway_index: KNNIndex, k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            pathway_index: Pathway KNN index instance
            k: Number of documents to retrieve
        """
        super().__init__()
        self.pathway_index = pathway_index
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: Search query string
            run_manager: LangChain callback manager
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Query the Pathway index
            results = self.pathway_index.get_nearest_items(
                query=query,
                k=self.k,
                metadata_filter=None
            )
            
            documents = []
            for result in results:
                # Extract content and metadata from Pathway result
                content = result.get('text', '')
                metadata = {
                    'source': result.get('source', ''),
                    'timestamp': result.get('timestamp', ''),
                    'score': result.get('dist', 0.0),
                    **result.get('metadata', {})
                }
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


class HeadlinesPipeline:
    """Real-time pipeline for processing headline JSON files with Pathway."""
    
    def __init__(
        self, 
        input_dir: str,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dimension: int = 384
    ):
        """
        Initialize the headlines pipeline.
        
        Args:
            input_dir: Directory to monitor for JSON files
            embedder_model: Sentence transformer model for embeddings
            index_dimension: Dimension of the embedding vectors
        """
        self.input_dir = Path(input_dir)
        self.embedder_model = embedder_model
        self.index_dimension = index_dimension
        self.index = None
        self.retriever = None
        
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_json_file(self, contents: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse JSON file contents and extract headlines.
        
        Args:
            contents: File contents as string
            metadata: File metadata from Pathway
            
        Returns:
            List of parsed headline records
        """
        try:
            data = json.loads(contents)
            records = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of headline objects
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        records.append({
                            'text': self._extract_text_from_item(item),
                            'source': metadata.get('path', ''),
                            'timestamp': metadata.get('modified_at', ''),
                            'item_index': i,
                            'metadata': item
                        })
            elif isinstance(data, dict):
                # Single headline object or container with headlines
                if 'headlines' in data:
                    # Container format: {"headlines": [...]}
                    for i, item in enumerate(data['headlines']):
                        records.append({
                            'text': self._extract_text_from_item(item),
                            'source': metadata.get('path', ''),
                            'timestamp': metadata.get('modified_at', ''),
                            'item_index': i,
                            'metadata': item
                        })
                else:
                    # Single headline object
                    records.append({
                        'text': self._extract_text_from_item(data),
                        'source': metadata.get('path', ''),
                        'timestamp': metadata.get('modified_at', ''),
                        'item_index': 0,
                        'metadata': data
                    })
            
            return records
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {metadata.get('path', 'unknown')}: {e}")
            return []
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from a headline item.
        
        Args:
            item: Dictionary containing headline data
            
        Returns:
            Extracted text content
        """
        # Common field names for headline text
        text_fields = ['headline', 'title', 'text', 'content', 'summary', 'description']
        
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                return item[field]
        
        # If no standard field found, combine available text fields
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts) if text_parts else str(item)
    
    def setup_pipeline(self) -> KNNIndex:
        """
        Set up the Pathway pipeline for real-time processing.
        
        Returns:
            Configured KNN index
        """
        # Create file system connector for JSON files
        files = pw.io.fs.read(
            path=str(self.input_dir),
            format="binary",
            with_metadata=True,
            autocommit_duration_ms=1000,  # Check for changes every second
        )
        
        # Filter for JSON files only
        json_files = files.filter(pw.this.path.str.endswith('.json'))
        
        # Parse file contents
        def parse_file_contents(row):
            """Parse individual file row."""
            try:
                contents = row.data.decode('utf-8')
                metadata = {
                    'path': row.path,
                    'modified_at': row.modified_at,
                    'owner': getattr(row, 'owner', ''),
                    'size': getattr(row, 'size', 0)
                }
                
                records = self._parse_json_file(contents, metadata)
                return records
                
            except Exception as e:
                print(f"Error processing file {row.path}: {e}")
                return []
        
        # Apply parsing and flatten results
        parsed_data = json_files.select(
            records=pw.apply(parse_file_contents, pw.this)
        ).flatten(pw.this.records)
        
        # Extract fields for indexing
        documents = parsed_data.select(
            text=pw.this.text,
            source=pw.this.source,
            timestamp=pw.this.timestamp,
            metadata=pw.this.metadata
        )
        
        # Create embeddings
        embedder = embedders.SentenceTransformerEmbedder(
            model=self.embedder_model
        )
        
        embedded_docs = embedder(documents, text_column="text")
        
        # Create KNN index
        self.index = KNNIndex(
            embedded_docs,
            dimensions=self.index_dimension,
            metadata_column="metadata"
        )
        
        return self.index
    
    def get_retriever(self, k: int = 5) -> PathwayHeadlinesRetriever:
        """
        Get a LangChain-compatible retriever.
        
        Args:
            k: Number of documents to retrieve per query
            
        Returns:
            PathwayHeadlinesRetriever instance
        """
        if self.index is None:
            raise ValueError("Pipeline not set up. Call setup_pipeline() first.")
        
        if self.retriever is None:
            self.retriever = PathwayHeadlinesRetriever(self.index, k=k)
        
        return self.retriever
    
    def run_server(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Run the Pathway server to start real-time processing.
        
        Args:
            host: Server host address
            port: Server port number
        """
        if self.index is None:
            raise ValueError("Pipeline not set up. Call setup_pipeline() first.")
        
        print(f"Starting Pathway server on {host}:{port}")
        print(f"Monitoring directory: {self.input_dir}")
        
        # Run the Pathway computation
        pw.run(
            host=host,
            port=port,
            with_cache=True,
            cache_backend=pw.persistence.Backend.filesystem("./pathway_cache")
        )


# Example usage and utility functions
def create_sample_headlines(output_dir: str, num_files: int = 3):
    """
    Create sample headline JSON files for testing.
    
    Args:
        output_dir: Directory to create sample files
        num_files: Number of sample files to create
    """
    import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sample_headlines = [
        {
            "headlines": [
                {"headline": "AI Revolution Transforms Healthcare Industry", "category": "technology", "priority": "high"},
                {"headline": "Climate Summit Reaches Historic Agreement", "category": "environment", "priority": "medium"},
                {"headline": "New Space Mission Discovers Water on Mars", "category": "science", "priority": "high"}
            ]
        },
        [
            {"title": "Economic Growth Surpasses Expectations", "sector": "finance", "impact": "positive"},
            {"title": "Breakthrough in Quantum Computing Achieved", "sector": "technology", "impact": "revolutionary"},
            {"title": "Global Vaccination Campaign Shows Progress", "sector": "health", "impact": "positive"}
        ],
        {
            "headline": "Major Cybersecurity Threat Neutralized",
            "description": "International cooperation leads to successful takedown of criminal network",
            "category": "security",
            "timestamp": datetime.datetime.now().isoformat()
        }
    ]
    
    for i in range(num_files):
        filename = f"headlines_{i+1}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_headlines[i % len(sample_headlines)], f, indent=2)
        
        print(f"Created sample file: {filepath}")


# Main execution example
if __name__ == "__main__":
    # Example usage
    input_directory = "./data/headlines"
    
    # Create sample data
    create_sample_headlines(input_directory)
    
    # Initialize pipeline
    pipeline = HeadlinesPipeline(input_directory)
    
    # Set up the pipeline
    index = pipeline.setup_pipeline()
    
    # Get retriever for LangChain
    retriever = pipeline.get_retriever(k=3)
    
    print("Pipeline initialized successfully!")
    print("You can now use the retriever with LangChain:")
    print("  docs = retriever.get_relevant_documents('AI technology news')")
    print("\nTo start real-time processing, call:")
    print("  pipeline.run_server()")