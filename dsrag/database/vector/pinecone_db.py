from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult, MetadataFilter
from typing import Optional, List
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import os


def format_metadata_filter(metadata_filter: MetadataFilter) -> dict:
    """
    Format the metadata filter to be used in the Pinecone query method.

    Args:
        metadata_filter (MetadataFilter): The metadata filter.

    Returns:
        dict: The formatted metadata filter.
    """
    field = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]

    operator_mapping = {
        "equals": "==",
        "not_equals": "!=",
        "in": "in",
        "not_in": "not_in",
        "greater_than": ">",
        "less_than": "<",
        "greater_than_equals": ">=",
        "less_than_equals": "<=",
    }

    formatted_operator = operator_mapping[operator]
    formatted_metadata_filter = {field: {formatted_operator: value}}

    return formatted_metadata_filter


class PineconeDB(VectorDB):

    def __init__(
            self,
            kb_id: str,
            api_key: str,
            dimension: int = 768,
            metric: str = "cosine"
    ):
        self.kb_id = kb_id
        self.api_key = api_key
        self.index_name = kb_id
        self.dimension = dimension
        self.metric = metric

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), load_plugins=False)

        # Create or connect to the index
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)

        # In-memory mapping for doc_id to vector IDs (consider persistent storage in production)
        self.doc_id_to_ids = {}

    def get_num_vectors(self):
        stats = self.index.describe_index_stats()
        return stats.get('total_vector_count', 0)

    def add_vectors(self, vectors: List[np.ndarray], metadata: List[dict]):
        # Convert NumPy arrays to lists
        vectors_as_lists = [
            vector.tolist() if isinstance(vector, np.ndarray) else vector
            for vector in vectors
        ]
        try:
            assert len(vectors_as_lists) == len(metadata)
        except AssertionError:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            )

        # Create the IDs from the metadata, defined as {metadata["doc_id"]}_{metadata["chunk_index"]}
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]

        # Prepare data for upsert
        upsert_data = []
        for vector_id, vector, meta in zip(ids, vectors_as_lists, metadata):
            upsert_data.append((vector_id, vector, meta))

            # Track IDs for each doc_id
            doc_id = meta["doc_id"]
            self.doc_id_to_ids.setdefault(doc_id, []).append(vector_id)

        # Upsert data into Pinecone
        self.index.upsert(vectors=upsert_data)

    def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 10,
            metadata_filter: Optional[MetadataFilter] = None
    ) -> List[VectorSearchResult]:
        num_vectors = self.get_num_vectors()
        if num_vectors == 0:
            return []

        # Convert query vector to list if it's a NumPy array
        query_vector = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        # Prepare filter if any
        if metadata_filter:
            formatted_metadata_filter = format_metadata_filter(metadata_filter)
        else:
            formatted_metadata_filter = None

        # Perform the query
        query_response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=formatted_metadata_filter
        )

        results: List[VectorSearchResult] = []
        for match in query_response.get('matches', []):
            similarity = match['score']  # For cosine similarity in Pinecone, higher scores mean more similar
            metadata = match['metadata']
            results.append(
                VectorSearchResult(
                    doc_id=metadata.get("doc_id"),
                    vector=None,
                    metadata=metadata,
                    similarity=similarity,
                )
            )

        # Results are already sorted by similarity
        return results

    def remove_document(self, doc_id: str):
        # Retrieve the IDs associated with the doc_id
        ids_to_delete = self.doc_id_to_ids.get(doc_id, [])
        if ids_to_delete:
            self.index.delete(ids=ids_to_delete)
            # Remove the doc_id entry
            del self.doc_id_to_ids[doc_id]

    def delete(self):
        pinecone.delete_index(self.index_name)

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "index_name": self.index_name,
        }

