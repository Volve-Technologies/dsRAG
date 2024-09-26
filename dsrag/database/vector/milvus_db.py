from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult, MetadataFilter
from typing import Optional
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


def format_metadata_filter(metadata_filter: MetadataFilter) -> str:
    """
    Format the metadata filter to be used in the Milvus search expression.

    Args:
        metadata_filter (dict): The metadata filter.

    Returns:
        str: The formatted metadata filter.
    """

    field = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]

    operator_mapping = {
        "equals": "==",
        "not_equals": "!=",
        "in": "in",
        "not_in": "not in",
        "greater_than": ">",
        "less_than": "<",
        "greater_than_equals": ">=",
        "less_than_equals": "<=",
    }

    formatted_operator = operator_mapping.get(operator)
    if formatted_operator is None:
        raise ValueError(f"Unsupported operator '{operator}' in metadata filter.")

    if operator in ["in", "not_in"]:
        if not isinstance(value, list):
            raise ValueError(f"Value must be a list for operator '{operator}'.")
        value_str = "[" + ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value) + "]"
    else:
        value_str = f"'{value}'" if isinstance(value, str) else str(value)

    expression = f"{field} {formatted_operator} {value_str}"
    return expression


class MilvusDB(VectorDB):
    def __init__(self, kb_id: str, uri: str, token: str, dim: int = 768):
        self.kb_id = kb_id
        self.dim = dim
        connections.connect(uri=uri, token=token)
        self.collection_name = kb_id
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="tender_or_bid_id", dtype=DataType.VARCHAR)
            ]
            schema = CollectionSchema(fields, description="Vector database collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            # Load the collection
            self.collection.load()
        else:
            self.collection = Collection(name=self.collection_name)
            self.collection.load()

    def get_num_vectors(self):
        return self.collection.num_entities

    def add_vectors(self, vectors: list, metadata: list):
        vectors_as_lists = [vector.tolist() if isinstance(vector, np.ndarray) else vector for vector in vectors]
        if len(vectors_as_lists) != len(metadata):
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            )

        # Create the ids from the metadata, defined as {metadata["doc_id"]}_{metadata["chunk_index"]}
        ids = [f"{meta['doc_id']}_{meta['chunk_index']}" for meta in metadata]

        # Prepare data for insertion
        data = {
            "pk": ids,
            "embedding": vectors_as_lists,
            "doc_id": [meta["doc_id"] for meta in metadata],
            "chunk_index": [meta["chunk_index"] for meta in metadata],
            "tender_or_bid_id": [meta.get("tender_or_bid_id", "") for meta in metadata]
        }

        # Convert data to list of lists (columns)
        data_to_insert = [data[field.name] for field in self.collection.schema.fields]

        # Insert data
        self.collection.insert(data_to_insert)

    def search(
        self, query_vector, top_k=10, metadata_filter: Optional[MetadataFilter] = None
    ) -> list[VectorSearchResult]:

        num_vectors = self.get_num_vectors()
        if num_vectors == 0:
            return []

        # Prepare search parameters
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        # Prepare the query vector
        query_vectors = [query_vector]

        # Specify which fields to return
        output_fields = ["doc_id", "chunk_index", "tender_or_bid_id"]  # Add other metadata fields if needed

        # Format metadata filter
        expression = format_metadata_filter(metadata_filter) if metadata_filter else None

        # Perform the search
        search_results = self.collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expression,
            output_fields=output_fields,
        )

        # Process the results
        results: list[VectorSearchResult] = []
        for hits in search_results:
            for hit in hits:
                similarity = 1 - hit.distance  # Convert distance to similarity
                metadata = {
                    "doc_id": hit.entity.get("doc_id"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    # Add other metadata fields here
                }
                results.append(
                    VectorSearchResult(
                        doc_id=metadata["doc_id"],
                        vector=None,
                        metadata=metadata,
                        similarity=similarity,
                    )
                )

        # Sort results by similarity in descending order
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results

    def remove_document(self, doc_id: str):
        expression = f"doc_id == '{doc_id}'"
        self.collection.delete(expr=expression)

    def delete(self):
        self.collection.drop()

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
        }
