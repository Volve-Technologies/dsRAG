from typing import Sequence, cast
import uuid
from dsrag.database.vector.types import ChunkMetadata, Vector, VectorSearchResult
from dsrag.database.vector.db import VectorDB
import numpy as np
from typing import Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range


def convert_id(_id: str) -> str:
    """
    Converts any string into a UUID string based on a seed.
    Qdrant accepts UUID strings and unsigned integers as point ID.
    We use a seed to convert each string into a UUID string deterministically.
    This allows us to overwrite the same point with the original ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, _id))


class QdrantVectorDB(VectorDB):
    """
    An implementation of the VectorDB interface for Qdrant - https://qdrant.tech/.
    """

    def __init__(
        self,
        kb_id: str,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
    ):
        """
        Initializes a QdrantVectorDB instance.
        Args:
            kb_id: An identifier for the knowledge base.
            location:
                If `":memory:"` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - use default values for `host` and `port`.
            url: either host or str of "Optional[scheme], host, Optional[port], Optional[prefix]".
                Default: `None`
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
            https: If `true` - use HTTPS(SSL) protocol. Default: `None`
            api_key: API key for authentication in Qdrant Cloud. Default: `None`
            prefix:
                If not `None` - add `prefix` to the REST URL path.
                Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.
                Default: `None`
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5 seconds for REST and unlimited for gRPC
            host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
                Default: `None`
            path: Persistence path for QdrantLocal. Default: `None`
        """

        self.kb_id = kb_id
        self.client_options = {
            "location": location,
            "url": url,
            "port": port,
            "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc,
            "https": https,
            "api_key": api_key,
            "prefix": prefix,
            "timeout": timeout,
            "host": host,
            "path": path,
        }
        self.client = QdrantClient(**self.client_options)

    def close(self):
        """
        Closes the connection to Qdrant
        """
        self.client.close()

    def add_vectors(
        self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]
    ) -> None:
        """
        Adds a list of vectors with associated metadata to Qdrant
        Args:
            vectors: A list of vector embeddings.
            metadata: A list of dictionaries containing metadata for each vector.
        Raises:
            ValueError: If the number of vectors and metadata items do not match.
        """
        try:
            assert len(vectors) == len(metadata)
        except AssertionError as exc:
            raise ValueError(
                "Error in add_vectors: the number of vectors and metadata items must be the same."
            ) from exc

        if not self.client.collection_exists(self.kb_id):
            self.client.create_collection(
                self.kb_id,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
        points = []
        for vector, meta in zip(vectors, metadata):
            doc_id = meta.get("doc_id", "")
            chunk_text = meta.get("chunk_text", "")
            chunk_index = meta.get("chunk_index", 0)
            uuid = convert_id(f"{doc_id}_{chunk_index}")
            tender_or_bid_id = meta.get("tender_or_bid_id", "")
            external_kb = meta.get("external_kb", "false")
            external_kb_region = meta.get("external_kb_region", "")
            external_kb_type = meta.get("external_kb_type", "")
            document_type = meta.get("document_type", "")
            type_id = meta.get("type_id", "")
            points.append(
                models.PointStruct(
                    id=uuid,
                    vector=vector,
                    payload={
                        "content": chunk_text,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "tender_or_bid_id": tender_or_bid_id,
                        "metadata": meta,
                        "external_kb": external_kb,
                        "external_kb_region": external_kb_region,
                        "external_kb_type": external_kb_type,
                        "document_type": document_type,
                        "type_id": type_id
                    },
                )
            )

        self.client.upsert(self.kb_id, points)

    def remove_document(self, doc_id) -> None:
        """
        Removes a document from a Qdrant collection.
        Args:
            doc_id: The UUID of the document to remove.
        """
        self.client.delete(
            self.kb_id,
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
        )

    def _delete_by_tender_bid_id(self, tender_or_bid_id: str):
        self.client.delete(
            self.kb_id,
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="tender_or_bid_id", match=models.MatchValue(value=tender_or_bid_id)
                    )
                ]
            ),
        )

    def search(
        self,
        query_vector: list,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """
        Searches for the top-k closest vectors to the given query vector.
        Args:
            query_vector: The query vector embedding.
            top_k: The number of results to return.
        Returns:
            A list of dictionaries containing the metadata and similarity scores of
            the top-k results.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        qdrant_filter = self._convert_metadata_filter(metadata_filter) if metadata_filter else None

        results: list[VectorSearchResult] = []
        response = self.client.query_points(
            self.kb_id,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True,
            timeout=60
        ).points
        for point in response:
            results.append(
                VectorSearchResult(
                    doc_id=cast(str, point.payload.get("doc_id")),
                    metadata=cast(ChunkMetadata, point.payload.get("metadata")),
                    similarity=point.score,
                    vector=cast(Vector, point.vector),
                )
            )
        return results

    def get_num_vectors(self):
        return self.client.count(self.kb_id).count

    def delete(self):
        """Deletes the collection from Qdrant"""
        self.client.delete_collection(self.kb_id)

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            **self.client_options,
        }

    def _convert_metadata_filter(self, metadata_filter: dict) -> Filter:
        """
        Converts a metadata_filter dictionary into a qdrant_client.models.Filter object.
        """
        field = metadata_filter.get("field")
        operator = metadata_filter.get("operator")
        value = metadata_filter.get("value")

        if operator == "equals":
            condition = FieldCondition(
                key=field,
                match=MatchValue(value=value)
            )
        elif operator == "in":
            condition = FieldCondition(
                key=field,
                match=MatchAny(any=value)
            )
        elif operator in ["greater_than", "greater_than_equals", "less_than", "less_than_equals"]:
            range_args = {}
            if operator == "greater_than":
                range_args["gt"] = value
            elif operator == "greater_than_equals":
                range_args["gte"] = value
            elif operator == "less_than":
                range_args["lt"] = value
            elif operator == "less_than_equals":
                range_args["lte"] = value
            condition = FieldCondition(
                key=field,
                range=Range(**range_args)
            )
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        return Filter(
            must=[condition]
        )