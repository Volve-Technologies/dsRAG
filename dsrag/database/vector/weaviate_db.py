from typing import Sequence, cast
from dsrag.database.vector.types import ChunkMetadata, Vector, VectorSearchResult
from dsrag.database.vector.db import VectorDB
import numpy as np
from typing import Optional
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from dsrag.database.vector.types import MetadataFilter
import logging


class WeaviateVectorDB(VectorDB):
    """
    An implementation of the VectorDB interface for Weaviate using the Python v4 client.

    This class provides methods for adding, removing, and searching for vectorized data
    within a Weaviate instance.
    """

    def __init__(
        self,
        kb_id: str,
        http_host="localhost",
        http_port="8099",
        http_secure=False,
        grpc_host="localhost",
        grpc_port="50052",
        grpc_secure=False,
        weaviate_secret="secr3tk3y",
        init_timeout: int = 2,
        query_timeout: int = 45,
        insert_timeout: int = 120,
        use_embedded_weaviate: bool = False,
    ):
        """
        Initializes a WeaviateVectorDB instance.

        Args:
            http_host: The hostname of the Weaviate server.
            http_port: The HTTP port of the Weaviate server.
            http_secure: Whether to use HTTPS for the connection.
            grpc_host: The hostname of the Weaviate server for gRPC connections.
            grpc_port: The gRPC port of the Weaviate server.
            grpc_secure: Whether to use gRPCs for the connection.
            class_name: The name of the Weaviate class to use for storing data.
            kb_id: An optional identifier for the knowledge base.
        """

        # save all of these parameters as attributes so they're easily accessible for the to_dict method
        self.kb_id = kb_id
        self.http_host = http_host
        self.http_port = http_port
        self.http_secure = http_secure
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.grpc_secure = grpc_secure
        self.weaviate_secret = weaviate_secret
        self.init_timeout = init_timeout
        self.query_timeout = query_timeout
        self.insert_timeout = insert_timeout
        self.use_embedded_weaviate = use_embedded_weaviate

        additional_headers = {}
        if use_embedded_weaviate:
            additional_headers["ENABLE_MODULES"] = (
                "backup-filesystem,text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai"
            )
            additional_headers["BACKUP_FILESYSTEM_PATH"] = "./weaviate/backups"
            self.client = weaviate.WeaviateClient(
                embedded_options=weaviate.embedded.EmbeddedOptions(
                    persistence_data_path="./weaviate/data",
                ),
                additional_headers=additional_headers,
            )
        else:

            self.client = weaviate.connect_to_wcs(
                cluster_url=self.http_host,
                auth_credentials=weaviate.auth.AuthApiKey(self.weaviate_secret))

        self.client.connect()
        self.collection_name = "dsrag_test"
        self.collection = self.client.collections.get(
            self.collection_name
        )

    def close(self):
        """
        Closes the connection to Weaviate.
        """
        self.client.close()

    def add_vectors(
        self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]
    ) -> None:
        """
        Adds a list of vectors with associated metadata to Weaviate.

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

        with self.collection.batch.dynamic() as batch:
            for vector, meta in zip(vectors, metadata):
                doc_id = meta.get("doc_id", "")
                chunk_text = meta.get("chunk_text", "")
                chunk_index = meta.get("chunk_index", 0)
                tender_or_bid_id = meta.get("tender_or_bid_id", "")
                uuid = generate_uuid5(f"{doc_id}_{chunk_index}")
                batch.add_object(
                    properties={
                        "content": chunk_text,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "tender_or_bid_id": tender_or_bid_id,
                        "metadata": meta,
                    },
                    vector=vector,
                    uuid=uuid,
                )

    def remove_document(self, doc_id) -> None:
        """
        Removes a document (data object) from Weaviate.

        Args:
            doc_id: The UUID of the document to remove.
        """
        self.collection.data.delete_many(
            where=wvc.query.Filter.by_property("doc_id").contains_any([doc_id])
        )

    def search(self, query_vector: list, top_k: int = 10, metadata_filter: Optional[dict] = None) -> list[
        VectorSearchResult]:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        results: list[VectorSearchResult] = []

        # Perform the vector search
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
            filters=wvc.query.Filter.by_property("tender_or_bid_id").contains_any(metadata_filter.get("value"))
        )

        for obj in response.objects:
            results.append(
                VectorSearchResult(
                    doc_id=cast(str, obj.properties["doc_id"]),
                    metadata=cast(ChunkMetadata, obj.properties["metadata"]),
                    similarity=cast(float, 1.0 - obj.metadata.distance),
                    vector=cast(Vector, obj.vector),
                )
            )
        return results

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "http_host": self.http_host,
            "http_port": self.http_port,
            "http_secure": self.http_secure,
            "grpc_host": self.grpc_host,
            "grpc_port": self.grpc_port,
            "grpc_secure": self.grpc_secure,
            "weaviate_secret": self.weaviate_secret,
            "init_timeout": self.init_timeout,
            "query_timeout": self.query_timeout,
            "insert_timeout": self.insert_timeout,
            "use_embedded_weaviate": self.use_embedded_weaviate,
        }

    def delete(self):
        pass
