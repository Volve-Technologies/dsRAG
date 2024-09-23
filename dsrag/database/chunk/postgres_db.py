import os
import time
import psycopg2
from psycopg2 import pool
from typing import Any, Optional
import logging
import threading
from dsrag.database.chunk.db import ChunkDB
from dsrag.database.chunk.types import FormattedDocument

connection_pool = None
pool_lock = threading.Lock()

class PostgresDB(ChunkDB):
    def __init__(self, kb_id: str, connection_string: str) -> None:
        self.kb_id = kb_id
        self.connection_string = connection_string
        self.columns = [
            {"name": "doc_id", "type": "TEXT"},
            {"name": "document_title", "type": "TEXT"},
            {"name": "document_summary", "type": "TEXT"},
            {"name": "section_title", "type": "TEXT"},
            {"name": "section_summary", "type": "TEXT"},
            {"name": "chunk_text", "type": "TEXT"},
            {"name": "chunk_index", "type": "INT"},
            {"name": "chunk_length", "type": "INT"},
            {"name": "chunk_page_start", "type": "INT"},
            {"name": "chunk_page_end", "type": "INT"},
            {"name": "created_on", "type": "TEXT"},
            {"name": "supp_id", "type": "TEXT"},
            {"name": "metadata", "type": "TEXT"},
        ]
        self.init_connection_pool()
        self.create_table_if_not_exists()

    def init_connection_pool(self):
        global connection_pool
        with pool_lock:
            if connection_pool is None:
                connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=100,
                    dsn=self.connection_string
                )
        logging.warning(f"The connection bool object is: {connection_pool}")

    def get_connection(self):
        return connection_pool.getconn()

    def put_connection(self, conn):
        connection_pool.putconn(conn)

    def create_table_if_not_exists(self):
        conn = self.get_connection()
        try:
            c = conn.cursor()
            c.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name='documents'"
            )
            if not c.fetchone():
                query_statement = "CREATE TABLE documents ("
                for column in self.columns:
                    query_statement += f"{column['name']} {column['type']}, "
                query_statement = query_statement[:-2] + ")"
                c.execute(query_statement)
                conn.commit()
                # Create index
                c.execute("CREATE INDEX idx_documents_doc_id_chunk_index ON documents (doc_id, chunk_index);")
                conn.commit()
            c.close()
        finally:
            self.put_connection(conn)

    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]], supp_id: str = "",
                     metadata: dict = {}) -> None:
        # Add the docs to the PostgreSQL table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        # Create a created on timestamp
        created_on = str(int(time.time()))

        # Turn the metadata object into a string
        metadata = str(metadata)

        # Get the data from the dictionary
        for chunk_index, chunk in chunks.items():
            document_title = chunk.get("document_title", "")
            document_summary = chunk.get("document_summary", "")
            section_title = chunk.get("section_title", "")
            section_summary = chunk.get("section_summary", "")
            chunk_text = chunk.get("chunk_text", "")
            chunk_page_start = chunk.get("chunk_page_start", None)
            chunk_page_end = chunk.get("chunk_page_end", None)
            chunk_length = len(chunk_text)
            c.execute(
                "INSERT INTO documents (doc_id, document_title, document_summary, section_title, section_summary, chunk_text, chunk_page_start, chunk_page_end, chunk_index, chunk_length, created_on, supp_id, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    doc_id,
                    document_title,
                    document_summary,
                    section_title,
                    section_summary,
                    chunk_text,
                    chunk_page_start,
                    chunk_page_end,
                    chunk_index,
                    chunk_length,
                    created_on,
                    supp_id,
                    metadata
                ),
            )

        conn.commit()
        conn.close()

    def remove_document(self, doc_id: str) -> None:
        # Remove the docs from the PostgreSQL table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        conn.commit()
        conn.close()

    def get_document(
            self, doc_id: str, include_content: bool = False
    ) -> Optional[FormattedDocument]:
        # Retrieve the document from the PostgreSQL table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        columns = ["supp_id", "document_title", "document_summary", "created_on", "metadata"]
        if include_content:
            columns += ["chunk_text", "chunk_index"]

        query_statement = (
            f"SELECT {', '.join(columns)} FROM documents WHERE doc_id=%s"
        )
        c.execute(query_statement, (doc_id,))
        results = c.fetchall()
        conn.close()

        # If there are no results, return None
        if not results:
            return None

        # Turn the results into an object where the columns are keys
        full_document_string = ""
        if include_content:
            # Concatenate the chunks into a single string
            for result in results:
                # Join each chunk text with a new line character
                full_document_string += result[columns.index("chunk_text")] + "\n"
            # Remove the last new line character
            full_document_string = full_document_string[:-1]

        supp_id = results[0][columns.index("supp_id")]
        title = results[0][columns.index("document_title")]
        summary = results[0][columns.index("document_summary")]
        created_on = results[0][columns.index("created_on")]
        metadata = results[0][columns.index("metadata")]

        # Convert the metadata string back into a dictionary
        if metadata:
            metadata = eval(metadata)

        return FormattedDocument(
            id=doc_id,
            supp_id=supp_id,
            title=title,
            content=full_document_string if include_content else None,
            summary=summary,
            created_on=created_on,
            metadata=metadata
        )

    def get_chunk_text_range(self, doc_id: str, chunk_index_start: int, chunk_index_end: int) -> Optional[str]:
        doc_id = str(doc_id)

        # Retrieve the chunk text from the PostgreSQL table for the given range of chunk indices
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT chunk_text FROM documents WHERE doc_id=%s AND chunk_index BETWEEN %s AND %s",
            (doc_id, chunk_index_start, chunk_index_end)
        )
        results = c.fetchall()
        conn.close()

        if results:
            # Concatenate all chunk_texts into a single string
            chunk_texts = "\n".join([result[0] for result in results])
            return chunk_texts
        return None

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the chunk text from the PostgreSQL table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT chunk_text FROM documents WHERE doc_id=%s AND chunk_index=%s", (doc_id, chunk_index)
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    #####
    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Optional[tuple[int, int]]:
        # Retrieve the chunk page numbers from the sqlite table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT chunk_page_start, chunk_page_end FROM documents WHERE doc_id=%s AND chunk_index=%s", (doc_id, chunk_index)
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result
        return None

    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the document title from the sqlite table
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            f"SELECT document_title FROM documents WHERE doc_id=%s AND chunk_index=%s", (str(doc_id), chunk_index)
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the document summary from the sqlite table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            f"SELECT document_summary FROM documents WHERE doc_id=%s AND chunk_index=%s", (doc_id, chunk_index)
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the section title from the sqlite table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            f"SELECT section_title FROM documents WHERE doc_id=%s AND chunk_index=%s", (doc_id, chunk_index)
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None

    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        # Retrieve the section summary from the sqlite table
        doc_id = str(doc_id)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(
            f"SELECT section_summary FROM documents WHERE doc_id=%s AND chunk_index=%s", (doc_id, chunk_index)
        )
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        return None
####
    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> list[str]:
        # Retrieve all document IDs from the PostgreSQL table
        conn = self.get_connection()
        c = conn.cursor()
        query_statement = "SELECT DISTINCT doc_id FROM documents"
        if supp_id:
            query_statement += " WHERE supp_id=%s"
            c.execute(query_statement, (supp_id,))
        else:
            c.execute(query_statement)
        results = c.fetchall()
        conn.close()
        return [result[0] for result in results]

    def get_document_count(self) -> int:
        # Retrieve the number of documents in the PostgreSQL table
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(DISTINCT doc_id) FROM documents")
        result = c.fetchone()
        conn.close()
        if result is None:
            return 0
        return result[0]

    def get_total_num_characters(self) -> int:
        # Retrieve the total number of characters in the PostgreSQL table
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("SELECT SUM(chunk_length) FROM documents")
        result = c.fetchone()
        conn.close()
        if result is None or result[0] is None:
            return 0
        return result[0]

    def delete(self) -> None:
        # Delete all documents (truncate table)
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("TRUNCATE TABLE documents")
        conn.commit()
        conn.close()

    def to_dict(self) -> dict[str, str]:
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
        }

