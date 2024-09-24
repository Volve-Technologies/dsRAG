__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import cohere
import os
from typing import List

from dsrag.database.vector.types import MetadataFilter
from dsrag.knowledge_base import KnowledgeBase
from dsrag.rse import get_best_segments
from dsrag.llm import AnthropicChatAPI, OpenAIChatAPI
from dsrag.reranker import CohereReranker
from shared.database.get_databases import get_chunk_db, get_weaviate_db
from scipy.stats import beta
from dotenv import load_dotenv
import numpy as np

load_dotenv()
def transform(x):
    """
    Transformation function to map the absolute relevance value to a value that is more uniformly distributed between 0 and 1
    - This is critical for the new version of RSE to work properly, because it utilizes the absolute relevance values to calculate the similarity scores
    - The relevance values given by the Cohere reranker tend to be very close to 0 or 1. This beta function used here helps to spread out the values more uniformly.
    """
    a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
    return beta.cdf(x, a, b)


def rerank_documents(query: str, documents: list) -> list:
    """
    Use Cohere Rerank API to rerank the search results
    """
    model = "rerank-multilingual-v3.0"
    client = cohere.Client(api_key=os.environ["CO_API_KEY"])
    decay_rate = 30

    reranked_results = client.rerank(model=model, query=query, documents=documents)
    results = reranked_results.results
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]  # in order of reranked_indices

    # convert back to order of original documents and calculate the chunk values
    similarity_scores = [0] * len(documents)
    chunk_values = [0] * len(documents)
    for i, index in enumerate(reranked_indices):
        absolute_relevance_value = transform(reranked_similarity_scores[i])
        similarity_scores[index] = absolute_relevance_value
        v = np.exp(-i / decay_rate) * absolute_relevance_value  # decay the relevance value based on the rank
        chunk_values[index] = v

    return similarity_scores, chunk_values

def get_context(query: List[str], kb: KnowledgeBase, metadata_filter: MetadataFilter):
    context = ""
    doc_ids = set()
    results = kb.query(query,
                       rse_params={
                           "max_length": 300,
                           "overall_max_length": 700,
                           "minimum_value": 0.01,
                           "irrelevant_chunk_penalty": 0},
                       metadata_filter=metadata_filter
                       )
    for result in results:
        context += result["text"] + "\n"
        doc_id = result["doc_id"]
        doc_ids.add(str(doc_id))
    return context, doc_ids

def get_kb_non_english(kb_id, language: str):
    reranker = CohereReranker(model="rerank-multilingual-v3.0")
    #llm = AnthropicChatAPI(model="claude-3-5-sonnet-20240620", max_tokens=4000)
    llm = OpenAIChatAPI(temperature=0.3, max_tokens=4000)
    vector = get_weaviate_db(kb_id)
    postgres = get_chunk_db(kb_id)
    kb = KnowledgeBase(kb_id, vector_db=vector, chunk_db=postgres, auto_context_model=llm, language=language, reranker=reranker, save_metadata_to_disk=False)
    #kb = KnowledgeBase(kb_id, vector_db=None, chunk_db=postgres, language=language, reranker=reranker,
    #                   save_metadata_to_disk=True)
    return kb


def get_context_test(query: str, kb: KnowledgeBase, doc_id: str):
    num_chunks = len(kb.chunk_db.data[doc_id])
    documents = []
    chunks = []
    for i in range(num_chunks):
        chunk = {
            "section_title": kb.chunk_db.get_section_title(doc_id, i),
            "chunk_text": kb.chunk_db.get_chunk_text(doc_id, i)
        }
        section_context = f"Section: {kb.chunk_db.get_section_title(doc_id, i)}"
        chunk_text = kb.chunk_db.get_chunk_text(doc_id, i)
        document = f"Document: Unknown\n{section_context}\n\n{chunk_text}"
        documents.append(document)
        chunks.append(chunk)

    similarity_scores, chunk_values = rerank_documents(query, documents)
    context = ""
    all_relevence_values = [[v - 0 for v in chunk_values]]
    document_splits = []
    max_length = 30
    overall_max_length = 50
    minimum_value = 0.01
    best_segments, scores = get_best_segments(all_relevence_values, document_splits, max_length, overall_max_length, minimum_value)
    for segment_start, segment_end in best_segments:
        segment_text = ""
        for i in range(segment_start, segment_end):
            segment_text += chunks[i]["chunk_text"] + "\n"
        context += segment_text
    return context