__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import cohere
import os
from typing import List
from dsrag.knowledge_base import KnowledgeBase
from dsrag.rse import get_best_segments
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

def get_context(query: List[str], kb_id: str):
    kb = KnowledgeBase(kb_id)
    context = ""
    results = kb.query(query,
                       rse_params={
                           "max_length": 30,
                           "overall_max_length": 50,
                           "minimum_value": 0.01,
                           "irrelevant_chunk_penalty": 0}
                       )
    for result in results:
        context += result["text"] + "\n"
    return context

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

def get_bid_query_prompt():
    return """
    You are an AI assistant that helps people find information. Do not answer to questions not related to the context given under 
    <document> tag: <document> {context} </document>.
    Skip any highly irrelevant questions. For example, if person asks 'Can you tell me a joke' reply that you cannot do that.
    The question is under <question> tag that you are supposed to answer: <question> {query} </question>.
    """

def get_bid_query_prompt_rfp_related():
    prompt = get_bid_query_prompt() + """\n Bear in mind that you have to give answers specifically related to the project under rfp tag: <rfp> {rfp_name} </rfp>"""
    return prompt



def get_rfp_name_prompt():
    return """
    Use only the following information under <document> tag: <document> {context} </document>. Give a very short and concise answer on question: What is the name of this project?
    """

def current_rfp_summary_prompt():
    return """
    You are an AI assistant that helps people find information.\r\nYou are going to be asked to summarize a 'summarization' under a RFP summarization. \r\nTopic in other words is a category like \"Project Title\" or \"About\" or \"Key Deliverables\"  \r\nPlease generate an RFP summarization for these documents marked with <document> tag: <document>{context}</document>    \r\nKey Points to Consider:  \r\n- An RFP (Request for Proposal) is a formal document used to solicit proposals from potential consultants, contractors or suppliers.  \r\n- RFP content can vary depending on project phase, contract type, and procurement process.    Synonyms:  \r\n- Invitation to Tender (ITT)  \r\n- Request for Tender (RFT)  \r\n- Request for Quotation (RFQ) \r\n- Often used for smaller projects  \r\n- Pre-qualification Questionnaire (PQQ)  \r\n- Qualification    \r\nThe RFP summarization should consist of below topics:   {query}     \r\nYour answer should be in a form of a JSON with the following format:  \r\n{{    \"topicKey\": \"summarization\",    \r\n\"statements\": [\"answer1 with doc1 citation tags\", \"answer2  with doc1 citation tags\", ...],\r\n   \"sources\": [\"exact quote for answer1 without doc1 citation tags\", \"exact quote for answer2 without doc1 citation tags\"],\r\n   \"foundTopic \": true,    \r\n\"thoughts\": \"brief thoughts on how you came up with the answer, e.g. what sources you used, what you thought about, etc. without doc1 citation tags \"\r\n}}    \r\nAdditional instructions:  \r\n- You must generate citation based on the retrieved documents in the response, include citation tags [doc1], [doc2] in answer statements, do not include them in the other response json fields like \"sources\" or \"thoughts\"\r\n- Prioritize clarity, accuracy, and conciseness in your summary.   \r\n- If any information for the topics is missing or unclear, indicate it in the summary and suggest potential areas for clarification.  \r\n- For each point you identify, note down the exact sentence or phrase from which you derived this information and put them into relevant \"sources\" array\r\n- Explain on how you came up with the answer, e.g. what sources you used, what you thought about, etc. and put that description into \"thoughts\" field  \r\nWrite RFP summarization in {language} language but you must preserve original language in quotes sources
    """
def current_bid_summary_prompt():
    return """
    You are an AI assistant that helps people find information.\r\nPlease generate a Bid summarization for these documents marked with <document> tag: <document>{context}</document>  \r\nHere's an explanation what Bid is:\r\n- A bid is a formal response to a Request for Proposal (RFP), submitted by a consultant, contractor or supplier.\r\n- Bid content can vary depending on the project phase, type of contract, and the specific requirements outlined in the RFP.\r\n\r\nSynonyms:\r\n- Proposal\r\n- Submission\r\n- Tender\r\n\r\nThe Bid summarization should consist of below topics:\r\n {query} \r\n\r\nYour answer should be in a form of a JSON with the following format:\r\n{{\r\n  \"topicKey\": \"summarization\"\r\n  \"statements\": [\"answer\", \"answer\", ...],\r\n  \"foundTopic \": true,\r\n  \"thoughts\": \/\/ brief thoughts on how you came up with the answer, e.g. what sources you used, what you thought about, etc.\r\n}}\r\n\r\nIf you can't find anything related to particular summarization in provided documents, then point it out.\r\nInclude citation tags [doc1], [doc2], etc. in relevant topicAnswer fields.  \r\n\r\n Write BID summarization in {language} language
    """

def bid_summary_prompt():
    return """
    You are an AI assistant that helps people find information.\r\nPlease generate a Bid summarization for these documents marked with <document> tag: <document>{context}</document> and with project of interest described in <rfp> tag: <rfp>{rfp}</rfp> \r\nHere's an explanation what Bid is:\r\n- A bid is a formal response to a Request for Proposal (RFP), submitted by a consultant, contractor or supplier.\r\n- Bid content can vary depending on the project phase, type of contract, and the specific requirements outlined in the RFP.\r\n\r\nSynonyms:\r\n- Proposal\r\n- Submission\r\n- Tender\r\n\r\nThe Bid summarization should consist of below topics:\r\n {query} \r\n\r\nYour answer should be in a form of a JSON with the following format:\r\n{{\r\n  \"topicKey\": \"summarization\"\r\n  \"statements\": [\"answer\", \"answer\", ...],\r\n \"sources\": [\"exact quote for answer\", \"exact quote for answer\"],\r\n \"foundTopic \": true,\r\n  \"thoughts\": \/\/ brief thoughts on how you came up with the answer, e.g. what sources you used, what you thought about, etc.\r\n}}\r\n\r\nFor each point you identify, note down the exact sentence or phrase from which you derived this information. If you can't find anything related to particular summarization in provided documents, then point it out.\r\nInclude citation tags [doc1], [doc2], etc. in relevant topicAnswer fields.  \r\n\r\n Write BID summarization in {language} language
    """

def bid_new_summary_prompt_anthropic():
    return """
    You are an AI assistant tasked with analyzing a bid document and providing a valuable summary based on a specific question. Your goal is to extract relevant information and present it in a structured format. Here's how to approach this task:

1. First, you will be given a bid document to analyze. The content of this document will be provided within <document> tags:

<document>
{context}
</document>

2. You will then be presented with a specific question about the document within <question> tags:

<question>
{query}
</question>

3. Your task is to analyze the document and provide a summary that answers the given question. The summary should consist of 4-5 most valuable and concrete points related to the question. Avoid vague statements and focus on specific, actionable information.

4. For each point in your summary, you must provide the exact sentence or phrase from the document that supports your statement. These will serve as sources for your claims.

5. Your output should be in JSON format with the following structure:

{{
  "topicKey": "summarization",
  "statements": ["answer1", "answer2", "answer3", "answer4"],
  "sources": ["exact quote for answer1", "exact quote for answer2", "exact quote for answer3", "exact quote for answer4"],
  "foundTopic": true
}}

6. The "topicKey" should be a brief description of the main summarization addressed by the question.

7. The "statements" array should contain 4-5 concise, valuable points that answer the question based on the document.

8. The "sources" array should contain the exact quotes from the document that support each statement in the same order.

9. Set "foundTopic" to true if you were able to find relevant information in the document to answer the question. If you couldn't find any relevant information, set it to false.

10. Use only the information provided in the document. Do not include any external knowledge or assumptions.

11. Before providing your final answer, use <scratchpad> tags to organize your thoughts and identify the most relevant information from the document.

Here's an example of a good response:

<scratchpad>
- Question asks about project timeline
- Document mentions start date on page 2: "Project commences on July 1, 2023"
- Completion date found on page 5: "Expected completion by December 31, 2023"
- Milestones listed on page 3: "Phase 1 (August), Phase 2 (October), Phase 3 (December)"
- Budget information on page 4: "Total project budget: $500,000"
</scratchpad>

{{
  "topicKey": "Project Timeline",
  "statements": [
    "The project starts on July 1, 2023",
    "The expected completion date is December 31, 2023",
    "The project has three main phases: August, October, and December",
    "The total project budget is $500,000"
  ],
  "sources": [
    "Project commences on July 1, 2023",
    "Expected completion by December 31, 2023",
    "Phase 1 (August), Phase 2 (October), Phase 3 (December)",
    "Total project budget: $500,000"
  ],
  "foundTopic": true
}}

Here's an example of a bad response:

{{
  "topicKey": "Project Details",
  "statements": [
    "The project seems to be well-planned",
    "It might take several months to complete",
    "There are multiple phases involved",
    "The budget appears to be substantial"
  ],
  "sources": [
    "N/A",
    "N/A",
    "N/A",
    "N/A"
  ],
  "foundTopic": true
}}

This response is poor because it provides vague statements without specific details and fails to include exact quotes from the document as sources.

Remember to always provide concrete, valuable information directly from the document and include the exact supporting quotes as sources. 

Now, analyze the provided document based on the given question and provide your response in the specified JSON format and answer only in {language} language.
    """

def rfp_summary_prompt():
    return """
    1. First, you will be provided with the RFP document:
<rfp_document>
{context}
</rfp_document>

2. Carefully read through the entire document, focusing on sections that might contain information about goals and deliverables. These are often found in sections labeled "Objectives," "Scope of Work," "Requirements," or "Deliverables."

3. As you read, {query}

4. Extract 3-4 key points that represent the overall goals and desired deliverables. These should be:
   - Very precise and specific to this RFP
   - Avoid generalizations
   - Directly derived from the text

5. For each point you identify, note down the exact sentence or phrase from which you derived this information.

6. Present your findings in the following format:
    {{
        "topicKey": "summarization",
        "statements": ["answer", "answer", ...],
        "source": "[Exact quote from the document]"
        "foundTopic": true,
        "thoughts": "brief thoughts on how you came up with the answer, e.g. what sources you used, what you thought about, etc."
    }}


7. If you cannot find any clear goals or deliverables in the document, or if there are fewer than 3 distinct points, state this clearly:
   <findings>
   No clear goals or deliverables were found in the provided RFP document.
   </findings>

8. Remember, only use the context provided in the RFP document. Do not include any external information or make assumptions beyond what is explicitly stated in the document.

9. Ensure that your extracted points are truly representative of the overall goals and desired deliverables, not minor details or procedural instructions.

10. If you're unsure about including a point, err on the side of caution and only include points that are clearly stated as goals or deliverables in the document.

Present your final output within <answer> tags in {language} language.
    """