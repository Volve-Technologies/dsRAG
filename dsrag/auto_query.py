import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from typing import List

SYSTEM_MESSAGE = """
You are a query generation system. Please generate one or more search queries (up to a maximum of {max_queries}) based on the provided user input. DO NOT generate the answer, just queries.

Each of the queries you generate will be used to search a knowledge base for information that can be used to respond to the user input. Make sure each query is specific enough to return relevant information. If multiple pieces of information would be useful, you should generate multiple queries, one for each specific piece of information needed.

{auto_query_guidance}
""".strip()

PROMPT = """
You are an advanced query generation system designed to create search queries for a Retrieval-Augmented Generation (RAG) system. Your task is to generate up to three search queries based on the provided user input. These queries will be used to search a knowledge base for relevant information.

Here is the user's input:

<user_input>
{query}
</user_input>

Instructions:
1. Analyze the user's input carefully.
2. Generate up to three search queries that will help retrieve the most relevant information from the knowledge base.
3. Ensure each query adheres to the following criteria:
   - Is a single sentence
   - Is short and concise
   - Is specific enough to return relevant information
   - Addresses a distinct aspect of the user's input (if multiple queries are generated)

Before generating the final queries, wrap your thought process inside <thinking> tags. In your thinking process:
1. Break down the user's input into key components or topics.
2. Consider different aspects or angles of the query that might require separate search queries.
3. For each potential query, evaluate its specificity and relevance to the user's input.
4. Determine how many queries (up to {max_queries}) are necessary to cover the user's input comprehensively.

After your analysis, provide the final queries in the following format:

class Queries(BaseModel):
    queries: List[str]

Your output should be a JSON representation of this structure, containing only the list of query strings.

Example output structure (do not use these specific queries, generate your own based on the user input):
{{
    "queries": [
        "What is X?",
        "How does Y affect Z?",
        "When was W discovered?"
    ]
}}

Remember, generate no more than 3 queries, and it's acceptable to generate fewer if the user's input doesn't require multiple queries.
""".strip()


def get_search_queries(user_input: str, auto_query_guidance: str = "", max_queries: int = 5):
    client = instructor.from_anthropic(Anthropic())

    class Queries(BaseModel):
        queries: List[str]

    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=400,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(query=user_input, max_queries=max_queries)
            }
        ],
        response_model=Queries,
    )

    return resp.queries[:max_queries]