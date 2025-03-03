"""
Domain-Specific Query Understanding with LLMs
Example from a former student who implemented domain-specific query processing
using Samba nova's LLM to improve search accuracy in their field.
"""

import os
from pydantic import BaseModel
from openai import OpenAI

class DomainContext(BaseModel):
    """
    Handles domain-specific context and terminology.
    Abstracts the specific industry/domain knowledge.
    """
    terms: dict = {}  # Domain-specific terms and their meanings
    relationships: dict = {}  # Related concepts
    synonyms: dict = {}  # Domain-specific synonyms
    
    def get_context_prompt(self):
        """Convert domain context into LLM prompt format"""
        context = ["Given the following domain-specific context:"]
        
        context.append("\nKey Terms:")
        for term, meaning in self.terms.items():
            context.append(f"- {term}: {meaning}")
            
        context.append("\nRelationships:")
        for concept, related in self.relationships.items():
            context.append(f"- {concept} relates to: {', '.join(related)}")
            
        return "\n".join(context)

class QueryResponse(BaseModel):
    """Structure for the LLM response"""
    identified_terms: list = []
    related_concepts: list = []
    expanded_interpretation: str = ""
    search_parameters: dict = {}

class QueryProcessor:
    """
    Handles query understanding and enhancement using Samba nova LLM.
    """
    def __init__(self, domain_context: DomainContext):
        self.context = domain_context
        self.client = OpenAI()
        
    def enhance_query(self, query: str) -> QueryResponse:
        """
        Enhance user query with domain understanding.
        Returns structured interpretation of the query.
        """
        prompt = f"""
        {self.context.get_context_prompt()}

        User Query: "{query}"

        Please analyze this query and provide:
        1. Identified domain terms
        2. Related concepts to consider
        3. Expanded interpretation
        4. Suggested search parameters

        Format response as JSON with the following structure:
        {{
            "identified_terms": [],
            "related_concepts": [],
            "expanded_interpretation": "",
            "search_parameters": {{}}
        }}
        """
        
        response = self.client.chat.completions.create(
            model="claude-v2",  # Samba nova model identifier
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        
        # Parse response into structured format
        try:
            return QueryResponse.model_validate_json(
                response.choices[0].message.content
            )
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return QueryResponse()

def main():
    # Example domain context
    domain_context = DomainContext(
        terms={
            "term1": "Definition of first key term",
            "term2": "Definition of second key term",
        },
        relationships={
            "concept1": ["related1", "related2"],
            "concept2": ["related3", "related4"],
        },
        synonyms={
            "term1": ["synonym1", "synonym2"],
            "term2": ["synonym3", "synonym4"],
        }
    )
    
    # Initialize query processor
    processor = QueryProcessor(domain_context)
    
    # Example query
    query = "Example query using domain-specific terminology"
    
    # Process query
    result = processor.enhance_query(query)
    
    # Print structured results
    print("\nEnhanced understanding of query:")
    print(f"Identified terms: {result.identified_terms}")
    print(f"Related concepts: {result.related_concepts}")
    print(f"Interpretation: {result.expanded_interpretation}")
    print(f"Search parameters: {result.search_parameters}")

if __name__ == "__main__":
    main() 