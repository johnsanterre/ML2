# Domain-Specific Query Understanding with LLMs

## About the Inspiration Author: David Stroud

David Stroud has been an integral part of SMU's Machine Learning community since 2018, serving as the longest-tenured Teaching Assistant for the Machine Learning II course. His journey from early-career data scientist to senior practitioner exemplifies the evolution of the field itself.

### Professional Background
Currently working at Fox News in New York, David specializes in Large Language Models and their practical applications in enterprise environments. His career has been marked by a deliberate choice to maintain deep technical expertise rather than transitioning to pure management roles, reflecting his commitment to the craft of data science.

### Areas of Expertise
- Reinforcement Learning
- Data Engineering
- Enterprise LLM Implementation
- Production ML Systems

### Teaching Philosophy
David's approach to teaching data science emphasizes practical implementation and real-world problem-solving. His contributions to this course material reflect years of hands-on experience and a deep understanding of what emerging data scientists need to succeed in production environments.

### Impact
As one of the original members of our data science community, David's perspective on market trends and technological developments continues to influence how we prepare students for real-world challenges. This example demonstrates his approach to implementing LLMs in production systems, drawing from his recent experiences and insights.

## Overview
This example demonstrates how to use Samba nova's LLM to enhance query understanding in a domain-specific context. The implementation was used to improve search accuracy in a specialized field by incorporating domain knowledge into query processing.

## Data Validation with Pydantic

This implementation uses Pydantic for data validation and structure. Pydantic is particularly valuable when working with LLMs because:

### Benefits in LLM Applications
- **Validates LLM Responses**: Ensures JSON responses from the LLM match our expected structure
- **Type Safety**: Automatically validates data types for domain context and query responses
- **Error Handling**: Provides clear error messages when LLM responses don't match expected format
- **JSON Integration**: Seamlessly converts between JSON and Python objects

### Example in Our Code
```python
class QueryResponse(BaseModel):
    identified_terms: list = []
    related_concepts: list = []
    expanded_interpretation: str = ""
    search_parameters: dict = {}
```

This structure helps ensure that our LLM responses are properly formatted and contain the expected data types, making our query processing more robust and reliable.

## Key Components

### DomainContext
- Manages domain-specific terminology and relationships
- Provides structured representation of domain knowledge
- Converts domain context into LLM-readable format

### QueryProcessor
- Interfaces with Samba nova's LLM
- Enhances queries with domain understanding
- Provides structured interpretation of user queries

## Usage

1. Set up your API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

2. Run the example:
```bash
python query_understanding.py
```

## Customization
To adapt this for your domain:
1. Modify the DomainContext initialization with your domain-specific terms
2. Adjust the prompt template in enhance_query()
3. Customize the response parsing for your needs

## Results
In production, this approach improved query understanding accuracy by X% and reduced the need for query refinement by Y%.

## Notes
- API key and sensitive information should be managed securely
- Consider caching frequent queries for performance
- Monitor API usage and costs 