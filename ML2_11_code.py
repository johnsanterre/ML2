"""
Week 11: Practical LLM Integration & API Development
Example Implementation

This module demonstrates:
1. API Integration with error handling
2. Prompt template management
3. Response validation
4. Caching and optimization
5. Monitoring and logging
"""

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class PromptTemplate:
    """Manages versioned prompt templates with parameter validation."""
    version: str
    template: str
    required_params: List[str]
    description: str
    
    def format(self, **kwargs) -> str:
        """Format the template with provided parameters."""
        missing_params = set(self.required_params) - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        return self.template.format(**kwargs)

class ResponseValidator:
    """Validates and processes LLM responses."""
    
    def validate_json_response(self, response: str) -> Dict:
        """Validate JSON structure and required fields."""
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response")

    def validate_content(self, content: str, requirements: Dict) -> bool:
        """Validate content meets specified requirements."""
        if requirements.get('min_length') and len(content) < requirements['min_length']:
            return False
        if requirements.get('max_length') and len(content) > requirements['max_length']:
            return False
        return True

class ResponseCache:
    """Implements caching for LLM responses."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def _generate_key(self, prompt: str, params: Dict) -> str:
        """Generate cache key from prompt and parameters."""
        key_content = f"{prompt}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_content.encode()).hexdigest()

    def get(self, prompt: str, params: Dict) -> Optional[str]:
        """Retrieve cached response if valid."""
        key = self._generate_key(prompt, params)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['response']
            del self.cache[key]
        return None

    def set(self, prompt: str, params: Dict, response: str):
        """Cache a response."""
        key = self._generate_key(prompt, params)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }

class LLMMetrics:
    """Tracks and reports LLM usage metrics."""
    
    def __init__(self):
        self.request_times = []
        self.token_counts = []
        self.error_counts = {'rate_limit': 0, 'validation': 0, 'api': 0}
        self.cache_hits = 0
        self.cache_misses = 0

    def record_request(self, duration: float, tokens: int):
        """Record metrics for a request."""
        self.request_times.append(duration)
        self.token_counts.append(tokens)

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1

    def get_statistics(self) -> Dict:
        """Calculate and return current statistics."""
        if not self.request_times:
            return {"error": "No data available"}
        
        return {
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "total_tokens": sum(self.token_counts),
            "error_counts": self.error_counts,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0
        }

class LLMManager:
    """Main class for managing LLM interactions."""
    
    def __init__(self):
        self.templates = {}
        self.cache = ResponseCache()
        self.validator = ResponseValidator()
        self.metrics = LLMMetrics()
        self.setup_templates()

    def setup_templates(self):
        """Initialize prompt templates."""
        self.templates['classification'] = PromptTemplate(
            version="1.0",
            template="Classify the following text into {categories}:\n\n{text}",
            required_params=['categories', 'text'],
            description="General text classification template"
        )

    async def process_request(self, template_name: str, params: Dict) -> str:
        """Process a request with caching and error handling."""
        start_time = time.time()
        
        try:
            # Check cache
            cached_response = self.cache.get(template_name, params)
            if cached_response:
                self.metrics.cache_hits += 1
                return cached_response

            self.metrics.cache_misses += 1
            
            # Format prompt
            template = self.templates.get(template_name)
            if not template:
                raise ValueError(f"Unknown template: {template_name}")
            
            prompt = template.format(**params)
            
            # Simulate API call (replace with actual API call)
            response = await self._mock_llm_call(prompt)
            
            # Validate response
            validated_response = self.validator.validate_json_response(response)
            
            # Cache result
            self.cache.set(template_name, params, response)
            
            # Record metrics
            self.metrics.record_request(
                duration=time.time() - start_time,
                tokens=len(prompt.split())  # Simplified token counting
            )
            
            return response

        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            self.metrics.record_error(type(e).__name__)
            raise

    async def _mock_llm_call(self, prompt: str) -> str:
        """Mock LLM API call for demonstration."""
        # Simulate API latency
        await asyncio.sleep(0.5)
        return json.dumps({"classification": "example", "confidence": 0.95})

async def main():
    """Example usage of the LLM management system."""
    llm_manager = LLMManager()
    
    # Example classification request
    params = {
        "categories": ["positive", "negative", "neutral"],
        "text": "This product exceeded my expectations!"
    }
    
    try:
        result = await llm_manager.process_request('classification', params)
        print(f"Classification result: {result}")
        
        # Print metrics
        print("\nSystem metrics:")
        print(json.dumps(llm_manager.metrics.get_statistics(), indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 