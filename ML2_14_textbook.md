# LLMs as Decision Makers and Agents

## 1. Foundations of LLM-based Decision Making

### 1.1 Structured Output Generation
Converting LLM responses into actionable decisions:

```python
def get_structured_decision(llm, prompt, decision_schema):
    """Generate structured decisions using JSON schema"""
    system_prompt = f"""
    You are a decision-making component. 
    Provide decisions in the following JSON format:
    {decision_schema}
    Only respond with valid JSON.
    """
    
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=prompt,
        temperature=0.2  # Lower temperature for more consistent decisions
    )
    
    try:
        decision = json.loads(response)
        return decision
    except json.JSONDecodeError:
        return None
```

### 1.2 Decision Confidence Scoring
Evaluating decision reliability:

```python
class DecisionEvaluator:
    def __init__(self, llm, validation_rules):
        self.llm = llm
        self.rules = validation_rules
    
    def score_confidence(self, decision, context):
        scores = []
        for rule in self.rules:
            score = rule.evaluate(decision, context)
            scores.append(score)
        
        confidence = sum(scores) / len(scores)
        return {
            'confidence': confidence,
            'rule_scores': dict(zip(self.rules, scores))
        }
```

## 2. Routing and Workflow Control

### 2.1 Intelligent Router Implementation
Managing decision flow:

```python
class LLMRouter:
    def __init__(self, llm, routes, fallback_handler):
        self.llm = llm
        self.routes = routes
        self.fallback = fallback_handler
    
    def route_request(self, request):
        # Generate routing decision
        decision = self.get_routing_decision(request)
        
        # Validate decision
        if self.validate_route(decision):
            handler = self.routes.get(decision['route'])
            return handler(request)
        
        return self.fallback(request)
    
    def get_routing_decision(self, request):
        prompt = f"""
        Route the following request:
        {request}
        
        Available routes: {list(self.routes.keys())}
        """
        return get_structured_decision(self.llm, prompt, ROUTING_SCHEMA)
```

### 2.2 State Management
Tracking workflow progress:

```python
class WorkflowManager:
    def __init__(self, llm):
        self.llm = llm
        self.state = {}
        self.history = []
    
    def update_state(self, decision, outcome):
        self.state.update({
            'last_decision': decision,
            'last_outcome': outcome,
            'timestamp': datetime.now()
        })
        self.history.append(self.state.copy())
    
    def get_next_action(self, context):
        state_context = f"""
        Current state: {self.state}
        Decision history: {self.history[-5:]}  # Last 5 decisions
        New context: {context}
        """
        return get_structured_decision(self.llm, state_context, ACTION_SCHEMA)
```

## 3. Agentic Workflows

### 3.1 Agent Implementation
Building autonomous agents:

```python
class LLMAgent:
    def __init__(self, llm, tools, goal):
        self.llm = llm
        self.tools = tools
        self.goal = goal
        self.memory = AgentMemory()
    
    def plan(self, context):
        prompt = f"""
        Goal: {self.goal}
        Context: {context}
        Available tools: {self.tools.descriptions}
        Memory: {self.memory.summarize()}
        
        Create a plan to achieve the goal.
        """
        return get_structured_decision(self.llm, prompt, PLAN_SCHEMA)
    
    def execute_step(self, plan_step):
        tool = self.tools.get(plan_step['tool'])
        result = tool.execute(plan_step['parameters'])
        self.memory.add(plan_step, result)
        return result
```

### 3.2 Tool Integration
Managing available actions:

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.descriptions = {}
    
    def register(self, name, tool, description):
        self.tools[name] = tool
        self.descriptions[name] = description
    
    def get(self, name):
        return self.tools.get(name)
    
    def list_available(self):
        return list(self.tools.keys())
```

## 4. Safety and Reliability

### 4.1 Decision Validation
Ensuring safe decisions:

```python
class DecisionValidator:
    def __init__(self, rules, fallback_decision):
        self.rules = rules
        self.fallback = fallback_decision
    
    def validate(self, decision, context):
        for rule in self.rules:
            if not rule.check(decision, context):
                return False, rule.description
        return True, "All checks passed"
    
    def get_safe_decision(self, decision, context):
        is_valid, reason = self.validate(decision, context)
        if is_valid:
            return decision
        return self.fallback
```

### 4.2 Error Recovery
Handling decision failures:

```python
class ErrorHandler:
    def __init__(self, llm, recovery_strategies):
        self.llm = llm
        self.strategies = recovery_strategies
    
    def handle_error(self, error, context, previous_decision):
        prompt = f"""
        Error occurred: {error}
        Context: {context}
        Previous decision: {previous_decision}
        
        Recommend recovery strategy.
        """
        strategy = get_structured_decision(self.llm, prompt, RECOVERY_SCHEMA)
        return self.strategies.get(strategy['name'])(error, context)
```

## 5. Monitoring and Logging

### 5.1 Decision Logging
Tracking system behavior:

```python
class DecisionLogger:
    def __init__(self, storage):
        self.storage = storage
    
    def log_decision(self, decision, context, outcome):
        entry = {
            'timestamp': datetime.now(),
            'decision': decision,
            'context': context,
            'outcome': outcome,
            'metadata': self.extract_metadata(decision, context)
        }
        self.storage.save(entry)
    
    def analyze_decisions(self, time_range):
        decisions = self.storage.query(time_range)
        return self.compute_metrics(decisions)
```

### 5.2 Performance Monitoring
Evaluating system effectiveness:

```python
class PerformanceMonitor:
    def __init__(self, metrics_registry):
        self.metrics = metrics_registry
        self.thresholds = {}
    
    def set_threshold(self, metric_name, threshold):
        self.thresholds[metric_name] = threshold
    
    def check_performance(self, recent_decisions):
        results = {}
        for metric in self.metrics:
            value = metric.compute(recent_decisions)
            threshold = self.thresholds.get(metric.name)
            results[metric.name] = {
                'value': value,
                'status': 'ok' if threshold is None or value >= threshold else 'alert'
            }
        return results
```

## Summary
Building LLM-based decision systems requires:
1. Structured decision generation
2. Robust validation and safety checks
3. State and workflow management
4. Error handling and recovery
5. Comprehensive monitoring

These components enable reliable autonomous systems while maintaining safety and control. 