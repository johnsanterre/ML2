# From Graph Theory to Deep Reinforcement Learning

## Overview
This session begins with the Ford-Fulkerson algorithm for network flow problems, then bridges these graph theory concepts to modern Deep Reinforcement Learning approaches. We'll see how understanding flow in networks provides intuition for value propagation in DRL.

## Learning Objectives
By the end of this session, students will:
- Understand max-flow min-cut theorem
- Implement Ford-Fulkerson algorithm
- Understand the relationship between graphs and state spaces
- Connect graph traversal to exploration strategies
- Recognize how value propagation relates to graph algorithms
- Apply graph concepts to DRL architectures

## Topics Covered

### 1. Network Flow Fundamentals
```python
class FlowNetwork:
    def __init__(self, graph):
        self.graph = graph  # Residual graph
        self.flow = {}     # Current flow
        
    def find_augmenting_path(self, source, sink, path):
        if source == sink:
            return path
        
        for node in self.graph[source]:
            residual = self.graph[source][node] - self.flow.get((source, node), 0)
            if residual > 0 and node not in path:
                result = self.find_augmenting_path(node, sink, path + [node])
                if result != None:
                    return result
        return None

    def ford_fulkerson(self, source, sink):
        max_flow = 0
        path = self.find_augmenting_path(source, sink, [source])
        
        while path != None:
            # Find minimum residual capacity along path
            flow = float('inf')
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                residual = self.graph[u][v] - self.flow.get((u, v), 0)
                flow = min(flow, residual)
            
            # Update flow
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                self.flow[(u, v)] = self.flow.get((u, v), 0) + flow
                self.flow[(v, u)] = self.flow.get((v, u), 0) - flow
            
            max_flow += flow
            path = self.find_augmenting_path(source, sink, [source])
        
        return max_flow
```

### 2. From Flow to Value Propagation
- State space as a graph
- Transitions as edges
- Flow capacity as reward bounds
- Path optimization problems

### 3. From Max-Flow to Value Iteration
```python
# Notice the similarity to Ford-Fulkerson
def value_iteration(states, actions, rewards, gamma=0.99):
    V = {state: 0 for state in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Like finding max flow, we find max value
            V[s] = max([rewards[s][a] + gamma * V[next_state] 
                       for a, next_state in actions[s].items()])
            delta = max(delta, abs(v - V[s]))
        if delta < 1e-3:
            break
    return V
```

### 4. Exploration Strategies
- Graph traversal algorithms
- Random walks
- Depth-first vs. Breadth-first
- ε-greedy as guided traversal

### 5. Deep Learning on Graphs
- Graph Neural Networks (GNN)
- Message passing
- Node embeddings
- Graph attention mechanisms

## Key Concepts

### Graph Structure in RL
- States as nodes
- Actions as edges
- Policies as path selection
- Value functions as node properties

### Modern Applications
- Graph-based policy networks
- Attention mechanisms over state spaces
- Message passing in multi-agent RL
- Graph embeddings for state representation

## Practical Examples

### 1. Navigation Problems
```python
class GraphEnvironment:
    def __init__(self, graph):
        self.graph = graph
        self.current_node = None
        
    def step(self, action):
        next_node = self.graph[self.current_node][action]
        reward = self.get_reward(next_node)
        return next_node, reward
```

### 2. Multi-Agent Systems
- Agent communication as graph edges
- Cooperative path finding
- Resource allocation networks
- Traffic flow optimization

## Bridging Concepts

### From Static to Dynamic
- Graph algorithms → Policy learning
- Fixed weights → Learned values
- Deterministic paths → Stochastic policies
- Single solution → Distribution over actions

### Modern Extensions
- Attention mechanisms
- Graph transformers
- Neural message passing
- Dynamic graph generation

## Looking Ahead
This foundation prepares us for:
- Advanced DRL architectures
- Multi-agent systems
- Graph-based world models
- Hierarchical reinforcement learning

## Key Takeaways
1. Graph theory provides fundamental structures for RL
2. Traditional algorithms inform modern approaches
3. Deep learning adds flexibility and learning capability
4. Graph-based thinking helps design better RL systems 