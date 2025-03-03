# Advanced Neural Network Architectures and Applications

## 1. Modern Architecture Components

### 1.1 Attention Mechanisms
The foundation of modern architectures:

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """Basic attention mechanism"""
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale matmul_qk
    dk = query.size()[-1]
    scaled_attention_logits = matmul_qk / math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### 1.2 Multi-Head Attention
Parallel attention processing:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
```

## 2. Advanced Training Techniques

### 2.1 Mixed Precision Training
Efficient computation with reduced precision:

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, dataloader, optimizer):
    scaler = GradScaler()
    
    for data, target in dataloader:
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 2.2 Gradient Accumulation
Training with limited memory:

```python
def train_with_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.zero_grad()
    for i, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
```

## 3. Advanced Optimization

### 3.1 Learning Rate Scheduling
Sophisticated learning rate control:

```python
def create_scheduler(optimizer):
    # Warmup followed by cosine decay
    def warmup_cosine_schedule(step):
        warmup_steps = 1000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
```

### 3.2 Advanced Optimizers
Modern optimization techniques:

```python
class AdaFactor(torch.optim.Optimizer):
    """Simplified AdaFactor implementation"""
    def __init__(self, params, lr=None, beta1=0.9, eps=1e-30,
                 clip_threshold=1.0, decay_rate=-0.8, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, eps=eps, clip_threshold=clip_threshold,
                       decay_rate=decay_rate, weight_decay=weight_decay)
        super().__init__(params, defaults)
```

## 4. Model Parallelism

### 4.1 Data Parallel Training
Scaling across multiple GPUs:

```python
def setup_parallel_training(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    return model
```

### 4.2 Distributed Training
Multi-node training setup:

```python
def setup_distributed_training(model, world_size):
    dist.init_process_group("nccl")
    local_rank = dist.get_rank()
    
    model = model.to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    
    return model
```

## 5. Advanced Architectures

### 5.1 Transformer Variants
Modern architecture implementations:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
```

### 5.2 Vision Transformers
Image processing with transformers:

```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, d_model):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = TransformerEncoder(d_model)
        self.head = nn.Linear(d_model, num_classes)
```

## 6. Production Considerations

### 6.1 Model Quantization
Reducing model size:

```python
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model
```

### 6.2 Model Pruning
Removing unnecessary weights:

```python
def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
```

## 7. Monitoring and Debugging

### 7.1 Advanced Logging
Comprehensive training monitoring:

```python
class TrainingMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def log(self, metrics):
        for k, v in metrics.items():
            self.metrics[k].append(v)
    
    def plot_metrics(self):
        for name, values in self.metrics.items():
            plt.figure(figsize=(10, 5))
            plt.plot(values)
            plt.title(name)
            plt.show()
```

### 7.2 Gradient Flow Analysis
Debugging training dynamics:

```python
def analyze_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append({
                'name': name,
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'norm': param.grad.norm().item()
            })
    return gradients
```

## Summary
Advanced neural networks require:
1. Understanding modern architecture components
2. Implementing efficient training techniques
3. Managing computational resources
4. Monitoring and debugging complex systems
5. Considering production deployment requirements

These concepts form the foundation for building and deploying state-of-the-art deep learning systems. 