# CS336 Lecture 02 --- Pytorch

## Goal of the Lecture

This lecture explains the **core components required to train a machine
learning model**, starting from the lowest-level objects and building
upward.

Bottom‑up structure: 
1. Tensors 
2. Tensor operations 
3. Gradients 
4. Models 
5. Optimizers 
6. Training loop

Two critical resources when training models: 
- **Memory (GB)** --- how much storage tensors require. 
- **Compute (FLOPs)** --- how many mathematical operations are performed.

Knowledge to take away:
- Mechanics : Pytorch
- Mindset : resource Accounting
- Intuition : broad stroke

------------------------------------------------------------------------
## Resource Accounting


### Tensors (Basic Building Block)

A **tensor** is a multi‑dimensional array used to store everything in deep-learning: 
- model parameters 
- gradients 
- training data 
- intermediate activations 
- optimizer state

Examples: 
- vector → 1D tensor 
- matrix → 2D tensor 
- batches → higher dimensions

Common tensor creation: 
```python
x = torch.tensor([[1.,2,3],[4,5,6]])
x = torch.zeros(4,8) # 4x8 matrix of all zeros 
x = torch.ones(4,8) # 4x8 matrix of all ones
x = torch.randn(4,8) # 4x8 matrix of iid Normal (0,1) samples
x = torch.empty(4,8) # allocate but dont initialized values
nn.init.trunc_normal_(x, mean=0, std = 1, a=-2,b=2) # custom logic to set the value
```

Important: initialization methods like **Kaiming initialization** are
used to set good starting weights.

------------------------------------------------------------------------

### Memory Accounting
Memory usage depends on:
    memory = number_of_elements × bytes_per_element

Example:

    16 × 32 float32 tensor
    = 512 elements
    = 512 × 4 bytes
    = 2048 bytes

$$value = (-1)^{sign} * (1+frac) * 2^{exp-127}$$
```
6.5 -> 6 + 0.5 -> b"110 + .1" = 110.1 = 1.101 * 2^2 
frac = .101, exp = 2+127(bias)
```
why `bias` -> to easily compare and represent negative exp

why `1+frac` cuz always 1.xxx , so make effective bit + 1 so implicit leading 1

`fraction bit = precision, while exponent bit = range` 

#### Floating Point Formats
  Type       |Bytes | Bit Allocate|Notes
  ---------- |-------| ------------|-------------------------------
  float32 (fp32)    |4  |1s,8e,23f|     default of pytorch
  float16 (fp16)    |2   |1s,5e,10f|    less memory but smaller numeric range
  bfloat16(bp16)  |2     |1s,8e,7f| same range as float32 but lower precision
  fp8        |1       |1s,4/5e,3/2f|newest format for deep learning

- notes : also have float64, but not use in deep-learning.

```python
xfp32 = torch.zeros(4,8) 
assert x.dtype == torch.float32 # its default of pytorch
xfp16 = torch.zeros(4,8,dtype=torch.float16) # 1e-8 ~ 0 in fp16 which is underflow
```

Tradeoff:
-   Higher precision → stable but expensive
-   Lower precision → faster and cheaper but risk instability

Solution: **mixed precision training**.
    - accumulate overtime ~ higher precision like params & optimizer use fp32
    - temporary use bp16

#### Total memory need
`4 * (num_params + num_activation + num_grad + num_optimizer_states)`
------------------------------------------------------------------------

### GPU Usage

Tensors start on the **CPU**.
```python
x = torch.zeros(32,32)
assert x.device == torch.device("cpu") # do something like this to makesure data in where we need
```

To run fast training they must be moved to **GPU memory**.
```python
if not torch.cuda.is_available(): # check do GPU existant
    return

num_gpus = torch.cuda.device_count() # number of gpus
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i) # getting properties of each GPU

memory_allocated = torch.cuda.memory_allocated() # memory used, expected 0

y = x.to("cuda:0") # move to first gpu
assert y.device == torch.device("cuda",0)
z = torch.zeros(32,32,device="cuda:0") # create in gpu directly
new_memory_allocated = torch.cuda.memory_allocated() # expected 2*(32*32*4) # y+z


```

GPU benefits: - massive parallel computation - optimized matrix operations
But GPU memory is limited.

------------------------------------------------------------------------

### Tensor Operations

Tensor operations fall into several categories. 

**PyTorch tensors are pointers into allocated memory 
... with metadata describing how to get to any element of the tensor.
```python
x = torch.tensor([ # 4x4 matrix 
    [0 ,1 ,2  ,3],
    [4 ,5 ,6  ,7],
    [8 ,9 ,10 ,11],
    [12,13,14 ,15],
])
# underlying storafe
[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# metadata
# stride[0] how to get next row = #col = 4
assert x.stride(0) == 4
# stride[1] how to get next col = 1
assert x.stride(1) == 1

# find an element with index  r,c = 1,2
index = r*x.stride(0) + c*stride(1) == 6
``` 

#### Views (No Copy, just another pointer with metadata)

Operations that **reuse the same memory** -> so mutations in one tensor affects the other. 
- slicing 
- transpose 
- view

These are cheap because no new memory is allocated.

metadata :
    - shape
    - stride
    - dtype
    - device

Contiguous/ Non-Contiguous
- Contiguous mean row-major transversal same way as how memory allocated
    - So GPU can coalesced memory access -> bulk read 
- Non-Contiguous mean not : transpose/permute will break contiguous
- **Some operation require tensor to be contiguous**
    - Sometime we use .contiguous() to reallocated memory (new memory allocated)

```python
x = torch.tensor([ # 4x4 matrix 
    [1 ,2  ,3],
    [4 ,5  ,6],
])

def same_storage(x : torch.Tensor, y : torch.Tensor):
    return x.untype_storage().data_ptr() == y.untype_storage().data_ptr()

# Get row 0
y = x[0] 
assert torch.equal(y, torch.tensor([1.,2,3]))
assert same_storage(x,y)
# Get col 1
y = x[:,1]
assert torch.equal(y, torch.tensor([2,5]))
assert same_storage(x,y)
# View
y = x.view(3,2) # from 2x3
assert torch.equal(y, torch.tensor([[1,2],[3,4],[5,6]]))
assert same_storage(x,y)
# Transpose
y = x.transpose(1,0)
# Mutate change both
x[0][0] = 100
assert y[0][0] == 100
# Check Contiguous
assert y.is_contiguous() # stride[i] = product(shape[i+1:])
```

------------------------------------------------------------------------

#### Copies

Some operations create new tensors: 
- reshape (sometimes) 
- arithmetic operations

These require both memory and compute.

------------------------------------------------------------------------

#### Elementwise Operations

Apply operation to each element and return new tensor of the same shape: 
    - so dont forget to define/inplace them 
```python
x = torch.tensor(y)
x.pow(2)
x.sqrt()
x.rsqrt()
x + x # x.add_(x) is in-place operation
x * 2
x / 0.5 
```

#### Aggregation

Reduce values:

Examples:

    mean()
    var()

#### Batching

Deep learning works best with **batches**.

Operations:

-   `stack` → adds a new dimension
-   `cat` → concatenates tensors

#### Matrix Multiplication

Most important operation in deep learning.

x.shape = (b,s,m,n) 

w.shape = (n,p)

its automatically iterate over values of the first 2 dimensions of x and multiply by y

Example:

    y = x @ w

Cost:

    FLOPs = 2 × m × n × p

This dominates compute in neural networks.
#### einops 
- A library for manipulating tensors where dimensions are named
------------------------------------------------------------------------

#### FLOPs (Compute Accounting)

**FLOP = Floating Point Operation**

Examples: 
- addition 
- multiplication

Large models require enormous compute:

Hardware speed is measured in:

    FLOP/s or FLOPS (Upper S)

**H100 only got 50% of peak performance for bf16/fp16 reported by spec without sparsity**

------------------------------------------------------------------------

#### Model FLOPs Utilization (MFU)

MFU measures efficiency:

    MFU = actual FLOP/s / promised FLOP/s

Typical good training efficiency:

    MFU ≥ 0.5

Low MFU indicates poor GPU utilization.

------------------------------------------------------------------------
#### Summary About Computing Accounting
- Matrix multiplications dominate : 2mnp FLOPs
- FLOP/s depends on **hardware** (H100>>A100) and **data-type** (bf16 >> fp32)
- MFU : actual FLOPs / promised FLOP/s

## Gradients (Backpropagation)

Training requires computing **gradients**.

Gradient definition:

    gradient = derivative of loss w.r.t parameters

PyTorch uses **automatic differentiation**.
```python
    x = torch.tensor([1.,2,3]) # input
    w = torch.tensor([1.,1,1],requires_grad = True) # want grad to update params
    pred_y = x @ w
    loss = 0.5 * (pred_y - label_y).pow(2) # MSE
    loss.backward() # dloss by dw
    w_grad = w.grad 
    w.grad = None # clear, else its will accumulate
```
This computes gradients for all parameters.

------------------------------------------------------------------------

## FLOPs for Training

- For neural networks: x @ w1 = h1, h1 @ w2 = y, y-label_y = loss
    - x = (B,D)
    - w1 = (D,D)
    - W2 = (D,K)

Forward pass: 
    `2 × (#data) × (#parameters) FLOPs`
- x @ w1 = `2 x B x (D x D)`
- h1 @ w2 = `2 x B x (D x K)`
- total = `2 x B x D(D+K)` 
- Number of data = B , not BD

Backward pass:
    `4 × (#data points) × (#parameters) FLOPs`

Total training compute:
    `6 × (#data points) × (#parameters)`

Backward pass is more expensive than forward.

------------------------------------------------------------------------
## Model Parameters

Parameters are stored as:
```python 
    w1 = nn.Parameter(torch.randn(input_dim,hidden_dim))
```
Weight initialization matters because large values can cause:
-   exploding gradients
-   unstable training

A common fix:
    `weight / sqrt(input_dim)`
```python
w1 = nn.Parameter(torch.randn(input_dim,hidden_dim)/np.sqrt(input_dim))
```
This is similar to **Xavier initialization**.

------------------------------------------------------------------------
## Embeddings

An **embedding** converts token IDs into vectors.

Example:

    nn.Embedding(vocab_size, hidden_dim)

Output shape:

    (batch_size, sequence_length, hidden_dim)

Embeddings are the first layer of most language models.

------------------------------------------------------------------------

## Data Loading

Language model data is stored as **integer token sequences**.

Large datasets cannot be fully loaded into memory.

Solution:
```python
    data = numpy.memmap("file_name",dtype=np.int32)
```
This loads only the required portions of the dataset.

------------------------------------------------------------------------

## Optimizers

Optimizers update model parameters.

Basic rule:

    parameter = parameter − learning_rate × gradient

Examples:

### SGD

Simple gradient descent.

### Momentum 

SGD + exponential averaging of grad

### AdaGrad

SGD + averaging by grad^2

### RMSProp

AdaGrad + exponentially averaging of grad^2

### Adam

RMSProp + momentum

------------------------------------------------------------------------

## Training Loop

Typical training loop:

1.  sample batch
2.  forward pass
3.  compute loss
4.  backward pass
5.  update parameters
6.  repeat

Pseudo‑structure:
```
    for step in training:
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
```

------------------------------------------------------------------------

## Randomness and Reproducibility

Randomness occurs in:
-   parameter initialization
-   dropout
-   data shuffling

For reproducibility set seeds:
```python
seed  = 0
torch.manual_seed(seed) 
numpy.random.seed(seed) # import numpy as np
random.seed(seed) # import random
```
------------------------------------------------------------------------

## Checkpointing

Training large models can run for weeks.

Regularly save:

-   model parameters `model.state_dict()`
-   optimizer state `optimizer.state_dict()`

Example:
```python
    torch.save(checkpoint, "model_checkpoint.pt")
```
This allows training to resume after crashes.

------------------------------------------------------------------------

## Mixed Precision Training

Use different numerical precision during training.

Strategy:

  Component             Precision
  --------------------- -----------------
  Forward activations   fp16 / bfloat16
  Parameters            float32
  Gradients             float32

Benefits:

-   lower memory usage
-   faster computation
-   minimal loss in accuracy

Modern frameworks provide **automatic mixed precision (AMP)**.

------------------------------------------------------------------------

# Key Takeaways

Training large neural networks requires careful accounting of:

1.  **Memory usage**
2.  **Compute cost (FLOPs)**
3.  **GPU efficiency**
4.  **Data pipeline**
5.  **Optimizer behavior**

Most compute in deep learning comes from:

    matrix multiplication
