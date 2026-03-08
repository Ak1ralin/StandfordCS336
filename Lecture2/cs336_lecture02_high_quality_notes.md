# CS336 Lecture 02 --- Training Infrastructure Notes

This document restructures the lecture into the **actual conceptual
layers used in modern ML training systems**.

Focus: - how models consume **memory** - how they consume **compute** -
how GPUs are utilized efficiently - how the full **training pipeline**
works

------------------------------------------------------------------------

# 1. Training Resources

Training large models consumes two fundamental resources:

## Memory

Memory stores:

-   model parameters
-   gradients
-   optimizer state
-   activations (intermediate layer outputs)
-   input data

Total memory roughly:

    Memory ≈ parameters + activations + gradients + optimizer_state

If using `float32`:

    1 value = 4 bytes

Example:

    tensor size = 16 × 32
    elements = 512
    memory = 512 × 4 bytes = 2048 bytes

------------------------------------------------------------------------

## Compute

Compute measures **how many mathematical operations are performed**.

Unit:

    FLOP = Floating Point Operation

Examples:

-   addition
-   multiplication

Hardware speed:

    FLOP/s = FLOPs per second

Example GPU:

    NVIDIA A100 ≈ 3.12e14 FLOP/s

------------------------------------------------------------------------

# 2. Memory Model of Training

Most tensors use floating point numbers.

## Floating Point Formats

  Type       Bytes   Properties
  ---------- ------- ---------------------------------
  float32    4       stable, default
  float16    2       smaller memory but narrow range
  bfloat16   2       same range as float32
  fp8        1       newest training format

Tradeoff:

    Higher precision → stable but expensive
    Lower precision → cheaper but unstable

Modern training uses:

    mixed precision training

------------------------------------------------------------------------

## Mixed Precision

Idea:

| Component \| Precision \|

\|------\|------\| Forward activations \| bf16 / fp16 \| Parameters \|
float32 \| Gradients \| float32 \|

Goal:

-   reduce memory
-   increase GPU throughput
-   maintain numerical stability

------------------------------------------------------------------------

# 3. Tensor Storage Model

A tensor is essentially:

    (pointer to memory) + (shape metadata) + (stride metadata)

Stride defines **how to move in memory to reach the next element**.

Example:

Matrix

    [1 2 3
     4 5 6]

Stride:

    row stride = 3
    column stride = 1

------------------------------------------------------------------------

## Views vs Copies

### Views

Operations that **reuse the same memory**:

-   slicing
-   transpose
-   view

Cost:

    almost zero

But modifying one tensor modifies the other.

------------------------------------------------------------------------

### Copies

Operations that allocate new memory:

-   arithmetic operations
-   some reshapes

Cost:

    extra memory + extra compute

Efficient ML code minimizes copies.

------------------------------------------------------------------------

# 4. Core Tensor Operations

## Elementwise Operations

Operate on each value independently.

Examples:

    x + y
    x * 2
    sqrt(x)

Compute cost:

    O(n)

------------------------------------------------------------------------

## Aggregation Operations

Reduce many values into one.

Examples:

    mean()
    var()
    sum()

------------------------------------------------------------------------

## Batching

Deep learning heavily relies on batch processing.

Instead of computing:

    one sample at a time

we compute:

    many samples simultaneously

Operations:

    stack → add new dimension
    cat → concatenate tensors

------------------------------------------------------------------------

# 5. Matrix Multiplication (The Core of Deep Learning)

Almost all deep learning compute comes from:

    matrix multiplication

Example:

    y = x @ w

If

    x : B × D
    w : D × K

then compute cost:

    FLOPs = 2 × B × D × K

Explanation:

Each output element requires:

    D multiplications
    D additions

Matrix multiplication dominates compute in:

-   Transformers
-   CNNs
-   MLPs

------------------------------------------------------------------------

# 6. FLOPs Accounting for Model Training

For a model with:

    #tokens = N
    #parameters = P

Approximate compute:

Forward pass:

    2 × N × P FLOPs

Backward pass:

    4 × N × P FLOPs

Total training cost:

    6 × N × P FLOPs

Backward pass is about **2× the forward cost**.

------------------------------------------------------------------------

# 7. GPU Utilization

GPU performance is measured using:

    Model FLOPs Utilization (MFU)

Definition:

    MFU = actual FLOP/s ÷ theoretical FLOP/s

Typical good training efficiency:

    MFU ≥ 0.5

Low MFU usually indicates:

-   small batch sizes
-   CPU bottlenecks
-   memory bandwidth limits

------------------------------------------------------------------------

# 8. Gradients and Backpropagation

Training requires computing:

    gradient = d(loss) / d(parameter)

PyTorch performs this automatically using:

    automatic differentiation

Example:

    loss.backward()

This computes gradients for all parameters in the graph.

------------------------------------------------------------------------

# 9. Model Parameters

In PyTorch parameters are stored as:

    nn.Parameter

Example:

    weight = nn.Parameter(torch.randn(input_dim, output_dim))

------------------------------------------------------------------------

## Weight Initialization

Bad initialization can cause:

-   exploding gradients
-   vanishing gradients

A common fix:

    weight / sqrt(input_dim)

This keeps activations stable across layers.

This idea is known as:

    Xavier initialization

------------------------------------------------------------------------

# 10. Embedding Layers

An embedding maps token IDs to vectors.

Example:

    embedding = nn.Embedding(vocab_size, hidden_dim)

Input:

    (B, L)

Output:

    (B, L, H)

Where:

    B = batch size
    L = sequence length
    H = hidden dimension

Embeddings are the first stage of language models.

------------------------------------------------------------------------

# 11. Data Pipeline

Training data for language models is typically stored as:

    integer token sequences

Large datasets can be terabytes.

Loading everything into RAM is impossible.

Solution:

    numpy.memmap

This loads only the required segments of the dataset.

------------------------------------------------------------------------

## Batch Sampling

Training samples are produced by:

    randomly sampling sequence windows

Example:

    data[start : start + sequence_length]

------------------------------------------------------------------------

# 12. Optimizers

Optimizers update parameters using gradients.

Basic gradient descent:

    θ = θ − lr × gradient

Where:

    lr = learning rate

------------------------------------------------------------------------

## AdaGrad

AdaGrad adapts learning rate per parameter.

Idea:

    accumulate squared gradients

Update rule:

    θ = θ − lr * g / sqrt(sum(g²))

Optimizer evolution:

    SGD → AdaGrad → RMSProp → Adam

------------------------------------------------------------------------

# 13. Training Loop

Typical training loop:

    for step in training:

        batch = sample_data()

        predictions = model(batch)

        loss = loss_fn(predictions, targets)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

Steps:

1.  sample batch
2.  forward pass
3.  compute loss
4.  backward pass
5.  update parameters

------------------------------------------------------------------------

# 14. Randomness and Reproducibility

Randomness appears in:

-   parameter initialization
-   dropout
-   data ordering

To make experiments reproducible set seeds:

    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

------------------------------------------------------------------------

# 15. Checkpointing

Training large models can run for:

    days → weeks → months

Failures are inevitable.

Therefore training periodically saves:

    model parameters
    optimizer state

Example:

    torch.save(checkpoint)

This allows training to resume.

------------------------------------------------------------------------

# 16. Big Picture of ML Training Systems

Training a large model involves coordinating:

1.  **Memory**
2.  **Compute**
3.  **GPU utilization**
4.  **Data pipeline**
5.  **Optimization algorithm**
6.  **Training infrastructure**

The majority of compute in modern AI systems comes from:

    matrix multiplication

Efficient training is largely about:

    maximizing GPU utilization
    while minimizing memory usage
