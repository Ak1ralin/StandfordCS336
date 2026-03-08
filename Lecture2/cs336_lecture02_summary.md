# CS336 Lecture 02 --- Training a Model (Summary)

## 1. Goal of the Lecture

This lecture explains the **core components required to train a machine
learning model**, starting from the lowest-level objects and building
upward.

Bottom‑up structure: 1. Tensors 2. Tensor operations 3. Gradients 4.
Models 5. Optimizers 6. Training loop

Two critical resources when training models: - **Memory (GB)** --- how
much storage tensors require. - **Compute (FLOPs)** --- how many
mathematical operations are performed.

------------------------------------------------------------------------

# 2. Tensors (Basic Building Block)

A **tensor** is a multi‑dimensional array used to store: - model
parameters - gradients - training data - intermediate activations -
optimizer state

Examples: - vector → 1D tensor - matrix → 2D tensor - batches → higher
dimensions

Common tensor creation: - `zeros()` - `ones()` - `randn()` - `empty()`

Important: initialization methods like **Kaiming initialization** are
used to set good starting weights.

------------------------------------------------------------------------

# 3. Memory Accounting

Memory usage depends on:

    memory = number_of_elements × bytes_per_element

Example:

    16 × 32 float32 tensor
    = 512 elements
    = 512 × 4 bytes
    = 2048 bytes

### Floating Point Formats

  Type       Bytes   Notes
  ---------- ------- -------------------------------------------
  float32    4       default
  float16    2       less memory but smaller numeric range
  bfloat16   2       same range as float32 but lower precision
  fp8        1       newest format for deep learning

Tradeoff:

-   Higher precision → stable but expensive
-   Lower precision → faster and cheaper but risk instability

Solution: **mixed precision training**.

------------------------------------------------------------------------

# 4. GPU Usage

Tensors start on the **CPU**.

To run fast training they must be moved to **GPU memory**.

Example:

    x.to("cuda:0")

GPU benefits: - massive parallel computation - optimized matrix
operations

But GPU memory is limited.

------------------------------------------------------------------------

# 5. Tensor Operations

Tensor operations fall into several categories.

### Views (No Copy)

Operations that **reuse the same memory**: - slicing - transpose - view

These are cheap because no new memory is allocated.

### Copies

Some operations create new tensors: - reshape (sometimes) - arithmetic
operations

These require both memory and compute.

------------------------------------------------------------------------

# 6. Important Tensor Operations

### Elementwise Operations

Apply operation to each element:

Examples:

    x + x
    x * 2
    sqrt(x)

Cost:

    O(n)

### Aggregation

Reduce values:

Examples:

    mean()
    var()

### Batching

Deep learning works best with **batches**.

Operations:

-   `stack` → adds a new dimension
-   `cat` → concatenates tensors

### Matrix Multiplication

Most important operation in deep learning.

Example:

    y = x @ w

Cost:

    FLOPs = 2 × m × n × p

This dominates compute in neural networks.

------------------------------------------------------------------------

# 7. FLOPs (Compute Accounting)

**FLOP = Floating Point Operation**

Examples: - addition - multiplication

Large models require enormous compute:

Example estimates:

  Model   FLOPs
  ------- --------
  GPT‑3   \~3e23
  GPT‑4   \~2e25

Hardware speed is measured in:

    FLOP/s

Example GPU:

-   NVIDIA A100 ≈ 312 teraFLOP/s

------------------------------------------------------------------------

# 8. Model FLOPs Utilization (MFU)

MFU measures efficiency:

    MFU = actual FLOP/s ÷ theoretical FLOP/s

Typical good training efficiency:

    MFU ≥ 0.5

Low MFU indicates poor GPU utilization.

------------------------------------------------------------------------

# 9. Gradients (Backpropagation)

Training requires computing **gradients**.

Gradient definition:

    gradient = derivative of loss w.r.t parameters

PyTorch uses **automatic differentiation**.

Example:

    loss.backward()

This computes gradients for all parameters.

------------------------------------------------------------------------

# 10. FLOPs for Training

For neural networks:

Forward pass:

    2 × (#data points) × (#parameters)

Backward pass:

    4 × (#data points) × (#parameters)

Total training compute:

    6 × (#data points) × (#parameters)

Backward pass is more expensive than forward.

------------------------------------------------------------------------

# 11. Model Parameters

Parameters are stored as:

    nn.Parameter

Weight initialization matters because large values can cause:

-   exploding gradients
-   unstable training

A common fix:

    weight / sqrt(input_dim)

This is similar to **Xavier initialization**.

------------------------------------------------------------------------

# 12. Embeddings

An **embedding** converts token IDs into vectors.

Example:

    nn.Embedding(vocab_size, hidden_dim)

Output shape:

    (batch_size, sequence_length, hidden_dim)

Embeddings are the first layer of most language models.

------------------------------------------------------------------------

# 13. Data Loading

Language model data is stored as **integer token sequences**.

Large datasets cannot be fully loaded into memory.

Solution:

    numpy.memmap

This loads only the required portions of the dataset.

------------------------------------------------------------------------

# 14. Optimizers

Optimizers update model parameters.

Basic rule:

    parameter = parameter − learning_rate × gradient

Examples:

### SGD

Simple gradient descent.

### AdaGrad

Adjusts learning rate using accumulated gradient squares.

Evolution of optimizers:

    SGD → AdaGrad → RMSProp → Adam

------------------------------------------------------------------------

# 15. Training Loop

Typical training loop:

1.  sample batch
2.  forward pass
3.  compute loss
4.  backward pass
5.  update parameters
6.  repeat

Pseudo‑structure:

    for step in training:
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

------------------------------------------------------------------------

# 16. Randomness and Reproducibility

Randomness occurs in:

-   parameter initialization
-   dropout
-   data shuffling

For reproducibility set seeds:

    torch.manual_seed()
    numpy.random.seed()
    random.seed()

------------------------------------------------------------------------

# 17. Checkpointing

Training large models can run for weeks.

Regularly save:

-   model parameters
-   optimizer state

Example:

    torch.save(checkpoint)

This allows training to resume after crashes.

------------------------------------------------------------------------

# 18. Mixed Precision Training

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
