# CS336 --- Mixture of Experts (MoE) 


------------------------------------------------------------------------

## Problem

LLM scaling law observation: More parameters → better performance

But compute cost grows with parameters:

    Dense Transformer mean more
    params ↑ → FLOPs ↑ → training cost explodes

So the question becomes: **Can we increase parameters without increasing compute**?

------------------------------------------------------------------------

## Idea of MoE

MoE answer:

    Exist many networks (FFN)
    but only activate a few 

Structure:

    Token
     ↓
    Router
     ↓
    Top‑k Experts
     ↓
    Output

Key property:

    Total parameters  >> Active parameters

Example:

    Total parameters = 600B
    Active parameters = 30B

Compute depends only on **active parameters**, not total parameters.

------------------------------------------------------------------------

# 3. Why This Works

Chain:

    Different tokens require different knowledge
    ↓
    A single FFN must learn everything (specialized-A knowledge got distract by others specialized-B knowledge)
    ↓
    Capacity is wasted
    ↓
    Instead specialize networks

MoE creates:

    Expert 1 → math
    Expert 2 → code
    Expert 3 → reasoning
    Expert 4 → language
    ...

Router decides which expert processes a token.

Result:

    Specialization + larger total capacity

------------------------------------------------------------------------

# 4. Where MoE Is Used

Most models modify **only the FFN layer**.

Transformer block:

    Attention
    ↓
    MoE (replaces FFN)
    ↓
    Residual

Reason:

    FFN contains most parameters

So replacing FFN gives **maximum scaling benefit**.

------------------------------------------------------------------------

# 5. Routing (Most Important Part)

Router decides:

    which expert processes each token

Typical method:

    Top‑K routing

Process:

    1. compute score for each expert
    2. choose K largest
    3. send token to those experts
    4. combine outputs

Typical K values:

    Switch Transformer → k=1
    Mixtral → k=2
    DBRX → k=4
    DeepSeek → k≈6–8

------------------------------------------------------------------------

# 6. Why Routing Is Hard

Problem:

    Top‑K selection is discrete

Discrete operations have:

    no gradient

Which means:

    backpropagation can't train it directly

Solutions explored:

    RL routing
    stochastic routing
    heuristic balancing losses

In practice:

    heuristic balancing losses win

Because they are simple and stable.

------------------------------------------------------------------------

# 7. Load Balancing Problem

Without regulation:

    router may choose the same expert for every token

Result:

    expert collapse

Fix:

    add auxiliary loss

that encourages

    equal expert usage

Effect:

    overused expert → penalty

------------------------------------------------------------------------

# 8. Fine‑Grained Experts

Recent models changed architecture.

Old design:

    few large experts

New design:

    many small experts

Example:

    DeepSeek V3
    256 experts
    top‑8 active

Reason:

    smaller experts → better specialization

------------------------------------------------------------------------

# 9. Shared Experts

Problem:

Some tokens need **general knowledge**.

Solution:

    shared experts

These experts:

    always run

Architecture:

    shared experts + routed experts

Used by:

    DeepSeek
    Qwen

------------------------------------------------------------------------

# 10. Systems Advantage

MoE enables new parallelism.

Observation:

    each expert is independent

So we can place experts on different GPUs.

This creates:

    Expert parallelism

Which scales very well to large clusters.

------------------------------------------------------------------------

# 11. Systems Difficulty

MoE introduces heavy communication.

Because:

    tokens must be routed across devices

So we must perform:

    token dispatch
    expert execution
    token gather

Efficient implementations use:

    sparse matrix kernels

Example library:

    MegaBlocks

------------------------------------------------------------------------

# 12. Training Stability Issues

Routers are sensitive.

Problems:

    logit explosion
    imbalanced routing
    numerical instability

Common fixes:

    router in FP32
    z‑loss regularization
    routing jitter

------------------------------------------------------------------------

# 13. Upcycling

Interesting technique:

    convert dense model → MoE

Steps:

    1 initialize experts from dense FFN
    2 duplicate weights
    3 continue training

Benefit:

    reuse pretrained models

Example:

    Qwen MoE

------------------------------------------------------------------------

# 14. DeepSeek MoE Evolution

### V1

    16B total
    2.8B active
    shared + routed experts

### V2

    236B total
    21B active
    improved communication balancing

### V3

    671B total
    37B active
    256 experts
    top‑8 routing

------------------------------------------------------------------------

# 15. Extra Techniques in DeepSeek V3

## MLA (Multi‑Head Latent Attention)

Idea:

    compress KV representations

Benefits:

    smaller KV cache
    lower memory

------------------------------------------------------------------------

## MTP (Multi‑Token Prediction)

Train model to predict:

    next token
    future token

Goal:

    better training efficiency

------------------------------------------------------------------------

# 16. Final Intuition

MoE scaling logic:

    Dense scaling
    params ↑ → compute ↑

MoE scaling:

    params ↑
    active params constant
    compute constant

So MoE gives:

    massive capacity
    nearly same compute

------------------------------------------------------------------------

# One‑Line Summary

MoE makes LLMs larger by **activating only a small subset of specialized
networks per token**, allowing extreme parameter scaling without
increasing compute.
