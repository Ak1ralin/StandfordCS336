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

## 3. Why This Works

Old FFN =  every token update same FFN, FFN must compromise, its actually lead to **gradient interference**.

Chain:

    Different tokens require different knowledge
    ↓
    A single FFN must learn everything (specialized-A knowledge got distract by others specialized-B knowledge)
    ↓
    Capacity is shared (capacity ~ function : for old FFN, capacity must shared it self to both token-A and token-B)
    ↓
    Instead specialize networks (specialized capacity)

Router decides which expert processes a token.

Result: Specialization + larger total capacity

------------------------------------------------------------------------

## 4. Where MoE Is Used

Most models modify **only the FFN layer**.

Transformer block:

    Attention
    ↓
    MoE (replaces FFN)
    ↓
    Residual

Reason: FFN contains most parameters, So replacing FFN gives **maximum scaling benefit**.

FYI : MHA of Attention is something looklike MoE is some senses, because it also contribute to specialized, but because overall params didnot increase so capacity also remain stable, and every token use all head so we could called it **"dense specialization"**, while MoE is sparse specialization.
    - MHA factorized representation, while MoE increase capacity
    - key reason attention not use MoE is because **Attention couples tokens together**
    - multiple attention subspaces ～ structured big matrix

### Why MoE training is more efficient
- Capacity increase (function represent easier/accutate), loss decrease easier
- Less gradient interference
- Divice fit / expert parallelism : each FFN can fit in a device
    - But MoE mostly is more complex so single step latency is slower

### Why haven't MoE been more popular?
- Infrastructure is complex, and required more devices
    - Will be great if you have multiple devices.
    - And can separate data to each of device to make expert parallelism, else its will lead to inter-device communication which is expensive.
- Training objective are somewhat heuristic.
    - because router selection is discrete but training need gradient, so its hard to train router, in the past most lead to **expert collapse** until **auxiliary load balancing loss**
------------------------------------------------------------------------


## 5. Routing (Most Important Part)

Router decides:

    which expert processes each token

Typical method:
- Top‑K routing
    - Token chooses expert (**heuristic, best option**)
    - Expert chooses token (token might not be pick)
    - Global routing (bipartite matching, total highest, but mostly solve with Hugarian algorithm too much computing) but actually its intuitively best routing algorithm.
- common baseline : hash routing -> `token_embedding % num_expert = rout_expert`, input space partition so naturally specialized, but it fixed so its normal that it will be bad than learnable like top-k
- Others sol :
    - RL routing : because top-k is discrete, so no gradient, so hard to use back-prop for training, so some try to use RL instead, and use loss as reward model to optimized router(policy)
        - Problem : gradient variance is extremely big, training unstable 

Process (token choose expert):

    1. compute score for each expert
    2. choose K largest
    3. send token to those experts
    4. combine outputs

------------------------------------------------------------------------

## 6. How Top-K work 
$$
s_{i,t} = h_t^T \cdot e_i
$$
$$
g_{i,t} = \left\{\begin{array}{rcl}
s_{i,t} : s_{i,t} \in TopK
\\
0 : otherwise  
\end{array}\right.
$$
$$
g_{i,t} = softmax(g_{i,t})
$$
$$
h_t = \sum_{i=1}^{N_{expert}} (g_{i,t}FFN_i(h_t)) + h_t
$$

------------------------------------------------------------------------

## 7. Fine‑Grained Experts

MoE each expert FFN $d_{ff}$ decrease so hidden embedding is smaller

Old design: `few large experts` 

New design: `many small experts` 

Reason:
    smaller experts → more specialized (function decomposition), less gradient interference, higher specialized resolution

------------------------------------------------------------------------

## 8. Shared Experts

Problem : Missroute (router might wrong) and some specialized is general needed for every tokens.

Solution: shared experts

These experts always run to prevent missroute, provide general backbone (base knowledge), help training stability

Architecture: shared experts (weight = 1) + routed experts
$$
h_t = \sum_{i \in R} (g_{i,t}FFN_i(h_t))+ \sum_{j \in S} (g_{j,t}FFN_j(h_t)) +  + h_t
$$

- FYI : OIMoE points out that MoE does not significantly improve performance.
    - nice to have, not fundamental

------------------------------------------------------------------------

## 9. Load Balancing Problem

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
