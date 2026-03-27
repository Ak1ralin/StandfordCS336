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
## 9. Training of MoE

Problem : need sparsity for training-time efficiency but sparse gating decisions are not differentiable -> no gradient, cant train router

- Solution :
    - Reinforcement Learning : router = policy, expert selecting = action, loss = -reward
        - Huge Variance -> training instability
    - Stochastic perturbations
        - Stochastic = ramdom -> router logits + **noise**
        - So router might explore different expert
    - Heuristic balancing losses
        - Load Balancing -> make every expert have approximate number of token

### Load Balancing Problem

Without regulation : router may choose the same expert for every token (positive feedback loop)

Result : `expert collapse`

Solution : add auxiliary loss that encourages equal expert usage (every expert expect training with $\frac{batch \times len_{seq}}{num_{expert}}$)

Effect: overused expert → penalty

Trade-Off : load-balancing sacrifices some routing optimality for training stability

#### Standard Auxiliary loss (Per-expert balancing) : N experts, batch $\Beta$ with T tokens
$$L_{total} = L_{task} + L_{aux}$$
$$L_{aux} = \alpha\cdot N \cdot \sum^N_{i=1}f_i\cdot P_i $$
$f_i$ is the fraction of tokens dispatched to expert i
$$
f_i = \frac{1}{T}\sum_{x\in \Beta} 1 * argmax( p(x) = i) 
$$
$P_i$ is the average probability allocated for expert i
$$
P_i = \frac{1}{T}\sum_{x \in \Beta}p_i(x)
$$

#### Per device balancing 
- just because if expert didnot equally distributed on device, some device might become bottleneck if using Per-device balancing, so we just change from i = expert to i = device

$f_i$ is the fraction of tokens dispatched to device i

$P_i$ is the average probability allocated for device i

Some papers found per-device balancing is important than per-expert balancing 

#### Auxiliary Loss free Balancing
- standard auxiliary loss : indirectly balanced by loss gradient 
- Problem : $L_{task}$ wanna choose best expert (optimal route), but $L_{aux}$ wanna balance token, **gradient interference** hard to choose $\alpha$
- Solution from DSv3 : Add a bias $b_i$ on routing score 
$$s_i \rightarrow s_i + b_i$$
routing from topk($s_i$) to topk($s_i + b_i$)

for every step $b_i$ will update based on load, feedback control
$$
b_{i,t} \leftarrow b_{i,t-1} + \eta(load_{target} - load_{actual}) 
$$

Advantage : no gradient interference , more stable

------------------------------------------------------------------------

## 10. Systems Advantage

MoE enables new parallelism.

Observation: each expert is independent, so we can place experts on different GPUs (less inter-device communication). This creates **"Expert parallelism"** which scales very well to large clusters.

Moreover on same device, different expert can do batch matmul (**MegaBlocks**)
- W1@x1 , W2@x2, W3@x3 = 3 matmuls
$$ W = \begin{bmatrix}
W_1 & 0 & 0 \\
0 & W_2 & 0 \\
0 & 0 & W_3
\end{bmatrix} $$

$$X = concat(x1,x2,x3)$$

$$ R = W@X$$

GPU kernel will only compute non-zero block, so it's fine

------------------------------------------------------------------------
## 11. MoE Inference stochasticity
- every expert have a limit on token processing in that batch, so sometimes if 1 expert have to process too much token -> it will drop some tokens off (because its sequencetial if we wait, will have stall which is waste of power), and the result will be 0, so this is the reason why 0 temperature which should be deterministic sometimes give different result -> sometimes drop, sometimes not

Solved by
- if any of expert is full -> route to second highest instead
- increase limit 

------------------------------------------------------------------------

## 12. Training Stability Issues

Like Z-loss, softmax is unstable based on the gradient of softmax, backprop will definitely increase correct logit -> logit drift, and because softmax is exponential function, so when hit a red-line will become softmax saturation

The solution : z-loss for router softmax 
    - prevent logit explosiion

### MoE Fine-tuning easier overfitting

Just because data for training distribution might not uniform, and during fine-tuning mostly will disbale balancing so some expert when it meet small sft set, they can easily memorize all of this -> overfitting

------------------------------------------------------------------------

## 13. Upcycling

Interesting technique : convert dense model → MoE

- Steps:
    - initialize experts from dense FFN , for finegrained -> partition dense FFN (its actually just mapping to higher feature space, so partition is like partition feature space and router is choosing which feature space is propriated)
    - duplicate weights
    - continue training

Benefit : reuse pretrained models

------------------------------------------------------------------------

## 14. DeepSeek MoE Evolution

### V1
    - 2 shared + 64 fine-grained expert
    - standard top-6 routing
    - aux-loss balancing (expert only)
### V2
    - 2 shared + 160 fine-grained expoert
    - top-M device routing (choose device -> choose top-6 expert with device)
    - aux-loss balancing (expert + communication(device) balancing loss)

### V3
    - 1 shared + 256 fine-grained expert
    - top-M device routing (choose device -> choose top-8 expert with device)
        - sigmoid top-k routing (use sigmoid instead of softmax)
            - softmax is competitive, sigmoid is independent among expert
    - seq-wise aux-loss balancing
        - within 1 sequence, token should distributed, enhance effective capacity 

------------------------------------------------------------------------

## 15. MLA (Multi‑Head Latent Attention)

Idea : compress KV representations 

normal : $h@W_K, h@W_V$

MLA : $c = W@h , c@W_k, c@W_v$

Benefits: smaller KV cache (only save c)

## 16. MTP (Multi-Token Prediction)
Have small, lightweight models that predict multiple steps ahead

------------------------------------------------------------------------

# One‑Line Summary

MoE makes LLMs larger by **activating only a small subset of specialized
networks per token**, allowing extreme parameter scaling without
increasing compute.
