
# LLM Architecture — Research + Mental Model Summary
*(Designed for long‑term understanding + fast technical recall)*

---
# 1 The Core Mental Model of Transformers

The most useful way to think about a Transformer is:

**A residual stream with read/write modules.**

Residual stream update:

x_{l+1} = x_l + F(x_l)

Interpretation:

- The **residual stream** is the shared memory of the network.
- Modules (Attention / MLP) **read from the stream and write updates back**.
- The model gradually refines representations layer by layer.

Therefore a Transformer block is essentially:

Residual Memory
→ Read
→ Compute
→ Write back

This view comes from **mechanistic interpretability work**.

---
# 2 Transformer Block (Modern LLM Design)

Modern LLM block:

x → RMSNorm → Attention → + residual
  → RMSNorm → FFN → + residual

Important design principles:

1. Residual stream is the primary pathway
2. Normalize before transformation (PreNorm)
3. Preserve matrix‑multiply friendly operations
4. Prefer multiplicative interactions (GLU)
5. Prefer relative positional information

---
# 3 Why Pre‑Norm Won

Original Transformer used **Post‑Norm**:

(x + F(x)) → LayerNorm

Problems:
- gradient explosion
- unstable deep training

Modern LLMs use **Pre‑Norm**:

x + F(Norm(x))

Effect:

Better gradient propagation across layers.

Key idea:

Norm keeps the residual stream scale stable, allowing deep networks.

---
# 4 RMSNorm Simplification

LayerNorm:

y = (x − mean(x)) / sqrt(var(x)+ε) * γ + β

RMSNorm:

y = x / sqrt(mean(x²)+ε) * γ

Differences:

LayerNorm normalizes both **mean and variance**  
RMSNorm normalizes **only magnitude**.

Why this works:

Residual stream tends to stay centered already.

Benefit:

Less computation and fewer memory operations.

---
# 5 Removing Bias Parameters

Modern LLMs remove most bias terms.

Reason:

Residual streams allow constant shifts.

(x + c)W = xW + cW

Thus residual pathways already behave like implicit bias.

Normalization ensures the residual magnitude does not drift.

---
# 6 Feed‑Forward Networks (MLP Block)

Traditional FFN:

FF(x) = activation(xW1)W2

Modern LLMs use **gated activations**.

Example (SwiGLU):

FF(x) = (SiLU(xW1) ⊙ (xV))W2

Key idea:

Multiplicative gating increases expressivity.

It allows:

f(x) = g(x) ⊙ h(x)

which is more expressive than single activations.

---
# 7 Multi‑Head Attention Interpretation

Attention computes:

s_ij = q_i k_j^T

Interpretation:

Attention measures **similarity between tokens**.

Multi‑head attention splits the representation space.

Typical rule:

d_head × n_heads ≈ d_model

Common head dimension:

64 – 128

Each head learns different interaction patterns.

---
# 8 Positional Encoding Problem

Attention alone is permutation invariant.

Without positions:

The model cannot distinguish word order.

Attention score:

s_ij = q_i k_j^T

If embeddings contain no positional signal, sequence order disappears.

---
# 9 Absolute Position Encoding

Absolute encoding:

x_i = token_i + p_i

Problem:

Attention learns patterns:

(i , j)

instead of

(j − i)

Example:

(1,2) strong
(5,6) strong

But model fails for unseen pairs.

This is the **pairwise generalization problem**.

---
# 10 Relative Position Idea

Desired attention form:

s_ij = q_i k_j^T + g(i − j)

Meaning attention depends on **distance**.

Relative embeddings implement:

s_ij = q_i (k_j + r_{i−j})^T

But this breaks GPU matrix multiplication structure.

Efficient attention prefers:

S = QK^T

single large GEMM.

---
# 11 Sinusoidal Positional Encoding

Define:

p_{i,2t} = sin(ω_t i)
p_{i,2t+1} = cos(ω_t i)

Property:

p_i^T p_j = cos(ω(i − j))

Thus relative position emerges naturally.

However only part of the attention score depends on distance.

---
# 12 RoPE (Rotary Position Embedding)

RoPE rotates query and key vectors by position.

For each dimension pair:

R_i =
[cosθ_i −sinθ_i
 sinθ_i  cosθ_i]

Apply rotation:

q'_i = R_i q_i
k'_j = R_j k_j

Attention becomes:

s_ij = q_i^T R_{j−i} k_j

Key property:

Attention becomes purely **relative position dependent**.

Benefits:

• preserves vector magnitude  
• long context stability  
• elegant mathematical structure  

---
# 13 Transformer Scaling Rules

Empirical scaling rules observed across LLMs.

FFN size:

d_ff ≈ 4 × d_model

GLU variant:

d_ff ≈ 2.66 × d_model

Attention heads:

d_head ≈ 64 – 128

Model aspect ratio:

d_model / n_layers ≈ 100 – 200

This balances:

parallel efficiency and training stability.

---
# 14 Regularization in LLM Pretraining

Traditional methods:

• Dropout  
• Weight decay  

Observation:

Modern LLMs often **disable dropout**.

Reason:

Datasets are extremely large, so overfitting is minimal.

Weight decay helps optimization rather than generalization.

---
# 15 Weight Norm Drift

LayerNorm introduces **scale symmetry**.

Scaling weights does not change model outputs.

Thus the gradient along scale direction is near zero.

SGD noise causes:

Random walk in weight magnitude.

Weight decay prevents this drift.

Loss with decay:

L' = L + λ||θ||²

---
# 16 Training Stability Tricks

Common stabilization methods:

Z‑Loss
Penalizes large logits.

QK Normalization
Normalize queries and keys before attention.

Logit Soft‑Capping

tanh(logits)

These prevent:

• softmax saturation  
• overflow  
• unstable gradients  

---
# 17 Efficient Attention Variants

Standard attention cost:

O(n²)

Optimizations focus on inference efficiency.

MQA (Multi‑Query Attention)

Many queries share a single key/value head.

Benefits:

• smaller KV cache  
• faster inference  

GQA (Grouped Query Attention)

Intermediate design between MHA and MQA.

Sparse Attention

Attend only to local or structured tokens.

Used for long context models.

---
# 18 Key System‑Level Design Principles

Modern LLM architecture follows a few consistent rules.

1 Preserve GPU‑friendly matrix operations

2 Keep residual stream stable

3 Prefer multiplicative interactions

4 Use relative positional encoding

5 Trade parameters for compute efficiency

---
# Final Mental Model

Transformer ≈

Residual Memory
+
Specialized Readers/Writers
+
Similarity‑Based Routing (Attention)
+
Large Nonlinear Transformations (MLP)

Layer by layer the model refines representations stored in the residual stream.

Modern LLM improvements mostly optimize:

• training stability
• compute efficiency
• scaling behavior

rather than inventing entirely new architectures.
