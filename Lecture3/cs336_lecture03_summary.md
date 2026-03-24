# CS336 Lecture 03 --- LLM Architecture & Training

## Goal of the Lecture

Understand **how modern large language models are architected and
trained**, and what design choices most real models share.

The lecture analyzes **many recent LLM papers** to determine:

-   What architecture components are common
-   Which parts vary across models
-   What hyperparameters matter most
-   What tricks make training stable

------------------------------------------------------------------------

## Baseline: Original Transformer

Original Transformer architecture contained:

Components: 
- Token embedding 
- Positional encoding (sin/cos) 
- Multi‑Head Self Attention 
- Feed Forward Network (FFN) 
- Residual Connections 
- LayerNorm 
- Final linear + softmax

Key design choices:

  Component          | Original Choice
  ------------------- |------------------------
  Position encoding  | Sinusoidal
  Activation         |ReLU
  Normalization       |Post‑LayerNorm
  FFN                 |Linear → ReLU → Linear

------------------------------------------------------------------------

## Modern Transformer Variant (Typical LLM)

Most modern LLMs use a slightly different architecture.

Common differences:

  Component          | Modern Choice
  -------------------| ---------------------------------------
  Normalization      | **Pre‑Norm with RMSNorm**
  Position encoding   |**RoPE (Rotary Position Embeddings)**
  Activation          |**SwiGLU / GeGLU**
  Bias terms          |Usually removed

Motivation: 
- Better training stability 
- Faster training 
- Fewer parameters

------------------------------------------------------------------------

### Pre‑Norm vs Post‑Norm

Normalization can be placed **before or after** the residual block.

#### Post‑Norm (original transformer)

![PostNorm](src/postNorm.png)

#### Pre‑Norm (modern LLMs)

![PreNorm](src/preNorm.png)

#### Why Pre‑Norm Wins

Empirical findings:

-   Better **gradient flow**
-   Prevents **gradient spikes**
-   Enables **larger learning rates**
-   More stable for **very deep networks**

Almost all modern LLMs use **pre‑norm**.

#### Double Norm

Putting 2 Norm layer double team the FFN/Att

![DoubleNorm](src/doubleNorm.png)

------------------------------------------------------------------------

### RMSNorm vs LayerNorm

#### LayerNorm

Normalizes using:

-   mean
-   variance

Formula:

$$ y = \frac{x − mean(x)}{\sqrt{var(x) + ε}} * γ + β $$

#### RMSNorm

Simpler normalization:

$$ y = \frac{x}{\sqrt{mean(x²) + ε}} * γ $$

Differences:

  Feature          |LayerNorm  | RMSNorm
  ---------------- |----------- |---------
  subtract mean    |yes         |no
  bias parameter   |yes         |no
  compute cost     |higher      |lower

#### Why RMSNorm -> Simplification 

Advantages:

-   fewer parameters (**no bias** term to store)
-   fewer operations (**no mean** calculation)
-   faster runtime (**similar performance**)

Even though matrix multiplies dominate FLOPs so you might thinking these improvement is nothing, but normalization still matters due to **memory movement costs** because Normalization take 25 % of runtime.

------------------------------------------------------------------------

### Dropping Bias Terms

Most modern transformers remove bias parameters from:

-   Linear layers
-   LayerNorm


Because of the residual connection

$$x_{l+1} = x_l + F(x_l)$$

the residual stream can introduce a constant shift vector c: `x' = x + c`
  - c from last F(x) = c -> this is possible 

When the next linear layer applies W, we get `(x + c)W = xW + cW`

Where `cW` behaves like a bias term.

Therefore residual connections make explicit bias parameters largely redundant.

However residual accumulation can cause the magnitude of the residual stream to drift across layers:

`x{l+1} = x_l + F(x_l)`

Without normalization this may lead to unstable activation scales.

Normalization layers (LayerNorm / RMSNorm) stabilize the scale of the
residual stream during training, which makes this bias-free parameterization trainable in practice.

Reasons:

1.  Slight memory savings 
2.  Slight compute savings 
3.  Empirically little performance loss
4.  Improve stability

------------------------------------------------------------------------

### Activation Functions

Many activation functions exist:

Examples:

-   ReLU (Rectified Linear Unit) ~ max(0,x)
-   GeLU (Gaussian Error LU) ~ x*prob_G_CDF = smoother than ReLU
-   Swish (Sigmoid) ~ x*sigmoid(x) ~ allow negative value pass
-   GLU variants

#### Feed Forward Layer

Standard form:

    FF(x) = activation(xW1)W2

------------------------------------------------------------------------

#### Gated Activations (GLU)

Problem of old LU : These activation function cant learn, nonlinearity shape is constant, GLU make Gate become learnable function (V)

GLU variants introduce **gating** learnable gate.

Example: ReGLU

    FF(x) = (ReLU(xW1) ⊙ (xV))W2

where:

⊙ = element‑wise multiplication

Popular variants:

  Activation   |Used by
  ------------ |------------------
  GeGLU        |T5
  SwiGLU       |LLaMA
  ReGLU        |some experiments

##### Key Insight

Gated activations consistently improve performance slightly, because function space become bigger.

Therefore **SwiGLU** became common.

------------------------------------------------------------------------

### Serial vs Parallel Transformer Blocks

Standard transformer blocks compute:

`output = x + MLP(LN(x + Att(LN(x))))`

This is **serial**.

Some models compute both in parallel: 

`output = x + MLP(LN(x)) + Att(LN(x))`
  
Both MLP and Att is neural network -> function approximator

if params big enough

`MLP(LN(x)) + Att(LN(x)) ~ MLP(LN(x + Att(LN(x))))`

Advantages:

-   faster training ~ 15% faster training
-   better kernel fusion (same input, only 1 memory read)
-   might have better gradient stability (2 gradient paths)

------------------------------------------------------------------------

### Position Embeddings

Transformers need positional information.  because inside embedding dont have positional info, its mean attention depend on text content which is wrong, so the effect of position embedding is adding positional info into embedding 

Without Positional Embedding :
$$x_0 = token_{emb}(tok)$$

$$Q_0 = x_0 W_Q$$

With Positional Embedding

$$x_0 = token_{emb}(tok) + emb_{pos}$$

$$Q_0 = x_0 W_Q$$

Common approaches:

  Type                 |Idea
  -------------------- |----------------------------------------
  Sinusoidal           |add sin/cos encoding
  Absolute embedding   |add learn position vector
  Relative embedding   |attention depends on relative distance
  RoPE                 |rotate vectors by position

You might confuse, why not just add 1 more dimension on embedding to represent position 

#### Self-Attention formula

Define Query and Key as 

$$q_i = x_i W_Q$$

$$k_j = x_j W_K$$

$$x_i \in \mathbb{R}^d$$
$$W_Q, W_K \in \mathbb{R}^{d \times d_k}$$

Attention score will be
$$
s_{ij} = q_i k_j^T
$$

$$
s_{ij} = (x_i W_Q)(x_j W_K)^T
$$

$$
s_{ij} = x_i W_Q W_K^T x_j^T
$$
let 
$
A = W_Q W_K^T
$


$$
s_{ij} = x_i A x_j^T = \sum_{a,b}x_{i,a}A_{a,b}x_{j,b}
$$

called **bilinear form ($x^TAy$)** ~ f(x,y) and linear to both x and y
 
---

#### 1-D Positional Encoding

Assume we just encode position into a last dimention：$p_i = position$

$$
s_{ij} = x_i A x_j^T
$$

$$
s_{ij} = \sum_{a,b}x_{i,a}A_{a,b}x_{j,b}
$$

the only part  **position interaction**：

$$
s_{ij}^{pos} = x_{i,d} A_{d,d} x_{j,d}
$$
because only 1 dimension so 
$$
A_{d,d} = c  
$$
$$
s_{ij}^{pos} = c \, p_i p_j
$$
mean position info model can get = `f(i,j) = cij` which can't stable represent distance infomation `distance = j - i` to extend semantic positional information we need 

as a result key problem of **1-D positional encoding**

`position interaction capacity too low, cji can not represent i-j`

#### Absolute Positional Encoding

Absolute positional encoding use high-dim embedding to express positional information

$$
p_i \in \mathbb{R}^d
$$

encode by
$
x_i = token_i + p_i
$


$$
s_{ij} = x_i A x_j^T
$$

$$
s_{ij} =
(token_i + p_i) A (token_j + p_j)^T
$$
$$
s_{ij} =
token_i A token_j^T
+ token_i A p_j^T
+ p_i A token_j^T
+ p_i A p_j^T
$$

focus on position interaction：

$$
s_{ij}^{pos} = p_i A p_j^T
$$

this time

$$
p_i A p_j^T =
\sum_{a,b} p_{i,a} A_{ab} p_{j,b}
$$

and $p_i ∈ R^d A ∈ R^{d×d}$, this time $f(position_i , position_j)$ with big capacity

Problem is that pair-wise relationship $f(position_i , position_j)$ because its not actually $f(j - i)$ model tend to learn **position pair patterns** instead of distance (1 → strong attention)
- (5,6) → strong attention
- (10,11) → strong attention
- (1,100) → weak attention

as a result, if L_train = 512 it will only have these pair (0..511 , 0..511), while eval have out of range pair the behavior might not stable

this is **pair-wise generalization problem**

---
#### Relative Positional Embedding

What model actually need is **relative position (distance)**

$$
s_{ij} = f(token_i, token_j, \mathbf{distance}(i,j))
$$

not **absolute position**

$$
f(token_i, token_j, i, j)
$$

because language semantic 

* dependency depends on **distance**
* not absolute index

so the goal become

$$
s_{ij} = q_i k_j^T + g(i-j)
$$

and

$$
s_{ij}^{pos} = g(i-j)
$$

based on (Shaw et al. 2018)
instead of using g(i-j)
which need to construct NxN (pair) matrix for store bias
- extra NxN matrix memory
- s not pure matmul but matmut then add, not prefered by GPU

As a result, paper change change $g(dist)$ to $q_ir^T_{i-j}$

$$
r_{i-j} \in \mathbb{R}^{d_k}
$$

is a **relative position embedding** learnable 

self attention become：
$$
s_{ij}
=
q_i k_j^T + q_i r_{i-j}^T
$$

$$
s_{ij} =
q_i (k_j + r_{i-j})^T
$$

$$
s_{ij}^{pos} = q_i r_{i-j}^T
$$

$$
r_{i-j} = f(i-j) \rightarrow
s_{ij}^{pos} = f(i-j)
$$

Relative embedding still have problem：

Transformer attention need to calculate $N^2$ scores, and memory access pattern is not prefered.

Normally Attention is done by 
$$S = QK^T$$
which called GEMM (General Matrix Multiply)

when come to relative positional
$$S_{i,j} = q_ik^T_j + q_ir^T_{i-j}$$
and r table is not sequential, its RAM -> `cache hit low` make memory latency become higher, R cant make it S truely express, so still need to RAM

---

#### Sinusoidal Positional Encoding

Due to the problem we meet in **Relative Positional** we try to find out a trick to make relative position automatic appear inside $q_ik^T_j$

like **Absolute positional embedding** 
just change p
$$
s_{ij} =
(token_i + p_i) A (token_j + p_j)^T
$$

We define $p_i$ like this , i = pos, t = # of sin/cos pair in $p_i$, d = dimension of $x_i/p_i$
$$
p_{i,2t} = \sin(\omega_t i)
$$
$$
p_{i,2t+1} = \cos(\omega_t i)
$$

$$
\omega_t = 10000^{-2t/d}
$$

$$
p_i =
[\sin(\omega_0 i),\cos(\omega_0 i),
\sin(\omega_1 i),\cos(\omega_1 i)...]
$$

Prove : 


$$
\sin(a-b) =
\sin a \cos b - \cos a \sin b
$$

$$
\cos(a-b) =
\cos a \cos b + \sin a \sin b
$$

So

$$
p_i^T p_j
$$

Become

$$
\cos(\omega (i-j))
$$

which mean
$$
p_i^T p_j = f(i-j)
$$

---

##### Prove

$$
s^{pos}_{ij} =
p_i W_qW_k^T  p_j^T
$$

$$
p_i =
[\sin(\omega i), \cos(\omega i)]
$$

$$
p_j =
[\sin(\omega j), \cos(\omega j)]
$$

$$ 
u_i = p_iW_q = a\sin(\omega i) + b\cos(\omega i) \in vector
$$

$$
v_j = p_jW_k = c\sin(\omega j) + d\cos(\omega j) \in vector
$$

$$
s^{pos}_{ij} =
u_iv^T_j = sinsin + coscos + sincos + cossin 
$$

which finally can be write in combination of sin(w(i-j)) and cos(w(i-j))

$$s^{pos}_{ij} = f(i-j)$$

---

##### Sin/Cos Problem

even it implicit represent relative position, but its not enough
$$
s_{ij} =
token_i A token_j^T 
+ token_i A p_j^T
+ p_i A token_j^T
+ p_i A p_j^T
$$

3 out of 4 (even more) of $s_{ij}$ has no **distance information**, so $s^{pos}_{ij} \in f(i-j)$ but $s_{ij}$ not its just partially relative thats the reason why **RoPE** come which make $s_{ij}$ become fully relative

---

#### RoPE (Rotary Position Embedding)
RoPE concept is insane：dont **add position** but **rotate vector by position**
- Others problem is absolute and sinusoidal will pollute input, but its ok because learnable params will reorganize to handle it.

RoPE rotates query/key vectors depending on position.

Core idea: The **inner product between rotated vectors encodes relative position**.


Rotate each 2 dimension separately
$$
(q_{2t}, q_{2t+1})
$$

by

$$
\theta_i = \omega_t i
$$

$$
R_i =
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i \\
\sin\theta_i & \cos\theta_i
\end{bmatrix}
$$

Apply to Q,K：

$$
q_i' = R_i q_i
$$

$$
k_j' = R_j k_j
$$

Attention score

$$
s_{ij}
= q_i'^T k_j'
$$
$$
(R_i q_i)^T (R_j k_j)
$$
$$
q_i^T R_i^T R_j k_j
$$
And
$$
R_i^T = \begin{bmatrix}
\cos\theta_i & \sin\theta_i \\
-\sin\theta_i & \cos\theta_i
\end{bmatrix}
$$
even_func : $\cos-\theta_i = \cos\theta_i$

odd_func :  $\sin-\theta_i = -\sin\theta_i$

$$
R_i^T = \begin{bmatrix}
\cos-\theta_i & \sin\theta_i \\
\sin-\theta_i & \cos-\theta_i
\end{bmatrix}
$$
$$R_i^T = R_{-i}$$
$$
R_{-i}R_j = R(-\theta_i)R(\theta_j) 
$$
$$
R(a)R(b) = R(a+b) 
$$
$$
R_i^T R_j = R(j-i) = R_{j-i}
$$
As a result
$$
s_{ij}
=
q_i^T R_{j-i} k_j
$$

$R_{j-i}$ depends on $(j-i)$
So
$$
s_{ij} = f(i-j)
$$

Actually implementing 


Properties:

-   preserves vector magnitude
-   allows attention to depend on **relative distance**
-   works well for long contexts

Implementation:

1.  split embedding dimensions into pairs
2.  rotate each pair using sin/cos
3.  apply rotation to **query and key** vectors

------------------------------------------------------------------------

### Hyperparameters in Transformers

Important hyperparameters include:

-   feedforward/model dimension
-   number of attention heads
-   vocabulary size
-   how to pick deeper/wider params distribution
-   regularization

------------------------------------------------------------------------

#### Feedforward Dimension Rule

$d_{ff}$ = FFN hidden embedding dimension

$d_{model}$ = token dimension (use by other block except FFN)

Empirical rule:

$$d_{ff} = 4 × d_{model}$$

##### GLU Exception

$$d_{ff} ≈ 2.66 d_{model}$$

Because GLU make the number of params higher in FFN (+$\frac{3}{2}$)
To remain number of FLOPs stable so $d_{ff}$ lower 

------------------------------------------------------------------------

#### Attention Head Dimensions

Attention is just make $W_q,W_k,W_v$ can be more specific

$head_{dim}$ mean dimension of output of each head

Typical relationship: $head_{dim} × num_{head} ≈ d_{model}$

most setting $d_{head} = \frac{d_{model}}{h}$ to remain params & FLOPs are same as single-head, in the meantime let each head work well in its own sub-space.

mostly set $d_{head} = 64-128$

------------------------------------------------------------------------

#### Model Aspect Ratio

Models vary in **depth vs width**.

Typical ratio:

    $d_model / n_layers ≈ 100–200$

Scaling depth models are harder to parallelize (both distributed and sequential computation).

width is good because each GPU only compute some part (distributed) and do it sequentially -> no need to communicate with others every step (parallel)

depth is hard because only 1 GPU is enough for this layer, other is idle, and we cant load params first -> might OOM, if load in others device -> communication every step 

------------------------------------------------------------------------

#### Vocabulary Size

Typical vocab sizes:

  Model Type            |Vocab Size
  --------------------- |------------
  Monolingual models    |30k--50k
  Multilingual models   |100k--250k

Large vocabularies are needed for multilingual/multimodal tasks.

------------------------------------------------------------------------

#### Regularization ~ prevent overfitting

Classic techniques:

- dropout ~ randomly close some neuron during training
  - make neuron can depends on multiple pre-step neuron 
  - mostly using Bernoulli distribution
- weight decay
  - adding L2 penalty in to loss
$$Loss'(\theta) = Loss(\theta) + \lambda\Vert\theta\Vert^2$$
$\theta$ is all params of model mostly except **Embedding** called weight

weight norm $\Vert\theta\Vert = \sqrt{\sum_{k_{(i,j)}} w_{{k_{(i,j)}}}^2}$

##### Why we need weight decay?
- Problem 
  - LayerNorm makes the network approximately invariant to weight scaling.
  - Different weight scales can produce nearly identical outputs after normalization.
  - This property is called **scale symmetry**.
  - The direction in parameter space that only changes weight scale but not the function is called the **scale direction**.
  - In the scale direction, the true gradient is approximately zero.
  - However, mini-batch gradients contain stochastic noise.
  - When the true gradient is close to zero, this noise dominates the update.
  - As a result, the optimizer performs a **random walk in the scale direction**.
  - Since the loss is insensitive to scale, this drift does not increase the loss.
  - This phenomenon is called **weight norm drift**.
- Designing
  - Because of scale symmetry, many parameter configurations represent the same function.
  - Among these equivalent solutions, optimization with weight decay prefers the **minimum norm solution**.
  - Minimum norm solutions tend to correspond to smoother functions and often generalize better.
  - Weight decay adds an L2 penalty:

  $$
  L' = L + \lambda ||\theta||^2
  $$

  - This breaks the scale invariance of the optimization objective, since the loss now increases with the magnitude of the parameters.

  - As a result, the optimizer is biased toward solutions with smaller parameter norms, which helps prevent weight norm drift and encourages a minimum-norm solution.



Observation:

Modern LLMs often **disable dropout** during pretraining.
  * **Dropout was originally designed to improve robustness** by randomly disabling some neurons during training, which prevents the model from relying too heavily on specific features and reduces **overfitting** (the model memorizing training data instead of learning general patterns).

  * **In modern LLM pretraining, datasets are extremely large**, so the risk of overfitting is already very small. Because of this, the regularization benefit of dropout becomes much less important.

  * **Dropout also introduces training noise**, which can hurt optimization stability when training very large models.

  * **It interacts poorly with attention mechanisms**, because randomly dropping units can disrupt the precise information flow needed for attention computations.

  * **Therefore, many modern LLMs disable dropout during pretraining** and rely instead on large datasets, weight decay, and other techniques for regularization.


Instead they rely on:

-   large datasets
-   weight decay

Weight decay mostly helps **optimization stability（scale direction random walk，weight norm drift）** rather than
overfitting.

------------------------------------------------------------------------

#### Training Stability Tricks

Training very large models can become unstable.

Common tricks:

##### 1. Z‑Loss

A regularization term added to the output softmax.

Purpose:

-   prevent logits from becoming extremely large
-   stabilize training

Used in:

-   PaLM
-   OLMo
-   Baichuan

------------------------------------------------------------------------

##### 2. QK Normalization

Normalize **queries and keys** before computing attention softmax.

Purpose:

-   prevent large dot products
-   stabilize attention distribution

------------------------------------------------------------------------

##### 3. Logit Soft‑Capping

Apply:

    tanh(logits)

This caps extremely large logits.

------------------------------------------------------------------------

#### Efficient Attention Variants

Standard attention cost:

    O(n²)

where:

n = sequence length

Several tricks reduce cost.

------------------------------------------------------------------------

#### Multi‑Query Attention (MQA)

Idea:

Use many queries but **only one key/value head**.

Benefits:

-   smaller KV cache
-   faster inference
-   less memory usage

Tradeoff:

-   slightly worse perplexity.

------------------------------------------------------------------------

#### Grouped Query Attention (GQA)

Middle ground between:

-   full multi‑head attention
-   MQA

Queries are split into groups that share keys/values.

Benefits:

-   better quality than MQA
-   still reduces inference cost

Used in many modern LLMs.

------------------------------------------------------------------------

#### Sparse / Sliding Window Attention

Instead of attending to **all tokens**, restrict attention to:

-   nearby tokens
-   structured patterns

Benefits:

-   reduces quadratic cost
-   supports long contexts

Example: **Mistral sliding‑window attention**.

------------------------------------------------------------------------

# Key Takeaways

Across modern LLMs there is significant convergence:

### Architecture

Common pattern:

-   Pre‑Norm
-   RMSNorm
-   RoPE
-   SwiGLU

### Hyperparameters

Typical values:

-   d_ff ≈ 4 × d_model
-   head_dim × heads ≈ d_model
-   vocab ≈ 30k--50k

### Efficiency Tricks

-   MQA / GQA
-   sparse attention
-   KV cache optimization

### Stability Tricks

-   z‑loss
-   QK normalization
-   logit soft‑capping

Overall insight:

Most modern LLMs are **very similar architectures with small
improvements** rather than radically different designs.
