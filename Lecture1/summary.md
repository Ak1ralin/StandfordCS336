# CS336 -- Language Models From Scratch (Lecture 1 Summary)

## 1. Course Goal

The course teaches how to **build a language model from scratch**.

Main philosophy: - **Understanding by building** - Implement every
component of an LLM pipeline yourself.

Focus areas: 
- Mechanics: how systems actually work. 
- Mindset: thinking about efficiency and scaling. 
- Intuitions: which data and modeling decision yield good accuracy.

The course intentionally avoids only using existing APIs or prompting
large models.

------------------------------------------------------------------------

# 2. Why This Course Exists

Modern researchers interact with AI at higher abstraction levels:

  Time           Typical Workflow
  -------------- -------------------------------------------
  10 years ago   Implement models from scratch
  5 years ago    Download model (e.g., BERT) and fine‑tune
  Today          Prompt large models (GPT‑4, Claude, etc.)

Problem: Researchers are becoming **disconnected from the underlying
systems**.

**The course rebuilds this understanding.**
**accuracy = efficiency x resource**
------------------------------------------------------------------------

# 3. Landscape of Language Models

Example frontier models:

-   GPT‑4
-   Claude
-   Gemini

Examples of earlier important models:

-   GPT‑2
-   GPT‑3
-   T5
-   LLaMA
-   Mistral
-   BLOOM
-   OPT

Key research trends: 
- Increasing **model size** 
- Studying **scaling laws** 
- Moving toward **compute‑optimal training** 
- Overtraining smaller efficient models

------------------------------------------------------------------------

# 4. Brief History of Language Models

Major milestones:

1.  **Shannon (1950)** -- Language models to measure English entropy.
2.  **n‑gram models** -- Statistical language modeling.
3.  **Bengio (2003)** -- First neural language model.
4.  **Seq2Seq models (2014)** -- Machine translation models.
5.  **Adam Optimizer (2014)** -- Base optimizer until now.
6.  **Attention mechanism (2014)** -- Improves sequence modeling.
7.  **Transformer (2017)** -- Replaces recurrence with attention.
8.  **Mixture of Experts (2017)** -- Partially activate params.
9.  **Model parallelism (2019)** -- Speed boosting.
10.  **GPT series** -- Large scale autoregressive models.

------------------------------------------------------------------------

# 5. LLM Training Pipeline

A simplified language‑model pipeline:

    Raw Data
       ↓
    Data Processing
       ↓
    Tokenization
       ↓
    Pretraining (Transformer LM)
       ↓
    Instruction Tuning
       ↓
    Preference Alignment

Each stage has major design decisions.

------------------------------------------------------------------------

# 6. Data Collection

Raw data sources include:

-   Web pages
-   Books
-   Research papers
-   Code repositories

Problems with raw data: 
- Low quality content 
- Harmful text 
- Duplicate documents

Processing steps: 
- **Filtering** -- remove bad data. 
- **Deduplication** -- avoid repeated training data. 
- **Conversion** -- convert formats (HTML → text).

------------------------------------------------------------------------

# 7. Tokenization

Tokenization converts text into numbers.

Definition: **Token** = smallest unit a model processes.

Example pipeline:
    text → tokens → embeddings → transformer

Trade‑off:

  |Large Vocabulary  |  Small Vocabulary
  |-------------------| ------------------
  |Shorter sequences  | Longer sequences
  |More memory        | Less memory

The course uses **Byte Pair Encoding (BPE)**.

------------------------------------------------------------------------

# 8. Model Architecture

The core model is a **Transformer Language Model**.

Transformer definition:

A neural network architecture that uses **attention mechanisms** instead
of recurrence to model relationships between tokens.

Variants often modify:

-   normalization layers
-   activation functions (e.g., SwiGLU)
-   attention implementations
-   positional embeddings

------------------------------------------------------------------------

# 9. Pretraining

During pretraining, the model learns:

    P(next_token | previous_tokens)

Key components:
-   Optimizer (e.g., AdamW)
-   Learning rate schedule (e.g., cosine decay)
-   Hyperparameters
    -   batch size
    -   number of heads
    -   hidden dimension

------------------------------------------------------------------------

# 10. Instruction Tuning

Instruction tuning uses examples of:

    (prompt, response)

Goal: Teach the model to **follow human instructions**.

Training objective:

    maximize P(response | prompt)

Datasets often used: - Alpaca - instruction datasets derived from LLM
outputs

------------------------------------------------------------------------

# 11. Preference Alignment

After instruction tuning, alignment improves response quality.

Process:

1.  Model generates multiple responses.
2.  Humans choose the better response.
3.  Model is trained using the preference signal.

Example data:

    (prompt, responseA, responseB, preference)

Common algorithms:

-   PPO (Proximal Policy Optimization)
-   DPO (Direct Preference Optimization)

------------------------------------------------------------------------

# 12. Core Theme of the Course: Efficiency

Training large models is limited by resources:

-   Data
-   Compute
-   Memory
-   Communication bandwidth

Goal:
**Train the best model possible given fixed resources.**

Examples:

-   Efficient data filtering
-   Tokenizers that reduce sequence length
-   GPU‑efficient architectures
-   Optimal scaling laws

------------------------------------------------------------------------

# 13. Key Takeaway

Building LLMs requires understanding the full stack:
-   data pipelines
-   tokenization
-   model architectures
-   optimization
-   alignment

The course focuses on **learning the engineering mindset required to
train large language models from scratch.**
