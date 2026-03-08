# Tokenizer
- Good tokenization web app: [https://tiktokenizer.vercel.app]
## Core Mechanism
- text
 - → tokenizer
 - → token_ids
 - → embedding table
 - → dense vectors
 - → transformer layers
- Token:
    - basic unit processed by LLM.
- Token ID:
    - integer index in vocabulary.
- Embedding:
    - lookup table mapping token_id → dense vector.
---

## Unicode Standard
### Character
- Human-readable symbols.

- Examples:
    - Letters (A, b, c)
    - Chinese characters (你, 好)
    - Emoji (😊)

Unicode tries to represent **all characters from all languages**.

---
### Code Point
- A **code point** is the unique integer (~149k) assigned to each character in Unicode.

- Examples : 
    - "A" → U+0041 → 65  
    - "你" → U+4F60 → 20320

```python
codePoint = ord("A") # 65 ord(char) will return Unicode code point of that char
char = chr(65) # A chr(int) will return character of that code point
```
- The reason why not using code point as tokenId -> huge vocab & long sequence
### UTF (Unicode Transformation Format)
- UTF defines how code points are stored as bytes in memory. UTF-X : X = Code unit size
    - UTF-8 : **Variable-length** using 1-4 bytes (at least 1 byte = 8 bits)
        - Examples:
            - A      → 41
            - 你      → E4 BD A0
            - 😊      → F0 9F 98 8A
        - Properties:
            - ASCII compatible
            - most common encoding on the internet
    - UTF-16 : Uses **2 or 4 bytes** per character.
        - Often used internally by some systems (e.g. Java, Windows).
        - UTF-16BE : Big Endian (Most significant first) : A -> 00 41
        - UTF-16LE : Little Endian (Least significant first) : A -> 41 00
    - UTF-32 : **Fixed 4 bytes** for every character.
        - Very simple decode but inefficient in memory.
```python
bytes = "안녕하세요 👋 (hello in Korean!)".encode("utf-8") # ascii is readable (0-127)
# else /xNN, /x mean hexa, NN is int so /xNN = 1 byte
ids = list(bytes) # list will iterate byte sequence
ids = list(map(int,bytes)) # map(function, iterable) apply function on every element of iterable
# Actually bytes can indexed like bytes[0] = int
# Knowledge bytearray is mutable, while bytes is not
```

## Tokenizer Design Tradeoff
- Need balance between: **vocabulary size vs sequence length**
- Because:
    - attention complexity = O(n²)
    - larger tokens → shorter sequence
    - smaller tokens → longer sequence
---
## Tokenization Granularity
### Byte Level
token = raw byte (0–255)
- Pros:
    - universal (works for any language)
- Cons:
    - sequence length becomes very long
    - attention cost increases
- Example:
    - "token"
    - byte tokens:
    - [116,111,107,101,110]
---
### Character Level
token = character
- Pros:
    - small vocabulary  
- Cons:
    - sequence still long
    - limited semantic information per token
---
### Subword Level (most common)
- token = frequently occurring substring.
- Example algorithms:
    - BPE (Byte Pair Encoding)
    - Unigram LM
    - WordPiece
- Pros:
    - good balance between vocabulary size and sequence length
    - handles rare words
#### Example
- "tokenization"
- subword tokens:
    - ["token", "ization"]

## Byte Pair Encoding (BPE) ~ Compression Algorithm

### Mechanism
1. Initialize vocabulary with bytes (0–255)
2. Count most frequent adjacent token pairs
3. Merge the most frequent pair into a new token
4. Repeat until reaching desired vocabulary size 
    - Hyper Params = num_merge = sweet spot : balanced vocab & sequence length
    - num_merge = vocab_size - 256
### Merge Rules
- BPE stores **merge rules**:
    - (a n) -> 256
    - (an a) -> 257
    - Order of rules must be preserved for encoding.

## Tokenizer
- Tokenizer is **trained separately from the LLM** (not required to use same training set).
- Output:
- text ↔ token ids
## Encode
-    text
-    ↓
-    UTF-8 bytes
-    ↓
-    byte tokens
-    ↓
-    apply merge rules
-    ↓
-    token ids 
## Decode 
- token ids
- ↓
- byte sequence
- ↓
- UTF-8 decode
- ↓
- text

## Regex to preprocess, separate wording, prevent to not merge some string pattern
    - gpt2 : 
```python
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
    - gpt4

## tiktoken library
```python 
# pip install tiktoken : inside have Ready-To-Use Vocab and MergeRule
import tiktoken
# GPT-2 (does not merge spaces)
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("    hello world!!!")) # [220, 220, 220, 23748, 995, 10185]

# GPT-4 (merges spaces)
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("    hello world!!!")) # [262, 24748, 1917, 12340]

# load by yourself
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json
import os, json
with open('encoder.json', 'r') as f:
    encoder = json.load(f) # <--- ~equivalent to our "vocab"

with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
# ^---- ~equivalent to our "merges"
```

## Special token we added e.g. <|endoftext|> in GPT2
    - Special tokens are used to encode structural or control information in the token sequence
        - **Structure information** : <bos>, <eos>, <pad>
            - sentense end
            - text end
            - character 
        - **Control signal** : <system>
            - when to answer
            - behavior
            - prompt
    - Soft Prompt : trainable embeddings got trained to represent similar thing
        - In some training setups (e.g. prompt tuning or distillation),
        a learned special token embedding can replace a long prompt,
        acting as a compressed representation of that prompt.

## SentencePiece
- SentencePiece itself is not a tokenization algorithm. It is a tokenizer framework that implements multiple algorithms.
- Main algorithms supported:
    - BPE (Byte Pair Encoding)
    - Unigram Language Model : maximize P(sentence)
### SentencePiece vs GPT Tokenizer
- A common misunderstanding is that SentencePiece introduced a fundamentally different tokenizer. In reality, modern LLM tokenizers usually follow one of two designs.
#### Byte-level tokenization
- Used by GPT-style tokenizers.
- Pipeline:
- text → UTF-8 encoding → bytes (0–255) → BPE merges → tokens
- Example:
    - hello  
    - ↓  
    - [104,101,108,108,111]
---
#### Character-based subword tokenization
- Used by SentencePiece by default.
- Pipeline:
`text → unicode characters → subword merges → tokens`
- Example:
    - hello  
    - ↓  
    - ["h","e","l","l","o"]
---

- The difference is the base unit.
| Tokenizer | Base Unit |
|------|------|
| SentencePiece | Unicode characters |
| GPT tokenizer | Raw bytes |

### Why Byte Tokenizers Became Popular
- Byte tokenizers solve a major issue: coverage.
- Coverage means the tokenizer can encode any possible input.
- Character-based tokenizers may encounter OOV (Out Of Vocabulary) characters if never seen in training set (but solved by **Byte fallback**).
- Examples:
    - rare Unicode symbols
    - emojis
    - corrupted text
    - mixed languages
- Byte tokenizers avoid this because: Any text → UTF-8 → bytes
- **Therefore coverage is effectively 100%.**
- This is one reason many modern LLMs prefer byte-level tokenization.
### Unigram
- Unigram Language Model takes a probabilistic approach.
- Idea:
    - Start with a large vocabulary of candidate tokens.
    - Then iteratively remove tokens that reduce corpus likelihood.
- Goal:
    - maximize P(corpus) Where P(corpus) is the probability assigned to the training text.

- Because it evaluates full tokenizations, Unigram LM often produces better segmentations.
- Advantages:
    - more flexible segmentation
    - better handling of ambiguous splits
    - more stable vocabulary
### Subword Regularization
- SentencePiece introduces a technique called **subword regularization**.
    -  means that during training the tokenizer samples multiple valid tokenizations of the same sentence.
- Example word:
- unbelievable
- Possible segmentations:
    - un + believable
    - unbeliev + able
    - un + believe + able
- Instead of always choosing one segmentation, the model randomly samples among them.
- Benefits:
    - acts like data augmentation
    - improves model robustness
    - reduces overfitting to a single segmentation
- This technique is widely used in neural machine translation and multilingual models.
### Space Encoding Trick
- SentencePiece replaces spaces with a special marker: "_" : most NLP pipeline historically normalize whitespace.
- Why this matters:
    - If spaces were removed, decoding would become ambiguous.
    - By embedding spaces into tokens, SentencePiece ensures lossless decoding.
    
**However, modern LLMs increasingly favor byte-level tokenizers because they guarantee full input coverage.**

## Tokenizer Code In Python
```python
from collections import defaultdict
from collections import Counter
def count_pair(ids): 
# input : list of int , output : dict -> counts consecutive pair
    counts = {}
    # counts = defaultdict(int) # default of int is 0
    # Counter(zip(ids, ids[1:]))
    for pair in zip(ids, ids[1:]): 
        # zip(a,b) = [tuple(a[n],b[n]) for n in range(min(len(a),len(b)))]
        counts[pair] = counts.get(pair, 0) + 1 # dict.get(key, default)
        # counts[pair] += 1 # no need to worry about key error
    return counts

stats = get_stats(tokens)
"""
stats = {
 (1,2): 5,
 (2,3): 2,
 (3,4): 7
}
"""
top_pair = max(stats, key=stats.get) # O(n), max(iterable, key=function), compared with function(element),dict.get = value/counts

## Using Priority queue to boost speed
top_pair = priority_queue((count,pair)) # O(logn)
import heapq # min heap, heap operations on list
heap = []
for pair, count in stats.items():
    heapq.heappush(heap, (-count, pair))
count, top_pair = heapq.heappop(heap)
while True: # lazy deletion get max count
    count, pair = heapq.heappop(heap)
    if stats[pair] == -count:
        break
        
def merge(ids, pair, idx): # O(n)
  # in the list of int **ids**, replace all **pair** with the new token **idx**
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
    # if ids[i:i+2] == list(pair): # slicing
      newids.append(idx) # O(1)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

## Tokenizer training pipeline -> Goal : Merges Rule & Vocabulary
vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens))
ids = list(tokens) # training data

merges = {} # (int, int) -> int, after py3.7 merges.items() = insertion order
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  ids = merge(ids, pair, idx)
  merges[pair] = idx


# Decode
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items(): # (int, int) -> int
    vocab[idx] = vocab[p0] + vocab[p1] # bytes concatenation

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace") # errors = "replace" mean show � for unk char
  return text

# Encode
def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8")) # get byte sequences
  while len(tokens) >= 2: # when only 1 token cant make a pair
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf"))) # lower ids -> prior
    if pair not in merges: # all inf mean no mergeable pair
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens
```
