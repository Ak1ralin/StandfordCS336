from .utils import count_pairs, merge, count_pairs_corpus, merge_corpus
import regex as re
import json
from pathlib import Path
PATTERN = r"""
(?i:'s
|'t
|'re
|'ve
|'m
|'ll
|'d)
|[^\r\n\p{L}\p{N}]?\p{L}+
|\p{N}{1,3}
|\x20?[^\s\p{L}\p{N}]+[\r\n]*
|\s*[\r\n]
|\s+(?!\S)
|\s+
"""
class LMYTokenizer :
    def __init__(self, merges=None, vocab=None, load_path=None):
        self.merges = merges or {} # merge order
        self.vocab = vocab or {idx:bytes([idx]) for idx in range(256)} # vocabulary
        self.pattern = re.compile(PATTERN,re.VERBOSE)
        if load_path is not None:
            self.load(load_path)

    # def train (self, text, vocab_size):      
    #     ids = list(text.encode("utf-8")) # since python 3, bytes[i] = int 
    #     if vocab_size <= len(self.vocab):
    #         return
        
    #     next_idx = len(self.vocab)
    #     num_merge = vocab_size - next_idx

    #     for _ in range(num_merge):
    #         stats = count_pairs(ids)
    #         if not stats:
    #             break 

    #         most_freq_pair = max(stats, key = stats.get)

    #         ids = merge(ids,most_freq_pair,next_idx)
    #         self.merges[most_freq_pair] = next_idx
    #         self.vocab[next_idx] = self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]
    #         next_idx += 1

    #     return 

    def train(self, text, vocab_size): # for pretokenized
        chunks = self.pretokenize(text)
        corpus_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
        if vocab_size <= len(self.vocab):
            return
        
        next_idx = len(self.vocab)
        num_merge = vocab_size - next_idx
        for _ in range(num_merge):
            stats = count_pairs_corpus(corpus_ids)
            if not stats:
                break

            most_freq_pair = max(stats, key = stats.get)

            corpus_ids = merge_corpus(corpus_ids,most_freq_pair,next_idx)
            self.merges[most_freq_pair] = next_idx
            self.vocab[next_idx] = self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]
            next_idx += 1
        
        return

    # def encode (self, text): # from String to list_ids
    #     bytes = text.encode("utf-8") # get bytes
    #     ids = list(bytes) # since python 3, bytes[i] = int 
    #     while len(ids) >= 2:
    #         stats = count_pairs(ids)
    #         lowest_merge_pair = min(stats,key = lambda p : self.merges.get(p,float('inf')))
    #         if lowest_merge_pair not in self.merges:
    #             break
    #         idx = self.merges[lowest_merge_pair]
    #         ids = merge(ids,lowest_merge_pair,idx)
    #     return ids

    def encode (self, text): # from String to list_ids
        chunks = self.pretokenize(text)
        corpus_ids = [list(chunk.encode("utf-8")) for chunk in chunks]
        encoded = []
        for ids in corpus_ids :
            while len(ids) >= 2:
                stats = count_pairs(ids)
                lowest_merge_pair = min(stats,key = lambda p : self.merges.get(p,float('inf')))
                if lowest_merge_pair not in self.merges:
                    break
                idx = self.merges[lowest_merge_pair]
                ids = merge(ids,lowest_merge_pair,idx)
            encoded.extend(ids) # flatten
        return encoded
    
    def decode (self, ids): # from list_ids to String
        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        return token_bytes.decode("utf-8", errors = "replace")
    
    def pretokenize(self, text):
        return self.pattern.findall(text)
    
    def export(self, output_path):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        merges_path = output_dir / "merges.json"
        vocab_path = output_dir / "vocab.json"

        # JSON object keys must be strings, so tuple keys are stored as "left,right".
        merges_data = {f"{left},{right}": idx for (left, right), idx in self.merges.items()}
        # Store bytes as hex strings to keep round-trip exact.
        vocab_data = {str(idx): token_bytes.hex() for idx, token_bytes in self.vocab.items()}

        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_data, f, ensure_ascii=False, indent=2)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        return

    def load(self, input_path):
        input_dir = Path(input_path)
        merges_path = input_dir / "merges.json"
        vocab_path = input_dir / "vocab.json"

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.merges = {
            tuple(map(int, pair_str.split(","))): idx
            for pair_str, idx in merges_data.items()
        }
        self.vocab = {int(idx_str): bytes.fromhex(hex_str) for idx_str, hex_str in vocab_data.items()}
        if not self.vocab:
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
        return
