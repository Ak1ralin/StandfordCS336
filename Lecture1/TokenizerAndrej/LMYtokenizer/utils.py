from collections import Counter

def count_pairs(ids) : # ids = list of ids 
    return Counter(zip(ids,ids[1:])) # zip(a,b) = [tuple(a[n],b[n]) for n in range(min(len(a),len(b))) ]

def count_pairs_corpus(corpus_ids) : # ids = list of list of ids, for pretokenized
    stats = Counter()
    for ids in corpus_ids:
        stats.update(zip(ids,ids[1:]))
    return stats # zip(a,b) = [tuple(a[n],b[n]) for n in range(min(len(a),len(b))) ]

def merge(ids, pair, new_idx): # replace pair with new_idx
    new_ids = []
    i = 0 
    while i < len(ids):
        if tuple(ids[i:i+2]) == pair:
            new_ids.append(new_idx)
            i += 2
        else :
            new_ids.append(ids[i])
            i += 1
    return new_ids

def merge_corpus(corpus_ids,pair,new_idx): # for pretokenized
    return [merge(ids,pair,new_idx) for ids in corpus_ids]
        