def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    hits = 0
    for rec, truth in zip(recommendations, ground_truth):
        top_k   = set(rec[:k])       # top-k items for this user
        relevant = set(truth)         # relevant items for this user
        if len(top_k & relevant) > 0: # at least 1 hit
            hits += 1
    
    return hits / len(recommendations)