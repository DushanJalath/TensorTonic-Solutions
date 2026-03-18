def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    intersection = set(set_a)&set(set_b)
    union = set(set_a)|set(set_b)
    if len(union)==0:
        return 0
    return float(len(intersection)/len(union))