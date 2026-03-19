def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    stopwords=set(stopwords)
    return [word for word in tokens if word not in stopwords]