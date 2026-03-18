def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    def getKey(record):
        return tuple(record[col] for col in key_columns)

    def none_count(record):
        return sum(1 for v in record.values() if v is None)

    key_order = []
    candidates = {}

    for record in records:
        key = getKey(record)

        if key not in candidates:
            key_order.append(key)
            candidates[key]=record
        elif strategy=="last":
            candidates[key]=record
        elif strategy=="most_complete":
            if none_count(record)<none_count(candidates[key]):
                candidates[key]=record

    return [candidates[key] for key in key_order]