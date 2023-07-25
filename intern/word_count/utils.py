def word_count(batch, count = None):
    """
    """
    count = count or {}
    for text in batch:
        if not hasattr(text, "hash"):
            text = " ".join(text)
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count