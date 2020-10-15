def getNormalizedTermFrequency(tokens):
    N = len(tokens)
    frequencyCounts = {}
    for token in tokens:
        if token not in frequencyCounts:
            frequencyCounts[token] = 1
        else:
            frequencyCounts[token] = frequencyCounts[token] + 1
    for count in frequencyCounts:
        frequencyCounts[count] = frequencyCounts[count] / N
    return frequencyCounts

