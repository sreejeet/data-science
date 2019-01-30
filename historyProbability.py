def roulette_probs(history):
    """
    Return the probability of the next roll based on the history.

    Example:
    roulette_probs([1, 3, 1, 5, 1])
    > {1: {3: 0.5, 5: 0.5},
       3: {1: 1.0},
       5: {1: 1.0}
      }
    """
    counts = {}
    for i in range(1, len(history)):
        roll, prev = history[i], history[i-1]
        if prev not in counts:
            counts[prev] = {}
        if roll not in counts[prev]:
            counts[prev][roll] = 0
        counts[prev][roll] += 1

    probs = {}
    for prev, nexts in counts.items():
        total = sum(nexts.values())
        sub_probs = {next_spin: next_count/total
                for next_spin, next_count in nexts.items()}
        probs[prev] = sub_probs
    return probs



history = [1, 3, 1, 5, 1, 7]
print(history)
print(roulette_probs(history))
