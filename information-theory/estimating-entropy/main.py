import math
import secrets

import matplotlib.pyplot as plt

outcomes = [0, 1, 2, 3, 4, 5, 6, 7]

def estimate_p(item, rounds):
    seen = 0.0
    for n in range(rounds):
        if secrets.choice(outcomes) == item:
            seen += 1.0
    return seen/rounds

each_n = []
each_H = []

for n in range(1, 1000):
    each_n.append(n)

    probabilities = []
    for item in outcomes:
        probabilities.append(estimate_p(item, n))
    
    h = 0.0
    for p in probabilities:
        if p != 0:
            h -= p * math.log2(p)
    
    each_H.append(h)

plt.plot(each_n, each_H)
plt.xlabel("Number of samples")
plt.ylabel("Estimated H(X)")
plt.show()
