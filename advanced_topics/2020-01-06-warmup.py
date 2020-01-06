%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(123)

# What probability distribution would you use to model the scenario outlined above?
# Binomial Distribution.


# Calculate all the requested probabilities.
# Use all the possible combinations of subject count and chance that a subject will stay in the study.
# For example, at first calculate the chance that at least half of the subjects stay in the study if there is a 70% probability that each subject sticks around and there are 10 subjects, then the probality that only one person leaves, then the probability that ll the subjects stay.



stats.binom(10, .70).sf(.5*10)
stats.binom(10, .70).pmf(10-1)
stats.binom(10, .70).pmf(10)

ns = [10,20]
ps = [.7,.8,.9]
scenarios = [n * .5, n - 1, n]

r = []
for p in ps:
    for n in ns:
        l = []
        p1 = stats.binom(n,p).sf(n*.5)
        l.append(p1)
        p2 = stats.binom(n,p).pmf(n-1)
        l.append(p2)
        p3 = stats.binom(n,p).pmf(n)
        l.append(p3)
        r.append(l)

l = []
l.append([1,2])

