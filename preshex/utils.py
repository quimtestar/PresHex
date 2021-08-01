import numpy as np

def randomPermutation(n, a=1):
    if n <= 0:
        return ()
    else:
        p = randomPermutation(n-1,a)
        c = np.random.choice(n,p=a**np.arange(n)*(1-a)/(1-a**n) if a!=1 else None)
        return (c,) + tuple(i if i < c else i + 1 for i in p)

