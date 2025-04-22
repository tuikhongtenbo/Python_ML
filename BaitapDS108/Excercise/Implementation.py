import numpy as np
import math
import matplotlib.pyplot as plt


x = np.random.uniform(-5, 5, 100000)
K = ['rectangular', 'triangular', 'parabolic', 'biweigh', 'gaussian', 'silverman']

def I(x):
    Ix = []
    for i in x: 
        if i >= -1 and i <= 1:
            Ix.append(1)
        else:
            Ix.append(0)
    return Ix

def visualize(Ix):
    plt.figure(figsize=(8,6), dpi=150)
    plt.plot(Ix, color='r')

def choose(I, k):
    if k == 'rectangular':
        f = I(x) * 0.5
        print(f)
        visualize(f)
        return

    if k == 'triangular': 
        f = (1 - abs(x)) * I(x)
        print(f)
        visualize(f)
        return
    
    if k == 'parabolic':
        f = 0.75 * (1 - x**2) * I(x)
        print(f)
        visualize(f)
        return
    
    if k == 'biweigh':
        f = 15/16 * ((1 - x**2)**2) * I(x)
        print(f)
        visualize(f)
        return
    
    if k == 'gaussian':
        f = a * math.exp(-0.5 * x**2)
        a = 1 / math.sqrt(2*math.pi)
        print(f)
        visualize(f)
        return
    
    if k == 'silverman':
        a = abs(x) / math.sqrt(2) + math.pi / 4
        b = -abs(x) / math.sqrt(2)
        f = 0.5 * math.exp(b) * math.sin(a) * I(x)
        print(f)
        visualize(f)
        return

Ix = I(x)
print(Ix)

choose(Ix, 'silverman')