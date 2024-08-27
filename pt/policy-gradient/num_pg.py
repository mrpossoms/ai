import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sim_track(s_t, a_t) -> tuple[np.array, float]:
    s_t1 = s_t + (a_t[0] - a_t[1])
    return s_t1, 1-np.abs(s_t1)

def softmax(a) -> np.array:
	e_a = np.power(np.e, a)
	return e_a / e_a.sum()

def grad(W, s_t, a_t, eps=0.000001) -> np.array:
    g = np.zeros((a_t.size, W.shape[0], W.shape[1]))
    W_0 = W.copy()

    # import pdb; pdb.set_trace()
    for ai in range(a_t.size):
        for ri in range(W.shape[0]):
            for ci in range(W.shape[1]):
                wi_0 = W[ri,ci]
                W[ri,ci] += eps
                a = softmax(W @ s_t)
                # print(f'{ai},{ri},{ci} =====')
                # print(f'wi_0: {wi_0}, wi_d: {W[ri,ci]}')
                # print('action before')
                # print(a_t)
                # print('action after')
                # print(a)
                
                g[ai][ri,ci] = (a.flatten()[ai] - a_t.flatten()[ai]) / eps
                W = W_0.copy()
                
    return g.sum(axis=0)/a_t.size

def run(s_t, W, sim=sim_track, epochs=10):
    S,A,R = [],[],[]
    for t in range(epochs):
        a = softmax(W @ s_t)
        s_t_1, r_t = sim(s_t, a)
        S.append(s_t.copy()); A.append(a); R.append(r_t[0,0])
        s_t = s_t_1
    return S,A,R

W = np.array([[1],[-0.1]])
S_0 = np.array([[-2]]) #np.random.random((1,1))
S,A,R = run(S_0.copy(), W)

# show the reward traj before optimizing
plt.plot(R)
plt.title("Before Optimization")
plt.xlabel("Time-step")
plt.ylabel("Reward")
plt.show()
print(W)
a = 100000000
import pdb; pdb.set_trace()
for e in range(1):
    S = S_0.copy()
    g = W * 0
    for s, a, r in zip(*run(S.copy(), W)):
        g += grad(W, s, a) * r
    # g /= len(S)
    print('----')
    print(g)
    W += g * a
    # W /= W.max()
    
print(W)
_,_,R = run(S_0.copy(), W)
plt.title("After Optimization")
plt.xlabel("Time-step")
plt.ylabel("Reward")
plt.plot(R)
plt.show()