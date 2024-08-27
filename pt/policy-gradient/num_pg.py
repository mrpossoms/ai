import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.seterr(all='raise')

def sim_track(s_t, a_t) -> tuple[np.array, float]:
    a = np.random.choice(list(range(2)), p=a_t.flatten())
    if a == 0:
        s_t1 = s_t + 0.1
    else:
        s_t1 = s_t - 0.1

    return s_t1, 1-np.abs(s_t1)

def softmax(a) -> np.array:
    try:
       e_a = np.power(np.e, a)
       return e_a / e_a.sum()
    except FloatingPointError:
        print(a)
        import pdb; pdb.set_trace()

def P(W, s_t) -> np.array:
    return softmax(W @ s_t)

def grad(W, s_t, a_t, eps=0.1) -> np.array:
    g = np.zeros((a_t.size, W.shape[0], W.shape[1]))
    W_0 = W.copy()

    # import pdb; pdb.set_trace()
    for ai in range(a_t.size):
        for ri in range(W.shape[0]):
            for ci in range(W.shape[1]):
                wi_0 = W[ri,ci]
                W[ri,ci] += eps
                a = P(W, s_t)
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
        a = P(W, s_t)
        s_t_1, r_t = sim(s_t, a)
        S.append(s_t.copy()); A.append(a); R.append(r_t[0,0])
        s_t = s_t_1
    return S,A,R

W = np.array([[0.1],[-0.1]])
S_0 = np.array([[-2]]) #np.random.random((1,1))
S,A,R = run(S_0.copy(), W)

# show the reward traj before optimizing
plt.plot(R)
plt.title("Before Optimization")
plt.xlabel("Time-step")
plt.ylabel("Reward")
plt.show()
print(W)
a = 0.001
import pdb; pdb.set_trace()
for e in range(100):
    # S = S_0.copy()
    g = W * 0
    for s_t, a_t, r_t in zip(*run(np.random.random((1,1)), W)):
        g += grad(W, s_t, a_t) * -r_t
    # g /= len(S)
    print('----')
    print(g)
    W += g * a
    W /= W.max()
    
print(W)
_,_,R = run(S_0.copy(), W)
plt.title("After Optimization")
plt.xlabel("Time-step")
plt.ylabel("Reward")
plt.plot(R)
plt.show()