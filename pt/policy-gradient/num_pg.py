import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.seterr(all='raise')

def sim_track(s_t, a_t) -> tuple[np.array, float]:
    if a_t == 0:
        s_t1 = s_t + 0.05
    else:
        s_t1 = s_t - 0.05

    return s_t1, s_t**2 - s_t1**2#1 - (s_t1**2)

def softmax(a) -> np.array:
    try:
       e_a = np.power(np.e, a)
       return e_a / e_a.sum()
    except FloatingPointError:
        print(a)
        import pdb; pdb.set_trace()

def P(W, s_t) -> np.array:
    return softmax(W @ s_t)

def grad(W, s_t, a_t, pr_t, eps=0.001) -> np.array:
    g = np.zeros((W.shape[0], W.shape[1]))
    W_0 = W.copy()

    log_pr_t = np.log(pr_t)

    # import pdb; pdb.set_trace()
    for ri in range(W.shape[0]):
        for ci in range(W.shape[1]):
            d = np.zeros(W.shape)
            d[ri,ci] = eps
            pr = P(W + d, s_t)       
            g[ri,ci] = (pr.flatten()[a_t] - pr_t.flatten()[a_t]) / eps
                
    return g / pr_t.flatten()[a_t]

def run(s_t, W, sim=sim_track, epochs=10, stochastic=True):
    S,A,Pr,R = [],[],[],[]
    for t in range(epochs):
        Pr.append(P(W, s_t))
        a = np.argmax(Pr[-1].flatten())
        if stochastic:
            a = np.random.choice(list(range(2)), p=Pr[-1].flatten())

        s_t_1, r_t = sim(s_t, a)
        S.append(s_t.copy()); A.append(a); R.append(r_t[0,0])
        s_t = s_t_1
    return S,A,Pr,R

# W = np.array([[0.1],[-0.1]])
# S_0 = np.array([[-2]]) #np.random.random((1,1))
# Pr = []
# import pdb; pdb.set_trace()
# for i in range(100):
#     pr = P(W, S_0)
#     Pr.append(pr)
#     g = grad(W, S_0, 0, pr)
#     W += g * 0.01
# Pr = np.array(Pr)
# plt.plot(Pr[:,0], label='a_0')
# plt.plot(Pr[:,1], label='a_1')
# plt.title("Prob of action a_0")
# plt.xlabel("Time-step")
# plt.ylabel("Probablity")
# plt.show()
# exit(0)
# ---- Sim stuff below ----
def vis(W, R=[]):
    fig = plt.figure()

    def update(frame, S_t, plot):
        plot[0].set_data([S_t[frame][0,0]], [0])

        return [plot]

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(R)

    # Show cart approaching from left
    S_t,_,_,_ = run(np.array([[-2]]), W, epochs=200, stochastic=False)
    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlim(-10, 10)

    point_plt = ax.plot([S_t[0][0,0]], [0], color='r', marker='o', markersize=5)
    ani1 = animation.FuncAnimation(fig, update, len(S_t), fargs=(S_t, point_plt), interval=50)

    # show cart approaching from right
    S_t,_,_,_ = run(np.array([[2]]), W, epochs=200, stochastic=False)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_xlim(-10, 10)

    point_plt = ax.plot([S_t[0][0,0]], [0], color='r', marker='o', markersize=5)
    ani2 = animation.FuncAnimation(fig, update, len(S_t), fargs=(S_t, point_plt), interval=50)
    plt.show()

W = np.array([[0.1, -0.1]]).T
S_0 = np.random.randn(1,1) * 5
print(W)

vis(W)
# _,_,_,R_0 = run(S_0.copy(), W, epochs=50, stochastic=True)

# show the reward traj before optimizing
# plt.plot(R)
# plt.title("Before Optimization")
# plt.xlabel("Time-step")
# plt.ylabel("Reward")
# plt.show()
# print(W)
a = 0.005
# import pdb; pdb.set_trace()

R = []
# import pdb; pdb.set_trace()
for e in range(500):
    # S = S_0.copy()
    g = W * 0
    t = 0
    r_e = 0
    S_0 = np.random.randn(1,1) * 5
    S_e,A_e,Pr_e,R_e = run(S_0, W, epochs=50)
    for s_t, a_t, pr_t, r_t in zip(S_e, A_e, Pr_e, R_e):
        r_e += r_t * 0.999**t
        g += grad(W, s_t, a_t, pr_t) * (r_t * 0.999**t)
        t += 1
    g /= len(A_e)
    W += g * a
    W /= np.linalg.norm(W)
    R.append(r_e)

print(W)
vis(W, R=R)