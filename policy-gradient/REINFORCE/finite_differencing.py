import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.seterr(all='raise')

class Environment:
    class Track:
        def __init__(self):
            self.state = self.make_state()

        @property
        def epochs(self) -> int:
            return 6000

        @property
        def state_action_size(self) -> tuple[int,int]:
            return 1, 2

        def make_state(self, x=None, sigma=1):
            if x is not None:
                return np.array([[x]])
            else:
                return np.random.randn(1,1) * sigma

        def step(self, action:int) -> tuple[np.array, float]:
            s_t = self.state
            if action == 0:
                self.state = s_t + 0.05
            else:
                self.state = s_t - 0.05

            return self.state, s_t[0,0]**2 - self.state[0,0]**2

    class Cart:
        def __init__(self):
            self.state = self.make_state()

        @property
        def epochs(self) -> int:
            return 20_000

        @property
        def state_action_size(self) -> tuple[int,int]:
            return 2, 2

        def make_state(self, x=None, sigma=1):
            if x is not None:
                return np.array([[x], [0]])
            else:
                return np.random.randn(2,1) * sigma

        def step(self, action:int) -> tuple[np.array, float]:
            s_t = self.state
            stm = np.array([
                [1, 0.1],
                [0,   1],
            ])
            self.state = stm @ s_t

            if action == 0:
                self.state[0,0] = s_t[0,0] + 0.05
            else:
                self.state[1,0] = s_t[1,0] - 0.05

            return self.state, s_t[0,0]**2 - self.state[0,0]**2

def track_policy_init() -> np.array:
    return np.random.randn(2,1) * 0.1

def cart_policy_init() -> np.array:
    return np.random.randn(2,2) * 0.1

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

def run(W, env, epochs=10, stochastic=True):
    S,A,Pr,R = [],[],[],[]
    for t in range(epochs):
        Pr.append(P(W, env.state))
        a = np.argmax(Pr[-1].flatten())
        if stochastic:
            a = np.random.choice(list(range(2)), p=Pr[-1].flatten())
        S.append(env.state.copy());
        s_t_1, r_t = env.step(a)
        A.append(a); R.append(r_t)
    return S,A,Pr,R

# ---- Sim stuff below ----
def vis(W, env, R=[]):
    fig = plt.figure()

    def update(frame, S_t, plot):
        plot[0].set_data([S_t[frame][0,0]], [0])

        return [plot]

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(R)

    # Show cart approaching from left
    env.state = env.make_state(-2)
    S_t,_,_,_ = run(W, env, epochs=200, stochastic=False)
    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlim(-10, 10)

    point_plt = ax.plot([S_t[0][0,0]], [0], color='r', marker='o', markersize=5)
    ani1 = animation.FuncAnimation(fig, update, len(S_t), fargs=(S_t, point_plt), interval=50)

    # show cart approaching from right
    env.state = env.make_state(2)
    S_t,_,_,_ = run(W, env, epochs=200, stochastic=False)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_xlim(-10, 10)

    point_plt = ax.plot([S_t[0][0,0]], [0], color='r', marker='o', markersize=5)
    ani2 = animation.FuncAnimation(fig, update, len(S_t), fargs=(S_t, point_plt), interval=50)
    plt.show()

def train(env=Environment.Track(), policy_param_init=track_policy_init):
    W = policy_param_init()
    env.state = env.make_state(sigma=5)
    print(W)

    vis(W, env)
    a = 0.0001

    R = []
    for e in range(env.epochs):
        # S = S_0.copy()
        env.state = env.make_state(sigma=5)
        g, t, r_e = W * 0, 0, 0
        S_e,A_e,Pr_e,R_e = run(W, env, epochs=50, stochastic=True)
        for s_t, a_t, pr_t, r_t in zip(S_e, A_e, Pr_e, R_e):
            r_e += r_t * 0.999**t
            g += grad(W, s_t, a_t, pr_t) * (r_t * 0.999**t)
            t += 1
        g /= len(A_e)
        W += g * a

        R.append(r_e)
        if e % 1000 == 0:
            print(f"{e}/{env.epochs} Epoch: {e}, Reward: {np.mean(R[-1000:])}")

    print(W)
    vis(W, env, R=np.convolve(R, np.ones(200)/200, mode='valid'))

if __name__ == '__main__':
    train(Environment.Cart(), policy_param_init=cart_policy_init)


def test_convergence():
    W = np.array([[0.1],[-0.1]])
    env = Environment.Track()
    env.state = env.make_state(-2)

    pr0 = P(W, env.state)
    a0 = np.argmax(pr0).flatten()
    _,r0 = env.step(a0)

    W += grad(W, env.state, a0, pr0) * r0

    pr1 = P(W, env.state)
    a1 = np.argmax(pr1).flatten()
    _,r1 = env.step(a1)
    a1 = np.argmax(pr1).flatten()

    assert(r1 > r0)

def test_gradient():
    W = np.array([[0.1],[-0.1]])
    S_0 = np.array([[-2]]) #np.random.random((1,1))
    Pr = []
    for i in range(100):
        pr = P(W, S_0)
        Pr.append(pr)
        g = grad(W, S_0, 0, pr) * 0.005
        W += g * 0.01
    Pr = np.array(Pr)

    assert(Pr[0,0] < Pr[-1,0])

    # plt.plot(Pr[:,0], label='a_0')
    # plt.plot(Pr[:,1], label='a_1')
    # plt.title("Prob of action a_0")
    # plt.xlabel("Time-step")
    # plt.ylabel("Probablity")
    # plt.show()
