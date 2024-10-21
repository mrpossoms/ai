import numpy as np

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
