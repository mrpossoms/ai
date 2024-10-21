import numpy as np
from numpy import array as arr
from numpy import argmax
from numpy.random import multinomial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
from env import *


warnings.filterwarnings("error", category=DeprecationWarning)
np.seterr(all='raise')

def Pi(W: [np.ndarray], s_t: np.ndarray) -> np.array:
    assert(isinstance(W, list))
    assert(isinstance(W[0], np.ndarray))
    return softmax(W[0] @ s_t).flatten()

def Tau(W: [np.ndarray], s_t: np.ndarray) -> np.ndarray:
    assert(isinstance(W, list))
    assert(isinstance(W[0], np.ndarray))
    a = W[0] @ s_t
    return a[:1], np.ln(np.exp(a[1:]))

def Pr_a_discrete(W: [np.ndarray], Pi: callable, s_t: np.ndarray, a_t: np.ndarray):
    '''
    The value that this function needs to compute is; how likley the choice of action a_t was
    given the distribution parameters or action probabilities. For a discrete action space, passed through
    a softmax function, the value at the action index can be used directly assuming the distribution is weighted
    by these values
    '''
    return Pi(W, s_t).flatten()[int(a_t[0])]


def Pr_a_gaussian(W: [np.ndarray], Pi: callable, s_t: np.ndarray, a_t: np.ndarray):
    mu = sig_2
    two_sig_sq = 2 * a_t[:2] ** 2

    pr = (1 / np.sqrt(np.pi * two_sig_sq)) * np.exp(-((mu-a_t)**2)/two_sig_sq)

    return pr.prod()


def grad(W: [np.ndarray], s_t: np.ndarray, a_t: np.ndarray, Pi:callable, Pr_a:callable, eps=0.001) -> [np.ndarray]:
    '''
    Compute the gradient of the policy with respect to the parameters W using finite differencing
    
    Parameters:
    W: np.array
        The policy parameters
    s_t: np.array
        The input state at time t
    a_t: np.array
        The action chosen for time t
    Pi: Callable
        The policy function which calculates the probability or distribution parameters for all possible
        actions that policy Pi could take
    eps: float, default 0.001
        The finite differencing step size applied to each parameter
    '''
    assert(isinstance(W, list))
    assert(isinstance(W[0], np.ndarray))
    assert(isinstance(s_t, np.ndarray))
    assert(isinstance(a_t, np.ndarray))

    G = [P * 0 for P in W]
    pr_t = Pr_a(W, Pi, s_t, a_t)

    for i, (P, g) in enumerate(zip(W, G)):
        for ri in range(P.shape[0]):
            for ci in range(P.shape[1]):
                d = np.zeros(P.shape)
                d[ri,ci] = eps
                P += d
                # log_pr = np.log(Pi(W, s_t)).sum()
                pr = Pr_a(W, Pi, s_t, a_t)
                P -= d      
                g[ri,ci] = (pr - pr_t) / eps    

    return G

def test_convergence_discrete():
    W = [np.array([[0.1],[-0.1]])]
    x0 = arr([-2])

    pr_a_0 = Pi(W, x0)

    for i in range(3):
        pr_a_t = Pi(W, x0)
        a_t = arr([argmax(multinomial(1, pr_a_t))])
        pr = Pr_a_discrete(W, Pi, x0, a_t)
        r = -1 if a_t[0] == 0 else 1

        g = grad(W, x0, a_t, Pi, Pr_a_discrete)
        # print(f'g: {g}')

        for Wi, gi in zip(W, g):
            Wi += gi * r

        print(f'r: {r} pr: {pr.flatten()} probs: {pr_a_t}')

    pr_a_n = Pi(W, x0)

    assert(pr_a_n[1] > pr_a_0[1])

def test_convergence_continuous():
    W = [np.array([[0.1],[-0.1]])]
    x0 = arr([-2])

    pr_a_0 = Tau(W, x0) # used for checking optimization below

    for i in range(3):
        pr_a_t = Tau(W, x0)

        a_t = arr(np.random.normal(pr_a_t[0], pr_a_t[1]))

        pr_0 = Pr_a_gaussian(W, Tau, x0, a_t)
        

        g = grad(W, x0, a_t, Tau, Pr_a_discrete)
        # print(f'g: {g}')

        for Wi, gi in zip(W, g):
            Wi += gi * r

        print(f'r: {r} pr: {pr.flatten()} probs: {pr_a_t}')

    pr_a_n = Tau(W, x0)

    assert(pr_a_n[1] > pr_a_0[1])

def test_gradient():
    W = np.array([[0.1],[-0.1]])
    S_0 = np.array([[-2]]) #np.random.random((1,1))
    Pr = []
    for i in range(100):
        pr = P(W, S_0)
        Pr.append(pr)
        g = grad([W], S_0) * 0.005
        W += g * 0.01
    Pr = np.array(Pr)

    assert(Pr[0,0] < Pr[-1,0])

    # plt.plot(Pr[:,0], label='a_0')
    # plt.plot(Pr[:,1], label='a_1')
    # plt.title("Prob of action a_0")
    # plt.xlabel("Time-step")
    # plt.ylabel("Probablity")
    # plt.show()

if __name__ == '__main__':
    def train_cart():
        train(Environment.Cart(), policy_param_init=cart_policy_init)

    funcs = {
        'train_cart': train_cart,
        'test_convergence_discrete': test_convergence_discrete,
        'test_gradient': test_gradient,
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, default='train_cart', help="Function to run options: " + ', '.join(funcs.keys()))

    args = parser.parse_args()
    funcs[args.function]()
