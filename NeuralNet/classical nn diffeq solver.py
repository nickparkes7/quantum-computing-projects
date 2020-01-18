import autograd.numpy as np
from autograd import grad
from autograd.scipy.special import expit
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Classical
    Neural
    Network
    representation
    with one inner layer of an
    arbitrary
    height
    """

    def __init__(self, func, inner_layer_size, learn_rate):
        self.func = func
        self.W = np.array([np.random.randn(1, inner_layer_size),
                  np.random.randn(inner_layer_size, 1)])
        self.learn_rate = learn_rate 

    def run_net(self, x):
        hidden = expit(np.dot(x.reshape(10, 1), self.W[0]))
        return np.dot(hidden, self.W[1])

    def d_net_dx(self, x, k):
        return np.dot(np.dot(self.W[1].T, self.W[0].T**k),
                      expit(x) * (1 - expit(x)))

    def loss_function(self, x):
        loss_sum = 0.
        for x_i in x:
            net = self.run_net(x)[0][0]
            psy_t = 1. + x_i * net
            d_net = self.d_net_dx(x_i, 1)[0][0]
            d_psy_t = net * x_i * d_net
            function = self.func(x_i, psy_t)
            err_sqr = (d_psy_t - function)**2

            loss_sum += err_sqr
        return loss_sum

    def train(self, x, batch):
        err = np.zeros(batch)
        for i in range(batch):
            err[i] = self.loss_function(x)
            loss_grad = grad(self.loss_function)(x)
            self.W[0] = self.W[0] - self.learn_rate * loss_grad[0]
            self.W[1] = self.W[1] - self.learn_rate * loss_grad[1]
        return err


if __name__ == "__main__":

    def a(x):

        # Left part of initial equation

        return x + (1. + 3.*x**2) / (1. + x + x**3)


    def b(x):

        # Right part of initial equation

        return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))


    def f(x, psy):

        # d(psy)/dx = f(x, psy)
        # This is f() function on the right

        return b(x) - psy * a(x)


    def psy_analytic(x):

        # Analytical solution of current problem

        return (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2


    nn = NeuralNetwork(f, 10, 0.01)

    x_space = np.linspace(0, 1, 10)
    y_analytical = psy_analytic(x_space)
    
    batch = 1000
    sqr_err = nn.train(x_space, batch)
    err_space = range(0, batch)
    
    y_nn = nn.run_net(x_space)

    plt.figure(0)
    plt.plot(x_space, y_analytical, label="actual")
    plt.plot(x_space, y_nn, label="network")
    plt.legend()
    plt.show()
    
    plt.figure(1)
    plt.plot(err_space, sqr_err)
    plt.show()
