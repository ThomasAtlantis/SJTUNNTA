import jax, time
import jax.numpy as jnp
from jax import jit, grad, random, value_and_grad, vmap
from jax.experimental import stax, optimizers
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm.notebook import tqdm as tqdm

rand_key = random.PRNGKey(1)

data = jnp.array(list(map(
    lambda _: list(map(float, _)), 
    [line.strip().split() for line in open("data.txt", "r").readlines()]
)))
train_X = data[:,:2].reshape(-1, 2)
train_y = data[:, 2].reshape(-1, 1)

# plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=colors.ListedColormap(["#c21f30", "#1661ab"]))
# plt.show()

class MLQP:
    def __init__(self, optimizer, featre_dim=2, hidden_dim=8, output_dim=1):
        self.opt_init, self.opt_update, self.get_params = optimizer
        self.u1 = random.normal(rand_key, (featre_dim, hidden_dim))
        self.v1 = random.normal(rand_key, (featre_dim, hidden_dim))
        self.b1 = random.normal(rand_key, (hidden_dim,))
        self.u2 = random.normal(rand_key, (hidden_dim, output_dim))
        self.v2 = random.normal(rand_key, (hidden_dim, output_dim))
        self.b2 = random.normal(rand_key, (output_dim,))
        self.params = [
            self.u1, self.v1, self.b1,
            self.u2, self.v2, self.b2,
        ]
        self.opt_state = self.opt_init(self.params)

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    def tanh(self, x):
        return jnp.tanh(x)

    def __call__(self, params, x):
        t = self.tanh(   jnp.square(x) @ params[0] + x @ params[1] + params[2])
        s = self.sigmoid(jnp.square(t) @ params[3] + t @ params[4] + params[5])
        return s
    
    def loss(self, params, x, y):
        preds = self(params, x)
        return -jnp.mean(y * jnp.log(preds) + (1-y) * jnp.log(1-preds))

    def accuracy(self, params, x, y):
        return jnp.mean(jnp.abs(self(params, x) - y) < 0.5)
    
    def loss_grad(self, x, y):
        return grad(self.loss)(self.params, x, y)
    
    def train(self, x, y, num_epochs=10):
        log_train_acc = [self.accuracy(self.params, x, y)]
        log_train_loss= [self.loss(self.params, x, y)]
        for epoch in range(num_epochs):
            start_time = time.time()
            for _x, _y in zip(x, y):
                grads = jit(self.loss_grad)(_x.reshape(1, 2), _y.reshape(1, 1))
                self.opt_state = self.opt_update(0, grads, self.opt_state)
                self.params = self.get_params(self.opt_state)
            epoch_time = time.time() - start_time
            log_train_acc.append(self.accuracy(self.params, x, y))
            log_train_loss.append(self.loss(self.params, x, y))
            print("EPOCH {:>2d} | TIME: {:0.2f} | ACC: {:0.3f} | LOSS: {:0.3f}".format(
                epoch+1, epoch_time, log_train_acc[-1], log_train_loss[-1]))

model = MLQP(optimizer=optimizers.adam(step_size=0.003))
model.train(train_X, train_y, num_epochs=100)
