import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import get_collection
import platform, pickle

rdm = RandomState(1)

# Hyper-parameters
inputs_dim, featre_dim = 310, 128
labels_dim, domain_dim = 64, 64
labels_num, domain_num = 3, 5

print("Environment: TensorFlow %s, Python %s" % (tf.__version__, platform.python_version()))


def data_load():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def data_fold(data, fold):
    train_key, train_index = list(data.keys()), list(range(domain_num))
    valid_key, _ = train_key.pop(fold), train_index.pop(fold)
    valid_X, valid_y = data[valid_key]['data'], data[valid_key]['label']
    train_X, train_y = np.vstack([data[k]['data'] for k in train_key]), np.hstack([data[k]['label'] for k in train_key])
    valid_d, train_d = np.ones(valid_y.size) * fold, np.repeat(train_index, valid_y.size)
    valid_y, valid_d = to_categorical(valid_y + 1).astype(np.float32), to_categorical(valid_d, num_classes=domain_num).astype(np.float32)
    train_y, train_d = to_categorical(train_y + 1).astype(np.float32), to_categorical(train_d, num_classes=domain_num).astype(np.float32)
    train_X, valid_X = train_X.astype(np.float32), valid_X.astype(np.float32)
    return (train_X, train_y, train_d), (valid_X, valid_y, valid_d)


class Layer:

    def __init__(self, i_dim, o_dim, activate, group='default'):
        self.w = tf.Variable(tf.random.normal([i_dim, o_dim], mean=0.0, stddev=0.1, seed=1))
        tf.compat.v1.add_to_collection(group, self.w)
        self.b = tf.Variable(tf.random.normal([o_dim], mean=0.0, stddev=0.0001, seed=1))
        tf.compat.v1.add_to_collection(group, self.b)
        self.f = activate

    def __call__(self, X):
        return self.f(tf.matmul(X, self.w) + self.b)


class BDNN:
    def __init__(self):
        self.E = [
            Layer(inputs_dim, featre_dim, tf.nn.tanh),
            Layer(featre_dim, featre_dim, tf.nn.sigmoid),
            Layer(featre_dim, labels_dim, tf.nn.tanh),
            Layer(labels_dim, labels_dim, tf.nn.tanh),
            Layer(labels_dim, labels_num, tf.nn.softmax),
        ]

    def forward(self, X):
        tmp = self.E[0](X)
        for i in range(1, len(self.E)):
            tmp = self.E[i](tmp)
        return tmp

    def loss(self, X, y):
        return tf.keras.backend.categorical_crossentropy(target=y, output=self.forward(X))

    def accuracy(self, X, y):
        return tf.keras.metrics.categorical_accuracy(y, self.forward(X))

    def train(self, train_data, valid_data, learning_rate=1e-4, epoch=20, batch_size=32):

        X = tf.compat.v1.placeholder(tf.float32, shape=(None, inputs_dim))
        y = tf.compat.v1.placeholder(tf.float32, shape=(None, labels_num))

        train_X, train_y = train_data
        valid_X, valid_y = valid_data

        var_group = [
            tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss(X, y)),
            tf.reduce_mean(self.loss(X, y)), tf.reduce_mean(self.accuracy(X, y))
        ]

        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()
            for i in range(epoch):
                train_log = []
                for k in range(train_X.shape[0] // batch_size):
                    batch_index = rdm.choice(train_X.shape[0], batch_size)
                    _, los, acc = sess.run(var_group, feed_dict={
                        X: train_X[batch_index], y: train_y[batch_index]})
                    train_log.append([los, acc])
                acc_v = sess.run(tf.reduce_mean(self.accuracy(X, y)), feed_dict={X: valid_X, y: valid_y})
                print("After {:>5d} training_step(s), loss: {:.4f}, train_acc: {:.4f}, valid_acc: {:.4f}"
                      .format(i, *np.mean(train_log, axis=0), acc_v))


class DANN:

    def __init__(self):
        self.E = [
            Layer(inputs_dim, featre_dim, tf.nn.tanh,    group="extrac"),
            Layer(featre_dim, featre_dim, tf.nn.sigmoid, group="extrac"),
        ]
        self.L = [
            Layer(featre_dim, labels_dim, tf.nn.tanh,    group="labels"),
            Layer(labels_dim, labels_dim, tf.nn.tanh,    group="labels"),
            Layer(labels_dim, labels_num, tf.nn.softmax, group="labels"),
        ]
        self.D = [
            Layer(featre_dim, domain_dim, tf.nn.tanh,    group="domain"),
            Layer(domain_dim, domain_dim, tf.nn.tanh,    group="domain"),
            Layer(domain_dim, domain_num, tf.nn.softmax, group="domain"),
        ]

    def classify_labels(self, X):
        tmp = self.E[1](self.E[0](X))
        for i in range(len(self.L)):
            tmp = self.L[i](tmp)
        return tmp

    def classify_domain(self, X):
        tmp = self.E[1](self.E[0](X))
        for i in range(len(self.D)):
            tmp = self.D[i](tmp)
        return tmp

    def loss_labels(self, X, y):
        output = self.classify_labels(X)
        return tf.keras.backend.categorical_crossentropy(target=y, output=output)

    def loss_domain(self, X, d):
        output = self.classify_domain(X)
        return tf.keras.backend.categorical_crossentropy(target=d, output=output)

    def accuracy(self, X, y):
        output = self.classify_labels(X)
        return tf.keras.metrics.categorical_accuracy(y, output)

    def accuracy_adversarial(self, X, d):
        output = self.classify_domain(X)
        return tf.keras.metrics.categorical_accuracy(d, output)

    def train(self, train_data, valid_data, lr_1=1e-4, lr_2=1e-4, epoch=20, batch_size=32, adversarial_rate=0.5):

        X = tf.compat.v1.placeholder(tf.float32, shape=(None, inputs_dim))
        y = tf.compat.v1.placeholder(tf.float32, shape=(None, labels_num))
        d = tf.compat.v1.placeholder(tf.float32, shape=(None, domain_num))

        train_X, train_y, train_d = train_data
        valid_X, valid_y, valid_d = valid_data
        # adversarial_rate *= (lr_1 / lr_2)

        optimizer_1 = tf.compat.v1.train.AdamOptimizer(lr_1)
        optimizer_2 = tf.compat.v1.train.AdamOptimizer(lr_2)
        grad_labels_labels = optimizer_1.compute_gradients(self.loss_labels(X, y), get_collection('labels'))
        grad_labels_extrac = optimizer_1.compute_gradients(self.loss_labels(X, y), get_collection('extrac'))
        grad_domain_domain = optimizer_2.compute_gradients(self.loss_domain(X, d), get_collection('domain'))
        grad_domain_extrac = optimizer_2.compute_gradients(self.loss_domain(X, d), get_collection('extrac'))
        grad_extrac = [(grad_labels - adversarial_rate * grad_domain, var)
                       for (grad_labels, var), (grad_domain, _) in zip(grad_labels_extrac, grad_domain_extrac)]
        update_op = tf.group(
            optimizer_1.apply_gradients(grad_labels_labels),
            optimizer_2.apply_gradients(grad_domain_domain),
            optimizer_1.apply_gradients(grad_extrac),
        )
        update_oq = tf.group(
            optimizer_2.apply_gradients(grad_domain_domain),
            optimizer_1.apply_gradients([
                (- adversarial_rate * grad_domain, var) for grad_domain, var in grad_domain_extrac])
        )

        los_train = tf.reduce_mean(self.loss_labels(X, y))
        acc_train = tf.reduce_mean(self.accuracy(X, y))
        acc_adver = tf.reduce_mean(self.accuracy_adversarial(X, d))
        var_group = [update_op, los_train, acc_train, acc_adver]

        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()
            for i in range(epoch):
                train_log = []
                for k in range(train_X.shape[0] // batch_size):
                    batch_index = rdm.choice(train_X.shape[0], batch_size)
                    _, los, acc, acc_a = sess.run(var_group, feed_dict={
                        X: train_X[batch_index],
                        y: train_y[batch_index],
                        d: train_d[batch_index],
                    })
                    train_log.append([los, acc, acc_a])
                # for k in range(valid_X.shape[0] // batch_size):
                    batch_index = rdm.choice(valid_X.shape[0], batch_size)
                    sess.run(update_oq, feed_dict={
                        X: valid_X[batch_index],
                        d: valid_d[batch_index],
                    })
                acc_v = sess.run(tf.reduce_mean(self.accuracy(X, y)), feed_dict={X: valid_X, y: valid_y})
                print("After {:>5d} training_step(s), loss: {:.4f}, train_acc: {:.4f}, adversarial: {:.4f} valid_acc: {:.4f}"
                      .format(i, *np.mean(train_log, axis=0), acc_v))


if __name__ == '__main__':
    data = data_load()

    for fold in range(domain_num):
        print("Cross Validation: Leave Fold %d Out" % fold)
        (*train_data, _), (*valid_data, _) = data_fold(data, fold)
        BDNN().train(train_data, valid_data, learning_rate=1e-4, epoch=60)
        print()

    for fold in range(domain_num):
        train_data, valid_data = data_fold(data, fold)
        DANN().train(
            train_data, valid_data,
            epoch=60, batch_size=32,
            lr_1=5e-5, lr_2=1e-4, adversarial_rate=0.3
        )
