import numpy as np
import matplotlib.pyplot as plt

dataset_and = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

dataset_or = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]

dataset_xor = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

dataset_xnor = [
    ([0, 0], 1),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

def generate_batch(dataset, batch_size):
    inputs = np.vstack([ex[0] for ex in dataset])
    targets = np.vstack([ex[1] for ex in dataset])
    rand_indexes = np.random.randint(0, len(dataset), batch_size)
    return inputs[rand_indexes], targets[rand_indexes]

number_of_examples = 100
inputs_and, targets_and = generate_batch(dataset_and, number_of_examples)
inputs_or, targets_or = generate_batch(dataset_or, number_of_examples)
inputs_xor, targets_xor = generate_batch(dataset_xor, number_of_examples)
inputs_xnor, targets_xnor = generate_batch(dataset_xnor, number_of_examples)

def sigmoid(a):
    return 1.0/(1.0 + np.exp(-a))

def sigmoid_prime(a):
    s = sigmoid(a)
    return np.multiply(s, np.subtract(1, s))

class nn_one_layer():
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        self.W2 = 0.1 * np.random.randn(hidden_size, output_size)

        self.f = sigmoid
    
    def forward(self, u):
        z = np.matmul(u, self.W1)
        h = self.f(z)
        v = np.matmul(h, self.W2)
        return v, h, z

def loss_mse(preds, targets):
    return np.sum((preds - targets)**2) * 0.5

def loss_deriv(preds, targets):
    return preds - targets

def backprop(W1, W2, dL_dPred, U, H, Z):
    dL_W2 = np.matmul(H.T, dL_dPred)
    dL_dH = np.matmul(dL_dPred, W2.T)
    dL_dZ = sigmoid_prime(Z) * dL_dH
    dL_W1 = np.matmul(U.T, dL_dZ)
    return dL_W1, dL_W2

def train_one_batch(nn, dataset, batch_size, lr):
    inputs, targets = generate_batch(dataset, batch_size)
    preds, H, Z = nn.forward(inputs)

    loss = loss_mse(preds, targets)
    dL_dPred = loss_deriv(preds, targets)

    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= lr * dL_dW1
    nn.W2 -= lr * dL_dW2

    return loss

def test(nn, dataset, batch_size):
    inputs, targets = generate_batch(dataset, batch_size)
    preds, _, _ = nn.forward(inputs)
    loss = loss_mse(preds, targets)
    return loss

input_size = 2
hidden_size = 5
output_size = 1

nn = nn_one_layer(input_size, hidden_size, output_size)

# train the xor dataset

batch_size = 5 # examples per batch
number_of_batches = 5000
lr = 0.1 # learning rate

losses = []
for i in range(number_of_batches):
    losses.append(train_one_batch(nn, dataset_xor, batch_size=batch_size, lr=lr))

plt.plot(np.arange(1, number_of_batches+1), losses)
plt.xlabel("# batches")
plt.ylabel("training MSE")
plt.show()

# test the network on the xor dataset

preds_xor, _, _ = nn.forward(inputs_xor)

_, inds = np.unique(inputs_xor, return_index=True, axis=0)

fig = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2)
plt.scatter(targets_xor[inds], preds_xor[inds], marker="x", c="black")
for i in inds:
    coord = f"({inputs_xor[i][0]}, {inputs_xor[i][1]})"
    xoffset = 0.05 if targets_xor[i] == 0 else -0.1
    yoffset = 0.003 if preds_xor[i] > np.mean(preds_xor[inds]) else -0.005
    plt.text(targets_xor[i] + xoffset, preds_xor[i] + yoffset, coord)
plt.xlabel("Target values")
plt.ylabel("Predicted values")
plt.ylim([np.min(preds_xor) - 0.01, np.max(preds_xor) + 0.01])
ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
plt.hist(targets_xor, color="blue")
ax2.set_title("target_values")
plt.ylabel("# in batch")
ax3 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1, sharey=ax2)
plt.hist(preds_xor, color="red")
ax3.set_title("predicted values")

fig.tight_layout()
plt.show()

# test the network on the other datasets

dataset_names = ["AND", "OR", "XOR", "XNOR"]
test_scores = [test(nn, dataset, batch_size=batch_size) for dataset in [dataset_and, dataset_or, dataset_xor, dataset_xnor]]

x = range(4)
plt.bar(x, test_scores)
plt.xticks(x, dataset_names, rotation="vertical")
plt.ylabel("test MSE")
plt.show()
