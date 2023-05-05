import numpy as np
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def compute(self, inputs): return np.maximum(0, inputs)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def compute(self, inputs): return 1 / (1 + np.exp(-inputs))

class Optimizer_SGD:
    def __init__(self, learning_rate,decay, momentum):
        self.learning_rate = learning_rate
        self. decay= decay
        self.iterations = 0
        self.momentum = momentum

    def update_params(self, layer):
        weight_updates = self.momentum * layer.weight_momentum - self.learning_rate * layer.dweights
        layer.weight_momentum = weight_updates
        bias_updates = self.momentum * layer.bias_momentum - self.learning_rate * layer.dbiases
        layer.bias_momentum = bias_updates
        layer.weights += weight_updates
        layer.biases += bias_updates

class Loss_BinaryCrossentropy:
    def forward(self, y_pred, y):
        y_p = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y * np.log(y_p) + (1 - y) * np.log(1 - y_p))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        dv = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / dv - (1 - y_true) / (1 - dv)) / len(dv[0])
        self.dinputs = self.dinputs / len(dv)



    def compute(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss



X= np.load('hyper_para/list_image.npy')
y= np.load('hyper_para/list_label.npy')
#normalisation contre l'overfitting + flatten
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]).astype(np.float32)/255
y = y.reshape(-1, 1)
##shuffle
def shuffle(data,label):
    n = np.shape(data)[0]
    for i in range(n):
        k = np.random.randint(i, n)
        data[i], data[k] = data[k], data[i]
        label[i],label[k]=label[k],label[i]
shuffle(X,y)
largeur_image=64
####################################################################

dense1 = Layer_Dense(largeur_image**2, largeur_image**2)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(largeur_image**2, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Sigmoid()
fonction_perte = Loss_BinaryCrossentropy()
para_aa = Optimizer_SGD(0.1, 1e-5, 0.5)
PERTE=[]
PRECISION=[]
EPOCH=[]

taille_des_lots = 32


for i in range(500):
    lot=tuple([np.random.choice(np.arange(len(X)),taille_des_lots)])
    dense1.forward(X[lot])
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = fonction_perte.compute(activation3.output, y[lot])
    A=fonction_perte.forward(activation3.output, y[lot])[:5]




    PERTE.append(data_loss)
    EPOCH.append(i)
    predictions = (activation3.output > 0.5) * 1
    precision = np.mean(predictions == y[lot])
    PRECISION.append(precision)
    print("##########")
    print('itération : {}, precision: {},perte: {}, learning rate: {}'.format(i, precision, data_loss,para_aa.learning_rate))
    print(A.T)
    print('\n')

    ############ backward ###############

    fonction_perte.backward(activation3.output, y[lot])

    activation3.backward(fonction_perte.dinputs)
    dense3.backward(activation3.dinputs)

    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #####_____ajustement____######

    para_aa.learning_rate = para_aa.learning_rate *  (1. / (1. + para_aa.decay * para_aa.iterations))
    para_aa.iterations+=1

    para_aa.update_params(dense1)
    para_aa.update_params(dense2)
    para_aa.update_params(dense3)



plt.plot(EPOCH,PERTE)
plt.show()