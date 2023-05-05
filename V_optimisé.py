# classiffication binaire 1er jet

import numpy as np
import nnfs
import os
import cv2
import matplotlib.pyplot as plt

nnfs.init()


# couche  de reseau dense
class Layer_Dense:

    def __init__(self, n_inputs, n_neurones):
        self.poids = 0.01 * np.random.randn(n_inputs, n_neurones)
        self.biais = np.zeros((1, n_neurones))

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.poids) + self.biais

    # backward pass
    def backward(self, dy):
        self.der_poids = np.dot(self.inputs.T, dy)
        self.der_biais = np.sum(dy, axis=0, keepdims=True)
        self.dinputs = np.dot(dy, self.poids.T)


# Sigmoide class
class Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def retropagation(self, dy):
        self.dinputs = dy * (1 - self.output) * self.output


# ReLU activation
class Activation_ReLU:
    # forward pass

    def forward(self, inputs):
        self.inputs = inputs  # enregistrement pour les derivées partielles
        self.output = np.maximum(0, inputs)  # relu

    def backward(self, dy):
        ##dy dérivé de la fonction de perte par rapport à la output de la focntion d'activtion des neuronnes de la couche suivante (pour bien "propager")
        self.dinputs = dy.copy()
        self.dinputs[self.inputs <= 0] = 0


class perte_croisée_d_entropie_binaire:

    def forward(self, output, y):
        output = np.clip(output, 1e-7, 1 - 1e-7)  ## empeche valeur=inf/0
        log_loss_binaire1 = -(y * np.log(output) + (1 - y) * np.log(1 - output))
        log_loss_binaire2 = np.mean(log_loss_binaire1, axis=-1)

        return log_loss_binaire2, log_loss_binaire1[0:5]

    def backward(self, output, y):
        samples = len(output)
        outputs = len(output[0])
        output2 = np.clip(output, 1e-7, 1 - 1e-7)
        self.dinputs = -(y / output2 - (1 - y) / (1 - output2)) / outputs
        self.dinputs = self.dinputs / samples

    def calculate(self, output, y):  # y = valeur objecftif/label

        sample_pertes, x = self.forward(output, y)
        data_perte = np.mean(sample_pertes)
        return data_perte, x


class descente_de_gradient:  # ajustement des para

    def __init__(self, learning_rate, decay, moment):
        # hyper_parametres
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.moment = moment

    def para_aams(self, couche):

        if self.moment:
            if hasattr(couche,
                       'moment') == False:  # si l'objet couche n'a pas d'attribut nomé weight moment alors on le créer rempli de zeros
                couche.moment = np.zeros_like(couche.poids)
            couche.bias_moment = np.zeros_like(couche.biais)
            poids_aa = self.moment * couche.moment - self.learning_rate * couche.der_poids
            couche.moment = poids_aa
            biais_aa = self.moment * couche.bias_moment - self.learning_rate * couche.der_biais
            couche.bias_moment = biais_aa

        else:
            poids_aa = -self.learning_rate * couche.der_poids
            biais_aa = -self.learning_rate * couche.der_biais

        couche.poids += poids_aa
        couche.biais += biais_aa


######Création de RN#########


X = []
Y = []

files = os.listdir('data_set/0')

for image in files:
    path = 'data_set/0/{}'.format(image)
    image_tp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res)  # data
    Y.append(0)  # label_entrainement
files2 = os.listdir('data_set/1')
for image in files2:
    path2 = 'data_set/1/{}'.format(image)
    image_tp2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp2, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res)
    Y.append(1)  # label


##shuffle
def shuffle(data, label):
    n = np.shape(data)[0]
    for i in range(n):
        k = np.random.randint(i, n)
        data[i], data[k] = data[k], data[i]
        label[i], label[k] = label[k], label[i]


X, y = np.array(X), np.array(Y).astype('uint8')
X_test, y_test = np.array(X), np.array(Y).astype('uint8')

# normalisaton contre l'overfitting + flatten

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2]).astype(np.float32) / 255
y = y.reshape(-1, 1)

shuffle(X, y)

largeur_image = 64

dense1 = Layer_Dense(largeur_image ** 2, largeur_image ** 2)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(largeur_image ** 2, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Sigmoid()
fonction_perte = perte_croisée_d_entropie_binaire()
para_aa = descente_de_gradient(0.3, 1e-2, 0.9)
PERTE = []
EPOCH = []

for i in range(100):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss, taille = fonction_perte.calculate(activation3.output, y)
    print(taille)
    PERTE.append(data_loss)
    EPOCH.append(i)
    predictions = (activation3.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    print('itération : {}, precision: {},perte: {}, learning rate: {}'.format(i, accuracy, data_loss,
                                                                              para_aa.learning_rate))

    ############ backward ###############

    fonction_perte.backward(activation3.output, y)
    activation3.retropagation(fonction_perte.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #####_____ajustement____######
    para_aa.learning_rate = para_aa.learning_rate * (
            1. / (1. + para_aa.decay * para_aa.iterations))
    para_aa.iterations += 1
    para_aa.para_aams(dense1)
    para_aa.para_aams(dense2)
    para_aa.para_aams(dense3)

plt.plot(EPOCH, PERTE)
plt.show()