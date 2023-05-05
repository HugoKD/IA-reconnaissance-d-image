#classiffication binaire 1er jet

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2
import matplotlib.pyplot as plt
import random as rd 


# couche  de reseau dense
class Layer_Dense:

    def __init__(self, n_inputs, n_neurones):
        self.poids = 0.01 * np.random.randn(n_inputs, n_neurones)
        self.biais = np.zeros((1, n_neurones))
        self.moment = np.zeros_like(self.poids)
        self.bias_moment = np.zeros_like(self.biais)

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
        n, p = np.shape(inputs)
        sigmo = inputs.copy()
        for i in range(n):
            for j in range(p):
                sigmo[i, j] = (1 + math.exp(-sigmo[i, j])) ** (-1)
        self.output = sigmo

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
        acti = self.inputs.copy()
        for i in range(np.shape(acti)[0]):  ##regle de la chaine
            for j in range(np.shape(acti)[1]):
                if acti[i, j] <= 0:
                    acti[i, j] = 0
                else:
                    acti[i, j] = 1
        self.dinputs *= acti


class perte_croisée_d_entropie_binaire:



    def forward(self, prediction, y):
        x_1 = np.clip(prediction, 1e-7, 1 - 1e-7)  ## empeche valeur=inf/0
        x_2 = -y * np.log(x_1) - (1 - y) * np.log(1 - x_1)
        log_loss_binaire = np.mean(x_2, axis=-1)
        data_perte = np.mean(log_loss_binaire)
        return data_perte

    def backward(self, a, y):
        taille_data = len(a)
        output = len(a[0])
        a2 = np.clip(a, 1e-7, 1 - 1e-7)
        self.dinputs = -(y / a2 - (1 - y) / (1 - a2)) / output
        self.dinputs = self.dinputs / taille_data


class descente_de_gradient:  # ajustement des para

    def __init__(self, lr,gamma,moment):
        # hyper_parametres
        self.lr = lr
        self.gamma=gamma
        self.moment = moment

    def para_aams(self, couche):

        if self.moment:
            poids_aa = self.moment * couche.moment - self.lr * couche.der_poids
            couche.moment = poids_aa
            biais_aa = self.moment * couche.bias_moment - self.lr * couche.der_biais
            couche.bias_moment = biais_aa

        else:
            poids_aa = -self.lr * couche.der_poids
            biais_aa = -self.lr * couche.der_biais

        couche.poids += poids_aa
        couche.biais += biais_aa


######Création de RN#########


X=[]
Y=[]


files= os.listdir('dataset/0')

for image in files:

    path = 'dataset/0/{}'.format(image)
    image_tp=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res) #data
    Y.append(0) #label_entrainement


files2= os.listdir('dataset/1')

for image in files2:

    path2 = 'dataset/1/{}'.format(image)
    image_tp2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp2, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res)
    Y.append(1)  # label

#data augmentation


def donnee_augmentation_flip(A):
    for i in range(32):
        j=63-i
        A[:,[i]],A[:,[j]]=A[:,[j]],A[:,[i]]
    return A



x=rd. randint(100,1000)

for i in range(x):
    y=rd.randint(0,len(X))
    B=donnee_augmentation_flip(X[y])
    X.append(B)


X,y=np.array(X), np.array(Y).astype('uint8')
X_test,y_test=np.array(X), np.array(Y).astype('uint8')

#normalisation contre l'overfitting + flatten

X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]).astype(np.float32)/255
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]).astype(np.float32)/255
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

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
activation3 = Sigmoid()
fonction_perte = perte_croisée_d_entropie_binaire()
para_aa = descente_de_gradient(0.2,0.7, 0.9)
PERTE=[]
EPOCH=[]

for i in range(10):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = fonction_perte.forward(activation3.output, y)
    PERTE.append(data_loss)
    EPOCH.append(i)
    predictions = (activation3.output > 0.5) * 1
    precision = np.mean(predictions == y)


    print('itération : {}, precision: {},perte: {}, learning rate: {}'.format(i, precision, data_loss,para_aa.lr))

        ############ backward ###############

    fonction_perte.backward(activation3.output, y)
    activation3.retropagation(fonction_perte.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #####_____ajustement____######
    if i>=2:
        if PERTE[i-1]<=data_loss:
            para_aa.lr*=para_aa.gamma
    para_aa.para_aams(dense1)

    para_aa.para_aams(dense1)
    para_aa.para_aams(dense2)
    para_aa.para_aams(dense3)

plt.plot(EPOCH,PERTE)
plt.show()