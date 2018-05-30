#!/usr/bin/env python3

import numpy as np
from os import listdir
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pcaTo2D(imgs):
    imgs = np.reshape(imgs, (-1, 30*30))
    pca = PCA(n_components=2)
    imgs = pca.fit_transform(imgs)
    return imgs
def displayImage(img):
    plt.imshow(img, cmap='gray', vmin=0,vmax=255)
    plt.show()
def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    path+='/' if path[-1] != '/' else '' 
    for image in imagesList:
        img = misc.imread(path + image)
        loadedImages.append(img)
    return loadedImages

class neural_network(object):
    def __init__(self):
        self.inputSize = 3
        self.hiddenSize = 12
        self.outputSize = 3
        # weights and bias
        #self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W1 = np.random.normal(0, 0.1, (self.inputSize, self.hiddenSize))
        self.B1 = np.full(self.hiddenSize, 0.1)
        #self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        self.W2 = np.random.normal(0, 0.1, (self.hiddenSize, self.outputSize))
        self.B2 = np.full(self.outputSize, 0.1)
    def status(self):
        print("W1: %s" % str(self.W1))
        print("B1: %s" % str(self.B1))
        print("W2: %s" % str(self.W2))
        print("B2: %s" % str(self.B2))
    def forward(self, data):
        # pca image input
        self.X = np.array(data, dtype=float)
        # labels of data
        self.a1 = np.matmul(self.X, self.W1) + self.B1
        self.z1 = self.sigmoid(self.a1)
        self.a2 = np.matmul(self.z1, self.W2) + self.B2
        self.o = self.softmax(self.a2)
        return self.o
    def backpropagation(self, o, y_):
        # 1x3
        delta_output_error = o - y_
        # 10x3
        delta_W2 = self.z1.T * delta_output_error 
        # 1x10 * 1x10 => 1x10
        delta_hidden_error = ((np.matmul(self.W2, delta_output_error.T)) *\
                (self.dsigmoid(self.z1)).T).T
        # 3x1 * 1x10 => 3x10
        delta_W1 = self.X.T * delta_hidden_error 
        # update the weights
        learning_rate = 0.05
        self.W1 = self.W1 - (learning_rate*delta_W1)
#        print(delta_output_error.shape)
#        print(self.B1.shape)
        self.B1 = self.B1 - (learning_rate * 1 * delta_hidden_error)
        self.W2 = self.W2 - (learning_rate*delta_W2)
        self.B2 = self.B2 - (learning_rate * 1 * delta_output_error)
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    def dsigmoid(self, s):  #take sigmoid() as input s
        return s * (1 - s)
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.reshape(e_x.sum(axis=1), (-1, 1))




# load one directory's BMP files to np array imgs
imgs_1 = np.array(loadImages('./Data_Train/Class1/'))
imgs_2 = np.array(loadImages('./Data_Train/Class2/'))
imgs_3 = np.array(loadImages('./Data_Train/Class3/'))
imgs = np.append(imgs_1, imgs_2, axis=0)
imgs = np.append(imgs, imgs_3, axis=0)
# pca 30*30 images to 2 principle components
pca_imgs = pcaTo2D(imgs)
# append the x0 to head of the matrix([x00, x01, x02], [x10, x11, x12].....)
X0 = np.full((pca_imgs.shape[0], 1), 1)
pca_imgs = np.append(X0, pca_imgs, axis=1)
# build the 2-layer NN model with 10 hidden layers
nn = neural_network()

train_bound = 600
data_limit = 1000
for i in range(train_bound):
    o = nn.forward(pca_imgs[i:i+1])
    nn.backpropagation(o, np.array([[1,0,0]]))
    o = nn.forward(pca_imgs[i+1000:i+1000+1])
    nn.backpropagation(o, np.array([[0,1,0]]))
    o = nn.forward(pca_imgs[i+2000:i+2000+1])
    nn.backpropagation(o, np.array([[0,0,1]]))
t=0
f=0
for i in range(train_bound, data_limit):
    o = nn.forward(pca_imgs[i:i+1])
    if(np.argmax(o) == 0):
        t+=1
    else:
        f+=1
    o = nn.forward(pca_imgs[i+1000:i+1000+1])
    if(np.argmax(o) == 1):
        t+=1
    else:
        f+=1
    o = nn.forward(pca_imgs[i+2000:i+2000+1])
    if(np.argmax(o) == 2):
        t+=1
    else:
        f+=1
print(t/(t+f))
print(t)
print(f)
