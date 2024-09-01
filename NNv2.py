import numpy as np
import random

class NeuralNetwork2:
    def __init__(self, max_iter=1000, alpha = 0.001) -> None:
        self.W = 0
        self.b = np.float32(random.random())
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_features = 0

    def preactivation(self, X):
        val = []
        for i in range(0,len(X[0])):
            sum = 0
            for j in range(0,len(X)):
                sum += X[j][i] * self.W[j]
            sum += self.b 
            val.append(sum)
        return np.array(val)

    def sigmoid(self,val):
        sig_val = 1 / (1 + np.exp(-val))
        return sig_val

    def model_out(self, sig_val):
        out = np.where(sig_val > 0.5, 1, 0)
        return out

    def calc_loss(self, y, sig_val):
        error = sig_val - y
        loss = np.sum(np.square(error))
        return loss

    def calc_gradient(self, y, sig_val, X):
        grad_W = np.zeros(self.n_features)
        error = sig_val - y
        for i in range(0,len(grad_W)):
            grad_W[i] = np.dot(error, X[i])
        grad_b = np.sum(error)/len(y)
        return grad_W, grad_b

    def update_weights(self, grad_W, grad_b):
        for i in range(0,len(self.W)):
            self.W[i] = self.W[i] - (self.alpha*grad_W[i])
        self.b = self.b - (self.alpha*grad_b)

    def fit(self, X, y):
        self.n_features = len(X)
        self.W = np.array([random.random() for i in range(0,self.n_features)])
        for i in range(self.max_iter+1):
            sig_val = self.sigmoid(self.preactivation(X))
            y_out = self.model_out(sig_val)
            loss_val = self.calc_loss(y,sig_val)
            if i%10000 == 0:
                print("Loss Value: ",np.mean(loss_val)," Iteration: ",i)
                print("Weights ",self.W)
            grad_W, grad_b = self.calc_gradient(y, sig_val, X)
            if i%10000 == 0:
                print("Gradient ",grad_W, "Bias ",grad_b)
            self.update_weights(grad_W, grad_b)

    def predict(self, X):
        return self.model_out(self.sigmoid(self.preactivation(X)))