import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

class Svm(object):
    def __init__(self):
        self.read_data()
        self.plot_data()


    def read_data(self):

        self.file_name = './data1.mat'
        data = scio.loadmat(self.file_name)
        self.x = data['X'][:-1]
        self.y = data['y'][:-1]

    def plot_data(self):
        self.x1 = np.array([self.x[i] for i in range(len(self.x)) if self.y[i] == 0])
        self.x2 = np.array([self.x[i] for i in range(len(self.x)) if self.y[i] == 1])

        plt.scatter(self.x1[:,0], self.x1[:,1], color='blue')
        plt.scatter(self.x2[:,0], self.x2[:,1], color='red')
        plt.show()


if __name__ == '__main__':
    svm_object = Svm()


