import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

"""
    this part uses two methods to min the cost function
        1. scipy.optimize.fmin_tnc
        2. gradient_decent
"""

def get_data():
    file_name = './data2.txt'
    lines = np.loadtxt(file_name, dtype=str, delimiter=',')
    lines = lines.astype(float)

    x, y = np.hsplit(lines, [2])
    return x, y

def data_normalize(data):
    mean_num = np.mean(data, axis=0)
    mu = np.std(data, axis=0)
    data = (data - mean_num)/mu

    return data

def sigmoid(z):
    result = 1/(1+np.exp(-z))
    return result

def cost_fuc(theta, x, y):
    m = x.shape[0]
    theta = theta.reshape(x.shape[1],1)

    h_x = sigmoid(np.matmul(x, theta))

    cost = -1/m * sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
    grad = 1/m * sum((h_x.reshape(m,1) - y) * x).reshape(-1,1)

    return cost,grad


def gradient_decent(x, y, theta, rate=0.01, steps=50000, plot = False):
    #  theta = np.random.randint(3,size=(n+1,1))
    theta_iter = []

    for i in range(steps):
        print(f"training{i}.........")
        cost, grad = cost_fuc(theta, x, y)
        theta = theta - rate * grad

        # theta_iter for plot
        theta_iter.append(theta)

    if plot:
        plot_acc(x, y, theta_iter)

    return theta

def plot_acc(x, y, theta_list):
    m = x.shape[0]
    acc_val = np.array([])

    for theta in theta_list:
        h_x = sigmoid(np.matmul(x, theta))
        count = 0
        for j in range(len(x)):
            h_x[j] = 0 if h_x[j] < 0.5 else 1
            count += 1 if h_x[j] == y[j] else 0

        acc_val = np.append(acc_val, count/len(y))
    print(acc_val)

    plot_x = np.array(range(len(theta_list)))
    plt.plot(plot_x, acc_val)
    plt.ylim([0,1])
    plt.show()

def plot_result(x, y, result, degree):

    x1 = np.linspace(-1, 1, 100000).reshape(100000, 1)
    x2 = np.linspace(-1, 1, 100000).reshape(100000, 1)
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    x_plot = np.c_[x1, x2]
    x_plot = feature_map(x_plot, degree)
    h_x = np.matmul(x_plot, result)
    print(h_x)
    x_plt = []
    for i in range(len(h_x)):
        if np.abs(h_x[i]) < 0.1:
            x_plt.append(x_plot[i])
    x_plt = np.array(x_plt)
    print("***********")
    print(x_plt.shape)
    print(x_plt)

    x1 = np.array([list(x[i][1:]) for i in range(len(y)) if y[i]==0])
    x2 = np.array([list(x[i][1:]) for i in range(len(y)) if y[i]==1])
    plt.scatter(x1[:,0], x1[:,1],c="#00CED1", alpha=0.8)
    plt.scatter(x2[:,0], x2[:,1],c="#DC143C", alpha=0.8)
    plt.scatter(x_plt[:,1], x_plt[:,2],c="b", alpha=0.8, s=20)
    plt.show()

def feature_map(x, degree):
    new_x = np.ones(x.shape[0])
    for i in range(1, degree+1):
        for j in range(i+1):
            new_col = np.power(x[:, 0], i-j) * np.power(x[:, 1], j)
            new_x = np.c_[new_x, new_col]

    return new_x


def main():

    x, y = get_data()
    x = feature_map(x, 4)
    #x = np.c_[np.ones(x.shape[0]), x]
    theta = np.zeros(x.shape[1])
    theta = theta.reshape(x.shape[1],1)

    result = opt.fmin_tnc(func=cost_fuc, x0=theta, args=(x, y))
    print(result)
    plot_result(x, y, result[0], 4)

    #  use gradient_decent
    #  result = gradient_decent(x, y, theta, 0.01, 200000, False)
    #  print(result)
    #  plot_result(x, y, result)


main()




