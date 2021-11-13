import numpy as np

def get_data():
    file_name = './lr_data.txt'
    lines = np.loadtxt(file_name,dtype=str, delimiter=',')
    lines = lines.astype(int)

    x,y = np.hsplit(lines, [2])
    return x, y

def data_normalize(data):
    mean_num = np.mean(data, axis=0)
    mu = np.std(data, axis=0)
    data = (data-mean_num)/mu

    return data

def cost_fuc(x, y, theta):
    # add '1' col to x
    m = x.shape[0]
    x = np.c_[np.ones(m), x]

    cost = 0.5 / m * np.square(np.matmul(x, theta)-y).sum()
    grad = 1 / m * ((np.matmul(x, theta)-y)*x).sum(axis=0).reshape(-1,1)

    return cost, grad

def gradient_decent(x, y, rate=0.1, steps=500):

    theta = np.random.randint(5, size=(x.shape[1]+1,1))

    for i in range(steps):
        print(f"Training{i}........")
        cost, grad = cost_fuc(x, y, theta)
        theta = theta - rate * grad

    return theta

def predict(x_in, data, theta):

    mean_num = np.mean(data, axis=0)
    mu = np.std(data, axis=0)
    x_in = (x_in-mean_num)/mu
    x = np.c_[np.ones(1), [x_in]]

    result = np.matmul(x, theta)

    return result

def main():
    x_raw , y = get_data()
    x = data_normalize(x_raw)

    rate = 0.1
    steps = 500
    theta = gradient_decent(x, y, rate, steps)

    result = predict([1650,3], x_raw, theta)
    print(result)

main()









