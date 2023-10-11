"""
reserched robustness1: shifted ReLu function derivative
reserched robustness2: expanded reduced range of ReLu function derivative
reserched robustness3: approached to Relu function derivative (sigmoid)
reserched robustness4: approached to Relu function derivative (Relu)
reserched robustness5: add noise to Relu function derivative
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

plt.rcParams["font.size"] = 18


def Relu(x):
    y = np.where(x >= 0, x, 0)
    return y


def Relu_derivative(x):
    y = np.where(0 < x, 1, 0)
    return y


# shifted Relu function derivative
def Relu_derivative_mod(x):
    y = np.where(0 < x + 0.0, 1, 0) + 0.0
    return y


# shifted Relu function derivative
def Relu_star(x):
    y = np.where((-0.05 < x) & (x < 0.05), 10 * x, 0)
    y = np.where(0.05 <= x, 0.5, y)
    y = np.where(-0.05 >= x, -0.5, y)
    return y + 0.5


# approached to relu function derivative
def sigmoid_mod(x):
    return 1.0 / (1 + np.exp(-1 * x))


# generate noise
def noise(ver, bes, no):
    noises = np.random.randn(ver, bes) * no
    return noises


# generate input data
def input_data(sample_number, dimension_number):
    noise = np.random.randn(sample_number, dimension_number) * 0.01
    m = np.random.rand(sample_number, dimension_number) > 0.5
    m = 2 * (m + 0.0) - 1  # standardized
    x_data = m + noise
    return x_data


# generate output data
def output_data(x_data, sample_number):
    # x_data[:, 0:A]
    y_data = np.prod(np.sign(x_data[:, 0:2]), axis=1).reshape([sample_number, 1])
    return y_data


def xor2d(N_samples, N_input_layer, N_middle_layer, epoch, learning_rates, learning_rule, active_fnc, active_derive, ):
    # initialized weight (Relu):--------------------------------------------------------
    # input layer ⇒ middle layer
    W1 = (np.random.rand(N_input_layer, N_middle_layer) - 0.5) * 0.02  # ReLu 0.02
    b1 = (np.random.rand(1, N_middle_layer) - 0.5) * 0.02
    # middle layer ⇒ output layer
    W2 = (np.random.rand(N_middle_layer, 1) - 0.5) * 0.02
    b2 = (np.random.rand(1, 1) - 0.5) * 0.02

    # update random value
    B1_ori = (np.random.rand(N_middle_layer, 1) - 0.5) * 0.5  # original
    B1 = np.random.randn(N_middle_layer, 1) * 0.1  # / np.sqrt(N_middle_layer / 2)
    B100 = ((np.random.rand(N_middle_layer, 1) < 1.0) - 0.5) * 0.2
    B80 = ((np.random.rand(N_middle_layer, 1) < 0.8) - 0.5) * 0.2
    B50 = ((np.random.rand(N_middle_layer, 1) < 0.0) - 0.5) * 0.2

    # generate fixed noise
    fix_noise = noise(batch_size, N_middle_layer, 1.5)
    # fix_noise[:, :] = fix_noise[0, :]

    train_loss = float('inf')
    test_predict = 0
    x_tests = np.random.randn(N_samples, N_input_layer)

    # data history
    history = {'train_loss': [], 'train_output': [], 'sigmoid': [], 'W2': [],
               'x_test': [], 'test_loss': [], 'test_output': []}

    for i in range(epoch):
        # generate train data-------------------------------------------------------------------------------------------
        x_trains = input_data(N_samples, N_input_layer)
        y_trains = output_data(x_trains, N_samples)
        # forward
        z1 = np.dot(x_trains, W1) + b1
        y1 = active_fnc(z1)
        z2 = np.dot(y1, W2) + b2
        train_predict = z2

        # loss function
        pre_loss = train_predict - y_trains
        train_loss = np.mean((train_predict - y_trains) ** 2)

        # generate test data-----------------------------------------------------
        x_tests = input_data(N_samples, N_input_layer)
        y_tests = output_data(x_tests, N_samples)

        # generate vary noise
        vary_noise = noise(batch_size, N_middle_layer, 1.5)
        # vary_noise[:, :] = vary_noise[0, :]

        # forward(test data)
        z1_test = np.dot(x_tests, W1) + b1
        y1_test = active_fnc(z1_test)
        z2_test = np.dot(y1_test, W2) + b2
        test_predict = z2_test

        # loss (test data)
        test_loss = np.mean((test_predict - y_tests) ** 2)

        # history
        history['train_loss'].append(train_loss)
        history['train_output'].append(train_predict)
        history['W2'].append(W2)
        history['x_test'].append(x_tests[:, 0:2])
        history['test_loss'].append(test_loss)
        history['test_output'].append(test_predict)

        # output layer backward------------------------------------
        W2 -= learning_rates * np.dot(y1.T, pre_loss)
        b2 -= learning_rates * np.sum(pre_loss, axis=0, keepdims=True)

        # input layer backward------------------------------------
        if learning_rule == "BP":
            dz1 = np.dot(pre_loss, W2.T) * (active_derive(z1))
            W1 -= learning_rates * np.dot(x_trains.T, dz1)
            b1 -= learning_rates * np.sum(dz1, axis=0, keepdims=True)

        elif learning_rule == "FA":
            dz1 = np.dot(pre_loss, B1_ori.T) * (active_derive(z1))
            W1 -= learning_rates * np.dot(x_trains.T, dz1)
            b1 -= learning_rates * np.sum(dz1, axis=0, keepdims=True)

        elif learning_rule == "FA_normal":
            dz1 = np.dot(pre_loss, B1.T) * (active_derive(z1))
            W1 -= learning_rates * np.dot(x_trains.T, dz1)
            b1 -= learning_rates * np.sum(dz1, axis=0, keepdims=True)

        elif learning_rule == "FA_Ex-100%":
            dz1 = np.dot(pre_loss, B100.T) * (active_derive(z1))
            W1 -= learning_rates * np.dot(x_trains.T, dz1)
            b1 -= learning_rates * np.sum(dz1, axis=0, keepdims=True)

        elif learning_rule == "FA_Ex-80%":
            dz1 = np.dot(pre_loss, B80.T) * (active_derive(z1))
            W1 -= learning_rates * np.dot(x_trains.T, dz1)
            b1 -= learning_rates * np.sum(dz1, axis=0, keepdims=True)

        elif learning_rule == "FA_Ex-50%":
            dz1 = np.dot(pre_loss, B50.T) * (active_derive(z1))
            W1 -= learning_rates * np.dot(x_trains.T, dz1)
            b1 -= learning_rates * np.sum(dz1, axis=0, keepdims=True)

        else:
            print("error")
            break

    return history


# ------------------------------------------------------------
if __name__ == "__main__":
    fig = plt.figure()
    # hyper parameter
    seed = 10
    np.random.seed(seed)
    repeat = 100
    epochs = 1000
    learning_rate = 0.005
    batch_size = 8
    input_dimension = 2 + 100
    neuron_number = 20

    for rules, color in zip(['BP', 'FA', 'FA_normal', 'FA_Ex-80%', 'FA_Ex-100%'],
                            ['black', 'green', 'lime', 'pink', 'red']):

        df_hist = pd.DataFrame()

        start = time.time()
        for i in range(repeat):
            predict = xor2d(batch_size, input_dimension, neuron_number, epochs,
                            learning_rate, rules, Relu, Relu_derivative_mod)
            predict = pd.DataFrame.from_dict(predict, orient='index').T
            df_hist = pd.concat([df_hist, predict], axis='columns')
        elapsed_time = time.time() - start

        print(rules)
        print("elapsed time:{0}".format(elapsed_time) + "[sec]")
        print(df_hist['test_loss'].mean(axis=1))

        mean_array = df_hist['test_loss'].mean(axis=1).to_numpy()
        sem_array = (df_hist['test_loss'].std(axis=1) / np.sqrt(repeat)).to_numpy()

        plt.plot(list(range(1, epochs + 1)), mean_array, label=rules, color=color)
        plt.fill_between(list(range(1, epochs + 1)), mean_array + sem_array,
                         mean_array - sem_array, alpha=0.4, color=color)

    plt.legend(fontsize=14, loc='lower left')
    # plt.legend(fontsize=14, loc='upper right')
    plt.vlines(500, 0, epochs, color='gray', linestyles='dotted')
    plt.hlines(0.1, 0, epochs, color='gray', linestyles='dotted')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0.0001, 10)
    plt.yscale('log')
    plt.subplots_adjust(left=0.15, bottom=0.13, right=0.99, top=0.97)

    fig.savefig("figure.svg")
    plt.show()
