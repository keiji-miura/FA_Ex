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


def xor2d(N_samples, N_input_layer, N_middle_layer, epoch,
          learning_rates, learning_rule, active_fnc, active_derive):
    # initialized weight (Relu):--------------------------------------------------------
    # input layer ⇒ middle layer
    W1 = (np.random.rand(N_input_layer, N_middle_layer) - 0.5) * 0.02  # ReLu 0.02
    b1 = (np.random.rand(1, N_middle_layer) - 0.5) * 0.02
    # middle layer ⇒ output layer
    W2 = (np.random.rand(N_middle_layer, 1) - 0.5) * 0.02
    b2 = (np.random.rand(1, 1) - 0.5) * 0.02

    # update random value
    B1_ori = (np.random.rand(N_middle_layer, 1) - 0.5) * 0.5  # 固定
    B1 = np.random.randn(N_middle_layer, 1) * 0.1  # / np.sqrt(N_middle_layer / 2)
    B100 = ((np.random.rand(N_middle_layer, 1) < 1.0) - 0.5) * 0.2
    B80 = ((np.random.rand(N_middle_layer, 1) < 0.8) - 0.5) * 0.2
    B50 = ((np.random.rand(N_middle_layer, 1) < 0.0) - 0.5) * 0.2

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

    # shift_range = np.arange(-1.0, 1.1, 0.1)
    shift_range = np.arange(-0.2, 0.22, 0.02)
    print(shift_range)

    for rules, color in zip(['BP', 'FA', 'FA_normal', 'FA_Ex-80%', 'FA_Ex-100%'],
                            ['black', 'green', 'lime', 'pink', 'red']):

        shift_history = []

        # sifted range Relu function derivative
        for i in shift_range:

            # defined sifted range Relu function derivative
            def Relu_derivative_mod(x):
                y = np.where(0 < x - i, 1, 0)
                # y = np.where(0 < x, 1, 0) + i
                return y

            df_hist = pd.DataFrame()

            for j in range(repeat):
                predict = xor2d(batch_size, input_dimension, neuron_number, epochs,
                                learning_rate, rules, Relu, Relu_derivative_mod)
                predict = pd.DataFrame.from_dict(predict, orient='index').T
                df_hist = pd.concat([df_hist, predict], axis='columns')

            print(rules)
            print("shift: {0}".format(i))

            mean_df = df_hist['test_loss'].mean(axis=1)  # .to_numpy()
            sem_array = (df_hist['test_loss'].std(axis=1) / np.sqrt(repeat))  # .to_numpy()
            print(mean_df)

            print((mean_df.values < 0.1).argmax())

            if (mean_df.values < 0.1).argmax() != 0:
                shift_history.append((mean_df.values < 0.1).argmax())
            else:
                shift_history.append(epochs)

        print(f'rule: {rules}, hist: {shift_range}')
        print(f'rule: {rules}, hist: {shift_history}')
        plt.plot(shift_range, shift_history, label=rules, color=color)

    plt.legend(fontsize=14)
    plt.xlabel('shift')
    plt.ylabel('epoch')
    plt.vlines(0, 0, epochs, color='gray', linestyles='dotted')
    plt.subplots_adjust(left=0.16, bottom=0.13, right=0.98, top=0.98)
    fig.savefig("figure.svg")
    plt.show()
