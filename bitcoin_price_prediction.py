# Note: this Python file is a modification of Javier Béjar's Air Quality RNN -- https://github.com/bejar/DLMAI/blob/master/AirQuality/AQPredictionRNN.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras import utils
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import argparse
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


WEIGHTED_PRICE_INDEX = 6

def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def lagged_vector(data, lag=1, ahead=0):
    """
    Returns a vector with columns that are the steps of the lagged time series
    Last column is the value to predict

    Because arrays start at 0, Ahead starts at 0 but actually means one step ahead

    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i])
    lvect.append(data[lag + ahead:])

    return np.stack(lvect, axis=1)


def lagged_matrix(data, lag=1, ahead=0):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []

    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i, :])
    lvect.append(data[lag + ahead:, :])
    return np.stack(lvect, axis=1)


def _generate_dataset_one_var(data, datasize, testsize, lag=1, ahead=1):
    """
    Generates dataset assuming only one variable for prediction
    Here ahead starts at 1 (I know it is confusing)

    :return:
    """

    scaler = MinMaxScaler(feature_range=(0, 1))#StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    # print('DATA Dim =', data.shape)

    bitcoin_train = data[:datasize, :]
    # print('Train Dim =', bitcoin_train.shape)

    train = lagged_vector(bitcoin_train, lag=lag, ahead=ahead - 1)

    train_x, train_y = train[:, :lag], train[:, -1:, 0]

    bitcoin_test = data[datasize:datasize + testsize, 0].reshape(-1, 1)
    test = lagged_vector(bitcoin_test, lag=lag, ahead=ahead - 1)

    test_x, test_y = test[:, :lag], test[:, -1:, 0]

    return train_x, train_y, test_x, test_y, scaler


def generate_dataset(config, ahead=1, data_path=None):
    """
    Generates the dataset for training, test and validation

    :param ahead: number of steps ahead for prediction

    :return:
    """
    dataset = config['dataset']
    datanames = config['datanames']
    datasize = config['datasize']
    testsize = config['testsize']
    vars = config['vars']
    lag = config['lag']

    btc = {}

    # Reads numpy arrays for all sites and keep only selected columns

    btcdata = np.load(data_path + 'bitcoin_price_history.npz')
    for d in datanames:
        btc[d] = btcdata[d]
        if vars is not None:
            btc[d] = btc[d][:, vars]

    if dataset == 0:
        return _generate_dataset_one_var(btc[datanames[0]][:, WEIGHTED_PRICE_INDEX].reshape(-1, 1), datasize, testsize, lag=lag, ahead=ahead)
    # Just add more options to generate datasets with more than one variable for predicting one value
    # or a sequence of values

    raise NameError('ERROR: No such dataset type')


def architecture(train_x, neurons, drop, nlayers, activation, activation_r, rnntype, impl=1):
    """
    RNN architecture

    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))

    #model.add(Dense(500))
    model.add(Dense(1))

    return model


def main():

    """Main function"""

    #pd.read_csv("BTC_price_history.csv", names=["Timestamp", "Low", "High", "Open", "Close", "Volume", "WeightedPrice"])
    #np.save("bitcoin_price_history.npy", price_matrix.values)

    #price_matrix = pd.DataFrame(np.load("bitcoin_price_history.npy"), columns=["Timestamp", "Low", "High", "Open", "Close", "Volume", "WeightedPrice"])
    #print(price_matrix.iloc[0:10])

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2

    config = load_config_file(args.config)
    ############################################
    # Data

    modfile = None

    ahead = config['data']['ahead']

    if args.verbose:
        print('-----------------------------------------------------------------------------')
        print('Steps Ahead = %d ' % ahead)

    # Modify conveniently with the path for your data
    aq_data_path = './'

    train_x, train_y, test_x, test_y, scaler = generate_dataset(config['data'], ahead=ahead, data_path=aq_data_path)

    ############################################
    # Model

    model = architecture(train_x=train_x,
                         neurons=config['arch']['neurons'],
                         drop=config['arch']['drop'],
                         nlayers=config['arch']['nlayers'],
                         activation=config['arch']['activation'],
                         activation_r=config['arch']['activation_r'], rnntype=config['arch']['rnn'], impl=impl)
    if args.verbose:
        model.summary()
        print('lag: ', config['data']['lag'],
              '/Neurons: ', config['arch']['neurons'],
              '/Layers: ', config['arch']['nlayers'],
              '/Activations:', config['arch']['activation'], config['arch']['activation_r'])
        print('Tr:', train_x.shape, train_y.shape, 'Ts:', test_x.shape, test_y.shape)
        print()

    ############################################
    # Training

    optimizer = config['training']['optimizer']

    if optimizer == 'rmsprop':
        if 'lrate' in config['training']:
            optimizer = RMSprop(lr=config['training']['lrate'])
        else:
            optimizer = RMSprop(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    cbacks = []

    if args.tboard:
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        cbacks.append(tensorboard)

    if args.best:
        modfile = './model%d.h5' % int(time())
        mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
        cbacks.append(mcheck)

    model.fit(train_x, train_y, batch_size=config['training']['batch'],
              epochs=config['training']['epochs'],
              validation_data=(test_x, test_y),
              verbose=verbose, callbacks=cbacks)

    utils.plot_model(model, to_file="model.png", show_shapes=True)

    ############################################
    # Results

    if args.best:
        model = load_model(modfile)

    score = model.evaluate(test_x, test_y, batch_size=config['training']['batch'], verbose=0)

    print()
    print('MSE test:', score)
    print('MSE test persistence:', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
    test_yp = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)

    test_y_transformed = scaler.inverse_transform(test_y)
    test_yp_transformed = scaler.inverse_transform(test_yp)
    '''for y, yp in zip(test_y_transformed, test_yp_transformed):
        print(y, yp)'''
    real_y = [y[0] for y in test_y_transformed]
    predicted_y = [yp[0] for yp in test_yp_transformed]
    plt.figure(figsize=(16, 9), dpi=500)
    plt.plot(range(len(real_y)), real_y, "g--", label="Real price")
    plt.plot(range(len(real_y)), predicted_y, "b--", label="Predicted price")
    plt.xlabel("Minute")
    plt.ylabel("Bitcoin Price (USD)")
    plt.legend(loc='lower right')
    plt.axis([0, len(real_y), min(min(real_y), min(predicted_y)),  max(max(real_y), max(predicted_y))])
    plt.savefig("prediction_plot.png")

    difference_y = [float(real_y[i] - predicted_y[i]) for i in range(len(real_y))]
    total_difference_y = sum(difference_y)
    print("Sum of real-predicted differences:", total_difference_y)
    mean_difference_y = total_difference_y / len(real_y)
    print("Forecast bias: {}$".format(round(mean_difference_y, 2)))
    seq1 = real_y[ahead:]
    seq2 = real_y[0:-ahead]
    data_consec_difference = [abs(float(seq1[i] - seq2[i])) for i in range(len(seq1))]
    print("Mean consecutive data bias: {}$".format(round(np.mean([float(seq1[i] - seq2[i]) for i in range(len(seq1))]), 2)))
    mean_abs_error = np.mean(np.abs(difference_y))
    print("Mean absolute error: {}$".format(round(float(mean_abs_error), 2)))
    mean_data_diff = np.mean(data_consec_difference)
    print("Data mean absolute difference between consecutive examples: {}$".format(round(float(mean_data_diff), 2)))
    mean_abs_percent = np.mean([abs(diff) / real for diff, real in zip(difference_y, real_y)]) * 100#mean_abs_error / np.mean(real_y) * 100
    print("Mean error percentage: {}%".format(round(mean_abs_percent, 2)))
    mean_data_diff_percent = np.mean([abs(diff) / real for diff, real in zip(data_consec_difference, real_y)]) * 100#mean_data_diff / np.mean(real_y) * 100
    print("Data mean difference between consecutive examples percentage: {}%".format(round(mean_data_diff_percent, 2)))

    r2test = r2_score(test_y, test_yp)
    r2pers = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
    print('R2 test:', r2test)
    print('R2 test persistence:', r2pers)

    resfile = open('result-%s.txt' % config['data']['datanames'][0], 'a')
    resfile.write('DATAS= %d, LAG= %d, AHEAD= %d, RNN= %s, NLAY= %d, NNEUR= %d, DROP= %3.2f, ACT= %s, RACT= %s, '
                  'OPT= %s, R2Test = %3.5f, R2pers = %3.5f\n' %
                  (config['data']['dataset'],
                   config['data']['lag'],
                   config['data']['ahead'],
                   config['arch']['rnn'],
                   config['arch']['nlayers'],
                   config['arch']['neurons'],
                   config['arch']['drop'],
                   config['arch']['activation'],
                   config['arch']['activation_r'],
                   config['training']['optimizer'],
                   r2test, r2pers
                   ))
    resfile.close()

    # Deletes the model file
    try:
        if modfile:
            os.remove(modfile)
    except OSError:
        pass


if __name__ == '__main__':

    main()
