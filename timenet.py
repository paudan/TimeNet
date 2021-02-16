import csv
import os
import shutil
from datetime import datetime
from collections import Iterable, OrderedDict
import warnings
import numpy as np
import tensorflow.keras.metrics as metrics
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, TimeDistributed, GRU, Dropout, Lambda, Masking
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

OUTPUT_DIR = os.path.join("models")
seed = 0


def normalize_series(series, normalization):
    if normalization is None or normalization.lower() == 'none':
        return series
    if normalization.lower() not in ('minmax', 'zscore'):
        warnings.warn("normalization parameter is not valid, reverting to None")
        return series
    if normalization == 'zscore':
        series = (series - np.mean(series)) / np.std(series)
    elif normalization == 'minmax':
        high, low = 0, 1
        series = (high - low) * (series - series.min())/(series.max() - series.min())
    mask_na = np.isnan(series) | np.isinf(series)
    if len(series[~mask_na]) == 0:
        return None
    series[mask_na] = np.mean(series[~mask_na])
    return series


class TimedCSVLogger(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch', 'time'] + self.keys, dialect=csv.excel)
            if self.append_header:
                self.writer.writeheader()
        row_dict = OrderedDict({'epoch': epoch, 'time': str(datetime.now())})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class SimpleSeriesGenerator(Sequence):

    def __init__(self, series, maxlen=None, batch_size=32, normalize=None, X_only=False):
        self.series = series
        self.batch_size = batch_size
        self.normalize = normalize
        self.X_only = X_only
        self.maxlen = maxlen
        self.indexes = np.arange(series.shape[0])

    def __getitem__(self, idx):

        def get_series(index):
            series = self.series.iloc[index]
            if self.maxlen is not None and len(series) > self.maxlen:
                series = series[:self.maxlen]
            series = normalize_series(series, self.normalize)
            return series

        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = [get_series(idx) for idx in indexes]
        # Skip series which resulted in NaNs after normalization
        X = np.array([x for x in X if x is not None])
        if self.maxlen is None:
            maxlen = np.max([len(x) for x in X])
        else:
            maxlen = self.maxlen
        X = pad_sequences(X, maxlen=maxlen, value=0., dtype='float', padding='post')
        X = np.expand_dims(X, axis=2)
        if self.X_only:
            return X
        else:
            X_rev = np.array([x[::-1] for x in X])
            return X, X_rev

    def __len__(self):
        return int(np.floor(self.series.shape[0] / self.batch_size))


class TimeNet:

    def __init__(self, size, num_layers, batch_size=32, dropout=0.0, model_name=None):
        self.size = size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.model_name = model_name
        np.random.seed(seed)
        self.init_model()

    def encoder_block(self, model_input):
        states_list = list()
        masked_input = Masking(mask_value=0.0, input_shape=(None,1))(model_input)
        encode, states = GRU(self.size, name='encode_1', return_state=True, return_sequences=True)(masked_input)
        states_list.append(states)
        if self.dropout > 0.0:
            encode = Dropout(self.dropout, name='drop_encode_1')(encode)
        for i in range(2, self.num_layers+1):
            encode, states = GRU(self.size, name='encode_{}'.format(i), return_state=True, return_sequences=True)(encode)
            states_list.append(states)
            if self.dropout > 0.0:
                encode = Dropout(self.dropout, name='drop_encode_{}'.format(i))(encode)
        states = K.concatenate(states_list, axis=1)
        return encode, states

    def decoder_block(self, input, encode):
        def trimOutputs(x):
            return x[0]*K.cast(K.not_equal(x[1],0), dtype=K.floatx())
        decode = K.reverse(encode, axes=0)
        for i in range(self.num_layers, 0, -1):
            if self.dropout > 0.0 and i > 0:  # skip these for first layer for symmetry
                decode = Dropout(self.dropout, name='drop_decode_{}'.format(i))(decode)
            decode = GRU(self.size, name='decode_{}'.format(i), return_sequences=True)(decode)
        decode = TimeDistributed(Dense(1, activation='linear'), name='time_dist')(decode)
        decode = Lambda(trimOutputs)([decode,input])  # Trim padded values
        return decode


    def init_model(self):
        model_input = Input(shape=(None, 1), name='main_input')
        encode, states = self.encoder_block(model_input)
        self.encoder = Model(model_input, states)
        decode = self.decoder_block(model_input, encode)
        self.model = Model(model_input, decode)
        self.model.summary()


    def train(self, generator, nb_epoch, lr=0.01, finetune_rate=None,
              validation_data=None, early_stop=5):
        run = self.get_run_id()
        optimizer = Adam(lr=lr if not finetune_rate else finetune_rate, clipnorm=1.)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=[
            metrics.RootMeanSquaredError(),
            metrics.MeanAbsoluteError(),
            metrics.MeanSquaredLogarithmicError()
        ])
        log_dir = os.path.join(OUTPUT_DIR, run)
        weights_path = os.path.join(log_dir, 'weights.h5')
        loaded = False
        if os.path.exists(weights_path):
            print("Loading {}...".format(weights_path))
            self.model.load_weights(weights_path)
            loaded = True
        if finetune_rate:  # write logs to new directory
            log_dir += "_ft{:1.0e}".format(finetune_rate).replace('e-', 'm')
        if (not loaded or finetune_rate):
            shutil.rmtree(log_dir, ignore_errors=True)
            os.makedirs(log_dir)
        weights_path = os.path.join(log_dir, 'weights.h5')
        history = self.model.fit(generator, epochs=nb_epoch, validation_data=validation_data,
                                 callbacks=[
                                     TensorBoard(log_dir=log_dir, write_graph=False),
                                     TimedCSVLogger(os.path.join(log_dir, 'training.csv'), append=True),
                                     ModelCheckpoint(weights_path, save_weights_only=True),
                                     ReduceLROnPlateau(),
                                     EarlyStopping(patience=early_stop)
                                 ])
        self.save_model(os.path.join(log_dir, "timenet_model.h5"))
        self.save_encoder(os.path.join(log_dir, "timenet_encoder.h5"))
        return history, log_dir

    def get_run_id(self):
        """Generate unique ID from model params."""
        run = "{}_x{}_drop{}".format(self.size, self.num_layers, int(100 * self.dropout)).replace('e-', 'm')
        if self.model_name:
            run += "_{}".format(self.model_name)
        return run

    def encode(self, generator):
        return self.encoder.predict(generator, verbose=True)

    def decode(self, series, return_series=True):
        if series is None:
            return None
        predicted = self.model.predict(series, verbose=False)
        if return_series is True:
            predicted = predicted.squeeze()
        return predicted

    def save_model(self, output_file):
        try:
            self.model.save(output_file)
        except:
            pass

    def save_encoder(self, output_file):
        self.encoder.save(output_file)
