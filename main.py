import argparse
import itertools
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from timenet import TimeNet, SimpleSeriesGenerator, normalize_series
from process import read_series_data, read_series_metadata


def create_train_valid_sets(series, validation_split=0.2, batch_size=32):
    x = range(series.shape[0])
    batches = int(np.floor(series.shape[0] / batch_size))
    batches_train, batches_valid = train_test_split(range(batches), test_size=validation_split, random_state=0)
    idx_train = sorted(itertools.chain(*[x[(ind * batch_size):((ind + 1) * batch_size)] for ind in batches_train]))
    idx_valid = sorted(itertools.chain(*[x[(ind * batch_size):((ind + 1) * batch_size)] for ind in batches_valid]))
    return series.iloc[idx_train], series.iloc[idx_valid]

def reconstruct(train_data, enc, log_dir, normalize=True):
    print("Creating reconstructions...")
    pd.concat([pd.DataFrame({'index': k,
                             'series': normalize_series(train_data.tolist()[k], normalize),
                             'decoded': enc.decode(train_data.iloc[k])})
               for k in range(len(train_data))], axis='rows')\
        .reset_index()\
        .to_feather(os.path.join(log_dir, 'reconstructed_train.feather'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset", help="Training dataset in feather format")
    parser.add_argument("--validation-dataset", help="Validation dataset in feather format (optional)", required=False)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument('--embeddings-dim', type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument('--finetune-rate', type=float, default=None, required=False)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='model')
    parser.add_argument('--dynamic-batches', dest='dynamic_batches', action='store_true', help='Use dynamic size batches')
    parser.add_argument('--normalize', dest='normalize', choices=['none', 'zscore', 'minmax'], default='none')
    parser.set_defaults(dynamic_batches=True)
    args = parser.parse_args()

    embeddings_dim = args.embeddings_dim
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    num_layers = args.num_layers
    normalize = args.normalize
    model_name = args.model_name or 'enc'

    training_file = args.training_dataset
    if training_file is None or len(training_file.strip()) == 0 or not os.path.isfile(training_file):
        raise Exception(f"Training dataset {training_file} does not exist")
    validation_file = args.validation_dataset
    if validation_file is None or len(validation_file.strip()) == 0 or not os.path.isfile(validation_file):
        warnings.warn(f"Validation dataset {training_file} does not exist, will use training dataset for validation")
        validation_file = None

    if validation_file is None:
        series_train, maxlen = read_series_data(training_file)
        train_data, valid_data = create_train_valid_sets(series_train['series'], batch_size=batch_size)
    else:
        series_train, maxlen = read_series_data(training_file)
        series_valid, _ = read_series_data(validation_file)
        train_data = series_train['series']
        valid_data = series_valid['series']
    if args.dynamic_batches is True:
        maxlen = None
    train_generator = SimpleSeriesGenerator(train_data, batch_size=batch_size, X_only=False, normalize=normalize, maxlen=maxlen)
    valid_generator = SimpleSeriesGenerator(valid_data, batch_size=batch_size, X_only=False, normalize=normalize, maxlen=maxlen)
    enc = TimeNet(embeddings_dim, num_layers=num_layers, batch_size=batch_size, model_name=model_name, dropout=args.dropout)
    history, log_dir = enc.train(train_generator, nb_epoch=n_epochs, validation_data=valid_generator,
                                 finetune_rate=args.finetune_rate, lr=args.learning_rate, early_stop=args.early_stop)
    print(history.history)
    print("Creating embeddings for the series dataset...")
    generator = SimpleSeriesGenerator(train_data, batch_size=batch_size, X_only=True, normalize=normalize, maxlen=maxlen)
    embed_train = enc.encode(generator)
    embed_train = pd.DataFrame(embed_train)
    embed_train.columns = list(map(str, range(embed_train.shape[1])))
    embed_train['series_id'] = series_train['series_id']
    train_meta = read_series_metadata(training_file)
    embed_train = embed_train.merge(train_meta, on='series_id')
    embed_train.to_feather(os.path.join(log_dir, 'embeddings.feather'))
