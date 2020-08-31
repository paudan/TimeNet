import os
import uuid
import numpy as np
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from tensorflow.keras.models import load_model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from timenet import SimpleSeriesGenerator


UCR_DATASET_PATH = '/mnt/DATA/data/Univariate_ts'

datasets_test = ["SyntheticControl", "PhalangesOutlinesCorrect", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
                 "DistalPhalanxTW", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW",
                 "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
                 "ElectricDevices", "MedicalImages", "SwedishLeaf", "TwoPatterns", "ECG5000", "ECGFiveDays", "Wafer",
                 "ChlorineConcentration", "Adiac", "Strawberry","CricketX", "CricketY", "CricketZ",
                  "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
                  "Yoga", "FordA", "FordB"]
datasets_train = ["ItalyPowerDemand", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2","FacesUCR",
                  "GunPoint", "WordSynonyms", "Lightning7", "DiatomSizeReduction", "Ham", "ShapeletSim", "TwoLeadECG", "Plane",
                  "ArrowHead", "ToeSegmentation1","ToeSegmentation2","OSULeaf","Fish","ShapesAll"]
datasets_valid = ["MoteStrain", "CBF", "Trace", "Symbols", "Herring", "Earthquakes"]

def create_series_dataset(datasets):

    def process_dataset(dst):
        train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TRAIN.ts"))
        test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(UCR_DATASET_PATH, dst, dst + "_TEST.ts"))
        series_list = [train_x.iloc[i][0] for i in range(train_x.shape[0])] + \
                      [test_x.iloc[i][0] for i in range(test_x.shape[0])]
        classes = np.hstack([train_y, test_y])
        return pd.concat([pd.DataFrame({'dataset': dst, 'series_id': str(uuid.uuid4()), 'value': series_list[i], 'class': classes[i]})
                          for i in range(len(series_list))])

    return pd.concat(map(process_dataset, datasets))

def create_datasets():
    data_train = create_series_dataset(datasets_train).reset_index(drop=True)
    data_train.to_feather(os.path.join('data', 'dataset_train.feather'))
    data_valid = create_series_dataset(datasets_valid).reset_index(drop=True)
    data_valid.to_feather(os.path.join('data', 'dataset_valid.feather'))
    data_test = create_series_dataset(datasets_test).reset_index(drop=True)
    data_test.to_feather(os.path.join('data', 'dataset_test.feather'))

def read_series_data(filename):
    dst_train = pd.read_feather(filename)
    series = dst_train[['series_id', 'value']].groupby(by='series_id')['value'].apply(np.array)
    series = pd.DataFrame({'series': series, 'length': series.apply(len)}) \
        .sort_values(by='length', ascending=False) \
        .reset_index()
    return series[['series_id', 'series']], series['length'].max()

def read_series_metadata(filename):
    dst_train = pd.read_feather(filename)
    return dst_train[['dataset','series_id', 'class']].drop_duplicates()

def visualize_embeddings(filename, n_iter=500, perplexity=10, lr=500, output_dir='.'):
    data = pd.read_feather(filename)
    datasets = data['dataset'].unique()
    for dst in datasets:
        data_dst = data[data['dataset'] == dst]
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, n_iter=n_iter, learning_rate=lr, init='pca')
        coords = tsne.fit_transform(data_dst.iloc[:, :-3])
        plot_coords(coords, labels=data_dst['class'].astype(int).values.tolist(),
                    output_filename=os.path.join(output_dir, f'{dst}_tsne.jpg'))

def visualize_pca(filename, output_dir='.'):
    data = pd.read_feather(filename)
    datasets = data['dataset'].unique()
    for dst in datasets:
        data_dst = data[data['dataset'] == dst]
        tsne = PCA(n_components=2)
        coords = tsne.fit_transform(data_dst.iloc[:, :-3])
        plot_coords(coords, labels=data_dst['class'].astype(int).values.tolist(),
                    output_filename=os.path.join(output_dir, f'{dst}_pca.jpg'))


def plot_coords(coords_data, labels=None, output_filename='tsne.png'):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(coords_data[:, 0], coords_data[:, 1], c=labels, marker='o', cmap='nipy_spectral', s=8)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    ax.grid(True)
    plt.savefig(output_filename)
    plt.close(fig)

def embed(data_file, encoder_path, batch_size=32, output_file='embeddings.feather'):
    series_train, maxlen = read_series_data(data_file)
    train_data = series_train['series']
    generator = SimpleSeriesGenerator(train_data, batch_size=batch_size, X_only=True, normalize='zscore', maxlen=None)
    model = load_model(encoder_path)
    embed_train = model.predict(generator)
    embed_train = pd.DataFrame(embed_train)
    embed_train.columns = list(map(str, range(embed_train.shape[1])))
    embed_train['series_id'] = series_train['series_id']
    train_meta = read_series_metadata(data_file)
    embed_train = embed_train.merge(train_meta, on='series_id')
    embed_train.to_feather(output_file)

