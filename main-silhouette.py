import warnings
import pathlib
from typing import Any, Union

warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def Kmeans(X, nclust):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return clust_labels, cent


if __name__ == '__main__':
    import click


    @click.command()
    @click.option('-i', '--file', 'file_in', help='CSV file')
    # @click.option('-o', '--out file', 'file_out', help='output name')
    def main(file_in):
        print('Reading Data')
        df = pd.read_csv(file_in, index_col=0)
        df2 = df.copy()

        name = file_in.split('/')
        # print(file_in)
        # print(name)
        name = name[-1].split('.')
        name = name[0]
        # print(name)
        save_path = str(pathlib.Path().absolute()) + '\\' + 'Silhouette'
        # print(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        sa = save_path + '\\' + 'Shiloutte_'
        df = df2[
            ['src_port', 'dst_port', 'protocol', 'bidirectional_packets', 'bidirectional_bytes', 'bidirectional_min_ps',
             'bidirectional_max_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps', 'bidirectional_first_seen_ms',
             'bidirectional_last_seen_ms', 'bidirectional_duration_ms', 'bidirectional_min_piat_ms',
             'bidirectional_max_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms']]
        # df.drop(columns=['src_ip', 'dst_ip', 'application_name', 'application_category_name'], inplace=True)
        X = df.fillna(0)
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        aux = []
        n_clusters: Union[int, Any]
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            aux.append(silhouette_avg)
            # aux.append(n_clusters)
            # print('\n\n caida2018_150s')
            # print("For n_clusters =", n_clusters,
            #      "The average silhouette_score is :", silhouette_avg)
        df = pd.DataFrame(aux)
        o = sa + str(name)
        df.to_csv(o + '.csv')
        print('Shiloutte', name, ' Done!')
main()
