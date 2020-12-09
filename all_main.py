import os
import pathlib
import pandas as pd
import numpy as np
import glob as gl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import log as ln
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler  # For scaling dataset
import warnings

warnings.filterwarnings('ignore')

"""
Configuration of the Font size and style for all the plots
"""

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})

"""
def get_h is to 
"""


def get_h(time_str):
    a = time_str.count(':')
    # print(a)
    if a < 2:
        # print(time_str)
        h, m = time_str.split(':')
        sec = int(h) * 3600 + int(m) * 60
        return sec / 3600
    else:
        h, m, s = time_str.split(':')
        sec = int(h) * 3600 + int(m) * 60 + int(s)
        return sec / 3600


def plotBIOMAS_ODA_vsTime(df_biomas, df_oda, name, paths):
    columns_names = list(df_biomas.columns)
    columns_names_oda = list(df_oda.columns)

    # print(columns_names)
    name_bio = 'Biomass Vs Time - ' + name
    fig = plt.figure(name_bio)
    ax = fig.gca()
    df_biomas.plot(kind='scatter', x=columns_names[0], y=columns_names[1], ax=ax)
    df_biomas.plot(kind='line', x=columns_names[0], y=columns_names[1], ax=ax)
    plt.axis([0, df_biomas[columns_names[0]].max() + 5, 0, df_biomas[columns_names[1]].max() + 2])
    ax.legend_ = None
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    plt.grid()

    if name_bio.find(' ') != -1:
        name_fig = paths + '\\' + name_bio.replace(' ', '') + '.pdf'
    else:
        name_fig = paths + '\\' + name_bio + '.pdf'
    plt.savefig(name_fig)

    name_oda = 'ODa Vs Time - ' + name
    fig_oda = plt.figure(name_oda)
    ax_oda = fig_oda.gca()
    df_oda.plot(kind='scatter', x=columns_names_oda[0], y=columns_names_oda[1], ax=ax_oda)
    df_oda.plot(kind='line', x=columns_names_oda[0], y=columns_names_oda[1], ax=ax_oda)
    plt.axis([0, df_oda[columns_names_oda[0]].max() + 5, 0, df_oda[columns_names_oda[1]].max() + 5])
    ax_oda.legend_ = None
    xticks_oda = ax_oda.xaxis.get_major_ticks()
    xticks_oda[0].label1.set_visible(False)
    plt.grid()

    if name_oda.find(' ') != -1:
        name_fig_Oda = paths + '\\' + name_oda.replace(' ', '') + '.pdf'
    else:
        name_fig_Oda = paths + '\\' + name_oda + '.pdf'
    plt.savefig(name_fig_Oda)


def lineal(x, a, b):
    return a * x + b


def exponential(x, a, k, b):
    return b / (1 + a * np.exp(-x * k))


def exponential2(x, a, b):
    return a * np.exp(x * b)


def biomassVSoda(df_biomas, df_oda, name, paths):
    columns_names_bio = list(df_biomas.columns)
    columns_names_oda = list(df_oda.columns)
    df_biomas_oda = df_biomas.copy()
    df_biomas_oda['ODa'] = 0.0

    for index, row in df_biomas.iterrows():
        t = round(row.at[columns_names_bio[0]])
        aux = df_oda.copy()
        aux.drop(aux[aux[columns_names_oda[0]] != t].index, inplace=True)
        df_biomas_oda.at[index, 'ODa'] = aux[columns_names_oda[1]]
        # print(df_biomas_oda)
        # print(t)
    x = df_biomas_oda['ODa']
    y = df_biomas_oda[columns_names_bio[1]]
    popt, _ = curve_fit(lineal, x, y)
    a, b = popt
    print('y = %.5f * x + %.5f' % (a, b))
    # define a sequence of inputs between the smallest and largest known inputs
    finalDF = pd.DataFrame({'x_line': [], 'y_line': [], 'x': [], 'y': []})
    finalDF['x_line'] = np.arange(min(x), max(x), 1)
    finalDF['y_line'] = lineal(finalDF['x_line'], a, b)
    finalDF['x'] = x
    finalDF['y'] = y

    columns_names_finalDF = list(finalDF.columns)
    # create a line plot for the mapping function
    name_bioVSoda = 'Biomass Vs ODa - ' + name
    fig_bioVSoda = plt.figure(name_bioVSoda)
    ax_bioVSoda = fig_bioVSoda.gca()
    finalDF.plot(kind='line', x=columns_names_finalDF[0], y=columns_names_finalDF[1], ls='--', ax=ax_bioVSoda,
                 color='red', label='y = %.5f * x + %.5f' % (a, b))
    finalDF.plot(kind='line', x=columns_names_finalDF[2], y=columns_names_finalDF[3], ax=ax_bioVSoda, label='Biomass')
    finalDF.plot(kind='scatter', x=columns_names_finalDF[2], y=columns_names_finalDF[3], ax=ax_bioVSoda)
    plt.axis([0, finalDF[columns_names_finalDF[0]].max() + 5, 0, finalDF[columns_names_finalDF[3]].max() + 5])
    xticks = ax_bioVSoda.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    plt.xlabel('ODa')
    plt.ylabel('Biomass')
    plt.grid()

    if name_bioVSoda.find(' ') != -1:
        name_fig_bioVSoda = paths + '\\' + name_bioVSoda.replace(' ', '') + '.pdf'
    else:
        name_fig_bioVSoda = paths + '\\' + name_bioVSoda + '.pdf'
    plt.savefig(name_fig_bioVSoda)
    #   plt.show()
    return a, b


def biomass_gDW_L(df_ODa, m, b, name, paths):
    df_ODa_biomass_GDW = df_ODa.copy()
    df_ODa_biomass_GDW['gDWL'] = 0.0
    ODa_columns = list(df_ODa_biomass_GDW.columns)
    for index, row in df_ODa.iterrows():
        x = row.at[ODa_columns[1]]
        df_ODa_biomass_GDW.at[index, ODa_columns[-1]] = (m * x) + b

    columns_names_ODa_biomass_GDW = list(df_ODa_biomass_GDW.columns)
    name_ODa_biomass_GDW = 'Biomass gDWL Vs Time - ' + name
    fig_ODa_biomass_GDW = plt.figure(name_ODa_biomass_GDW)
    ax_ODa_biomass_GDW = fig_ODa_biomass_GDW.gca()
    # df_ODa_biomass_GDW.plot(kind='scatter', x=columns_names_ODa_biomass_GDW[0], y=columns_names_ODa_biomass_GDW[
    # -1], ax=ax_ODa_biomass_GDW)
    df_ODa_biomass_GDW.plot(kind='line', x=columns_names_ODa_biomass_GDW[0], y=columns_names_ODa_biomass_GDW[-1],
                            ax=ax_ODa_biomass_GDW)
    plt.axis([0, df_ODa_biomass_GDW[columns_names_ODa_biomass_GDW[0]].max() + 5, 0,
              df_ODa_biomass_GDW[columns_names_ODa_biomass_GDW[-1]].max() + 5])
    ax_ODa_biomass_GDW.legend_ = None
    xticks_ODa_biomass_GDW = ax_ODa_biomass_GDW.xaxis.get_major_ticks()
    xticks_ODa_biomass_GDW[0].label1.set_visible(False)
    slash = r'/'
    plt.ylabel('Biomass (gDW ' + slash + ' L)')
    plt.grid()
    # plt.show()

    if name_ODa_biomass_GDW.find(' ') != -1:
        name_fig_ODa_biomass_GDW = paths + '\\' + name_ODa_biomass_GDW.replace(' ', '') + '.pdf'
    else:
        name_fig_ODa_biomass_GDW = paths + '\\' + name_ODa_biomass_GDW + '.pdf'
    plt.savefig(name_fig_ODa_biomass_GDW)

    return df_ODa_biomass_GDW


def growth_rate(df_gDW_l, name, paths):
    df_gDw_GR = df_gDW_l.copy()
    df_gDw_GR['GR-Exp'] = 0.0
    columns_names = list(df_gDW_l)
    # ['Time (h)', 'Oda', 'gDWL']
    df_gDW_l[columns_names[-1]] = df_gDW_l.gDWL.astype(float)
    df_gDW_l[columns_names[0]] = df_gDW_l[columns_names[0]].astype(np.int64)

    x_data = np.array(df_gDW_l[columns_names[0]])
    y_data = np.array(df_gDW_l[columns_names[-1]])

    popt, _ = curve_fit(exponential, x_data, y_data)
    a, k, b = popt
    y_l = exponential(x_data, a, k, b)

    # popt, _ = curve_fit(exponential2, x_data, y_data)
    # a, b = popt
    # y_l = exponential2(x_data, a, b)

    finalDF = pd.DataFrame({'x_ex': [], 'y_ex': [], 'y': [], 'fitted': []})
    finalDF['x_ex'] = x_data
    finalDF['y_ex'] = y_data
    finalDF['y'] = y_l
    columns_names_finalDF = list(finalDF.columns)
    name_Umax = 'Umax - Exp' + name
    fig_Umax = plt.figure(name_Umax)
    ax_Umax = fig_Umax.gca()
    print('uMax: ', 'y = %.5f/(1 + %.5f*exp(-x*%.5f))' % (a, k, b))
    # print('uMax: ', 'y = %.5f*exp(x*%.5f))' % (a, b))

    for index, row in df_gDw_GR.iterrows():
        x = row.at['gDWL']
        df_gDw_GR.at[index, 'GR-Exp'] = b / (1 + a * np.exp(-x * k))
    fitted = df_gDw_GR['GR-Exp']
    finalDF['fitted'] = fitted
    slash = r'/'
    finalDF.plot(kind='line', x=columns_names_finalDF[0], y=columns_names_finalDF[2], ls='--', ax=ax_Umax,
                 color='red', label='y = %.5f/(1 + %.5f*exp(-x*%.5f))' % (a, k, b))

    # finalDF.plot(kind='line', x=columns_names_finalDF[0], y=columns_names_finalDF[-1], ls='--', ax=ax_Umax,
    #             color='red', label='y = %.5f*exp(x*%.5f))' % (a, b))

    finalDF.plot(kind='line', x=columns_names_finalDF[0], y=columns_names_finalDF[1], ax=ax_Umax, label='Biomass')
    # finalDF.plot(kind= 'line', x=columns_names_finalDF[0], y= columns_names_finalDF[-1], ax=ax_Umax,
    # label ='Biomass fitted')
    plt.axis([0, finalDF[columns_names_finalDF[0]].max() + 5, 0,
              finalDF[columns_names_finalDF[2]].max() + 5])
    xticks_Umax = ax_Umax.xaxis.get_major_ticks()
    xticks_Umax[0].label1.set_visible(False)
    plt.ylabel('Biomass (gDW ' + slash + ' L)')
    plt.xlabel('Time (h)')
    plt.grid()

    if name_Umax.find(' ') != -1:
        name_fig_Umax = paths + '\\' + name_Umax.replace(' ', '') + '.pdf'
    else:
        name_fig_Umax = paths + '\\' + name_Umax + '.pdf'
    plt.savefig(name_fig_Umax)

    for index, row in df_gDW_l.iterrows():
        df_gDw_GR.at[index, 'GR'] = ln(row.at[columns_names[-1]])

    y = df_gDw_GR['GR']
    x = df_gDw_GR[columns_names[0]]
    popt, _ = curve_fit(lineal, x, y)
    a, b = popt
    print('y = %.5f * x + %.5f' % (a, b))
    # define a sequence of inputs between the smallest and largest known inputs
    finalDF2 = pd.DataFrame({'x_line': [], 'y_line': [], 'x': [], 'y': []})
    finalDF2['x_line'] = np.arange(min(x), max(x), 1)
    finalDF2['y_line'] = lineal(finalDF2['x_line'], a, b)
    finalDF2['x'] = x
    finalDF2['y'] = y

    columns_names_finalDF2 = list(finalDF2.columns)
    # create a line plot for the mapping function
    name_lineal = 'uMax - lineal' + name
    fig_l = plt.figure(name_lineal)
    ax_l = fig_l.gca()
    finalDF2.plot(kind='line', x=columns_names_finalDF2[0], y=columns_names_finalDF2[1], ls='--', ax=ax_l,
                  color='red', label='y = %.5f * x + %.5f' % (a, b))
    finalDF2.plot(kind='line', x=columns_names_finalDF2[2], y=columns_names_finalDF2[3], ax=ax_l, label='Biomass')
    finalDF2.plot(kind='scatter', x=columns_names_finalDF2[2], y=columns_names_finalDF2[3], ax=ax_l)
    # plt.axis([0, finalDF2[columns_names_finalDF2[0]].max() + 5, 0, finalDF2[columns_names_finalDF2[3]].max()])
    # plt.yscale("log")
    plt.yscale('log')
    xticks = ax_l.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    plt.xlabel('Biomass (gDW ' + slash + ' L)')
    plt.ylabel('Time (h)')
    plt.grid()

    if name_lineal.find(' ') != -1:
        name_fig_l = paths + '\\' + name_lineal.replace(' ', '') + '.pdf'
    else:
        name_fig_l = paths + '\\' + name_lineal + '.pdf'
    plt.savefig(name_fig_l)
    # print(df_gDw_GR)
    df_gDw_GR['mu'] = 0.0
    i = 0
    for index, row in df_gDw_GR.iterrows():
        if i == len(df_gDw_GR) - 1:
            break
        x1 = row.at['gDWL']
        x2 = df_gDw_GR.at[index + 1, 'gDWL']

        df_gDw_GR.at[index, 'mu'] = (ln(x2) - ln(x1)) / (x2 - x1)
        i = i + 1
    #        print(x1, x2)

    return df_gDw_GR
    # plt.show()


def Kmeans(X, nclust):
    model = KMeans(nclust)
    model.fit(X)
    cluster_labels = model.predict(X)
    cent = model.cluster_centers_
    return cluster_labels, cent


if __name__ == '__main__':

    import click


    @click.command()
    @click.option('-bio', '--Bio', 'file_inBiomas', help='Path for getting the data of Biomas')
    @click.option('-ODa', '--ODa', 'file_inODa', help='Path for getting the data of ODa')
    @click.option('-d', '-- One file or directory', 'd', help='push (d) for loading multiple file or (f) for loding '
                                                              'just one file :D')
    @click.option('-o', '--out', 'file_out', help='Name of the file in which data will be stored')
    def main(file_inBiomas, file_inODa, d, file_out):
        print('*** Reading Data ***')
        paths = str(pathlib.Path().absolute()) + '\\' + file_out

        if not os.path.exists(paths):
            os.mkdir(file_out)

        if d == 'd':
            dataPath_bio = gl.glob(file_inBiomas)
            dataPath_ODa = gl.glob(file_inODa)
            for i in range(0, len(dataPath_bio)):
                df_biomas = pd.read_csv(dataPath_bio[i])
                df_ODa = pd.read_csv(dataPath_ODa[i])

        if d == 'f':
            df_biomas = pd.read_csv(file_inBiomas)
            df_ODa = pd.read_csv(file_inODa)

            # To get the file name
            if file_inBiomas.find('\\') != -1:
                name = file_inBiomas.split('\\')
                name = name[-1].split('.')
                name = name[0]
            else:
                name = file_inBiomas.split('\\')
                name = name[-1].split('.')
                name = name[0]
            # print(name)

            columns_names_oda = list(df_ODa.columns)
            for index, row in df_ODa.iterrows():
                if type(row[columns_names_oda[0]]) == str:
                    t = row.at[columns_names_oda[0]]
                    tt = get_h(t)
                    df_ODa.at[index, columns_names_oda[0]] = tt

            plotBIOMAS_ODA_vsTime(df_biomas, df_ODa, name, paths)
            m, b = biomassVSoda(df_biomas, df_ODa, name, paths)
            df_gDW_L = biomass_gDW_L(df_ODa, m, b, name, paths)
            df_gDW_GR = growth_rate(df_gDW_L, name, paths)
            # e(df_gDW_L)

            # print(df_gDW_GR)

            from tslearn.clustering import TimeSeriesKMeans

            model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
            model.fit(df_gDW_GR)
            y_pred = model.fit_predict(df_gDW_GR)
            sz = df_gDW_GR.shape[1]
            print(y_pred)
            df_gDW_GR = df_gDW_GR.fillna(0)
            # ML Part

            from tslearn.clustering import TimeSeriesKMeans
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
                TimeSeriesResampler
            seed = 0
            np.random.seed(seed)
            # np.random.shuffle(df_gDW_GR)
            # Keep only 50 time series
            X_train = TimeSeriesScalerMeanVariance().fit_transform(df_gDW_GR[['Time (h)', 'mu']])
            # Make time series shorter
            # X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
            sz = X_train.shape[1]

            # Euclidean k-means
            print("Euclidean k-means")
            km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
            y_pred = km.fit_predict(X_train)
            print(X_train)
            print(y_pred)

            plt.figure()
            for yi in range(3):
                plt.subplot(3, 3, yi + 1)
                for xx in X_train[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(km.cluster_centers_[yi].ravel(), "r-")
                plt.xlim(0, sz)
                plt.ylim(-4, 4)
                plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                         transform=plt.gca().transAxes)
                if yi == 1:
                    plt.title("Euclidean $k$-means")

            # DBA-k-means
            print("DBA k-means")
            dba_km = TimeSeriesKMeans(n_clusters=3,
                                      n_init=2,
                                      metric="dtw",
                                      verbose=True,
                                      max_iter_barycenter=10,
                                      random_state=seed)
            y_pred = dba_km.fit_predict(X_train)

            for yi in range(3):
                plt.subplot(3, 3, 4 + yi)
                for xx in X_train[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
                plt.xlim(0, sz)
                plt.ylim(-4, 4)
                plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                         transform=plt.gca().transAxes)
                if yi == 1:
                    plt.title("DBA $k$-means")

            # Soft-DTW-k-means
            print("Soft-DTW k-means")
            sdtw_km = TimeSeriesKMeans(n_clusters=3,
                                       metric="softdtw",
                                       metric_params={"gamma": .01},
                                       verbose=True,
                                       random_state=seed)
            y_pred = sdtw_km.fit_predict(X_train)

            for yi in range(3):
                plt.subplot(3, 3, 7 + yi)
                for xx in X_train[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
                plt.xlim(0, sz)
                plt.ylim(-4, 4)
                plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                         transform=plt.gca().transAxes)
                if yi == 1:
                    plt.title("Soft-DTW $k$-means")

            plt.tight_layout()
            # plt.show()

            nclust = 5
            clust_labels_Kmeans, cent = Kmeans(df_gDW_GR[['gDWL', 'mu']], nclust)
            kmeans = pd.DataFrame(clust_labels_Kmeans)
            df_gDW_GR.insert((df_gDW_GR.shape[1]), 'kmeans', kmeans)
            print(df_gDW_GR)

            columns_names_gDW_GR = list(df_gDW_GR.columns)
            targets = [0, 1, 2, 3, 4]
            fig = plt.figure(name)
            ax = fig.add_subplot(2, 1, 2)
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('mu')

            colors = ['tan', 'g', 'y', 'm', 'c']
            for target, color in zip(targets, colors):
                if target == 0:
                    m = '.'
                if target == 1:
                    m = 'v'
                if target == 2:
                    m = '<'
                if target == 3:
                    m = 'X'
                if target == 4:
                    m = 'D'
                indicesToKeep = df_gDW_GR['kmeans'] == target
                # df_gDW_GR_aux = df_gDW_GR.copy()
                # df_gDW_GR_aux.drop(df_gDW_GR_aux[df_gDW_GR_aux['kmeans'] != indicesToKeep].index, inplace=True)
                # df_gDW_GR_aux.plot(kind='line', x=columns_names_gDW_GR[0], y=columns_names_gDW_GR[5], ax=ax,
                #              label='mu', color=indicesToKeep)
                # df_gDW_GR_aux.plot(kind='scatter', x=columns_names_gDW_GR[2], y=columns_names_finalDF2[3], ax=ax_l)
                # indicesToKeep = finalDf['kmeans'] == target
                ax.scatter(df_gDW_GR.loc[indicesToKeep, 'Time (h)']
                           , df_gDW_GR.loc[indicesToKeep, 'mu']
                           , c=color,
                           marker=m
                           )
            for i in range(0, len(targets)):
                targets[i] = targets[i] + 1
            ax.legend(targets)
            plt.grid()

            ax.scatter(df_gDW_GR['Time (h)'], df_gDW_GR['GR'])

            ax = fig.add_subplot(2, 1, 1)
            ax.set_ylabel('gDWL')
            df_gDW_GR.plot(kind='line', x=columns_names_gDW_GR[0], y=columns_names_gDW_GR[2], ax=ax, label='Biomass')

            ax.grid()
            plt.show()



        else:
            print('I do not know what you want, push (d) for loading multiple file or (f) for loding just one file :D')
main()
