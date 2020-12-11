import os
import pathlib
import pandas as pd
import numpy as np
import glob as gl
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import log as ln
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from yellowbrick.cluster import kelbow_visualizer, KElbowVisualizer
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')

import warnings

warnings.filterwarnings('ignore')

"""
Configuration of the Font size and style for all the plots
"""

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})


def get_h(time_str):
    """This function is to convert the time format to integers, more specifically hours to integers"""
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


def save_plot(name, paths, format_fig):
    if name.find(' ') != -1:
        name = paths + '\\' + name.replace(' ', '') + format_fig
    else:
        name = paths + '\\' + name + format_fig
    plt.savefig(name)


def plotBIOMAS_ODA_vsTime(df_biomas, df_oda, name, paths):
    """This function is responsible for plotting the directly biomass measurement vs Time, and biomass values based
    in absorbance from the online probe (ODa) vs Time
    :param paths: This is the paths in which all the files generated will be saved
    :param name: This variable contains ID name, which will the ID of the plot generated
    :param df_oda: main DataFrame of the biomass values based in absorbance from the online probe
    :type df_biomas: main DataFrame of the direct biomass measurement"""

    columns_names = list(df_biomas.columns)
    columns_names_oda = list(df_oda.columns)

    name_bio = 'Biomass Vs Time - ' + name
    fig = plt.figure(name_bio)
    ax = fig.gca()
    df_biomas.plot(kind='scatter', x=columns_names[0], y=columns_names[1], ax=ax)
    df_biomas.plot(kind='line', x=columns_names[0], y=columns_names[1], ax=ax)
    plt.axis([0, df_biomas[columns_names[0]].max() + 5, 0, df_biomas[columns_names[1]].max() + 2])
    ax.legend_ = None
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    # plt.grid()
    save_plot(name_bio, paths, '.pdf')

    name_oda = 'ODa Vs Time - ' + name
    fig_oda = plt.figure(name_oda)
    ax_oda = fig_oda.gca()
    df_oda.plot(kind='scatter', x=columns_names_oda[0], y=columns_names_oda[1], ax=ax_oda)
    df_oda.plot(kind='line', x=columns_names_oda[0], y=columns_names_oda[1], ax=ax_oda)
    plt.axis([0, df_oda[columns_names_oda[0]].max() + 5, 0, df_oda[columns_names_oda[1]].max() + 5])
    ax_oda.legend_ = None
    xticks_oda = ax_oda.xaxis.get_major_ticks()
    xticks_oda[0].label1.set_visible(False)
    # plt.grid()
    save_plot(name_oda, paths, '.pdf')


def lineal(x, a, b):
    """This function fits a set of points to a lineal model (y= Xm +b)"""
    return a * x + b


def exponential(x, a, k, b):
    """This function fits a set of points to an Exp model (y = b / 1+ a*exp(-x*K)"""
    return b / (1 + a * np.exp(-x * k))


def exponential2(x, a, b):
    """This function fits a set of points to an Exp model (y = a*Exp(bx)"""
    y = a*np.exp(x*b)
    return y

def biomassVSoda(df_biomas, df_oda, name, paths):
    """ This function is responsible for calculating in-direct biomass based on ODa measurement, i.e., gDCWL = a ODa
    + b
    :param paths: This is the paths in which all the files generated will be saved
    :param name: This variable contains ID name, which will the ID of the plot generated
    :param df_oda: main DataFrame of the biomass values based in absorbance from the online probe
    :type df_biomas: main DataFrame of the direct biomass measurement"""

    columns_names_bio = list(df_biomas.columns)
    columns_names_oda = list(df_oda.columns)
    df_biomas_oda = df_biomas.copy()
    df_biomas_oda['ODa'] = 0.0

    for index, row in df_biomas.iterrows():
        t = round(row.at[columns_names_bio[0]])  # if it is greater than .5 it is rounded up
        aux = df_oda.copy()
        aux.drop(aux[aux[columns_names_oda[0]] != t].index, inplace=True)
        df_biomas_oda.at[index, 'ODa'] = aux[columns_names_oda[1]]

    x = df_biomas_oda['ODa']
    print('asdfasd', columns_names_bio)
    y = df_biomas_oda[columns_names_bio[1]]
    poppet, _ = curve_fit(lineal, x, y)
    a, b = poppet  # Coefficients of y = a x + b
    print('*** Calibration of ODa x measured biomass (X) *** \n', '\n y = %.5f * x + %.5f' % (a, b), '\n')
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
    # plt.grid()


    save_plot(name_bioVSoda, paths, '.pdf')

    return a, b


def biomass_gDCW_L(df_ODa, m, b, name, paths):
    """ This function is responsible for calculating in-direct biomass based on ODa measurement, i.e., gDCWL = a ODa + b
    :param m: coefficient y = m X+b
    :param b: Coefficient y = m X+b
    :param df_ODa: DataFrame of the biomass values based in absorbance from the online probe
    :param paths: This is the paths in which all the files generated will be saved
    :param name: This variable contains ID name, which will the ID of the plot generated """

    df_ODa_biomass_GDW = df_ODa.copy()
    df_ODa_biomass_GDW['gDCWL'] = 0.0  # create a new columns named gDCWL to store the in direct biomass, i.e.,
    # biomass based on ODa
    ODa_columns = list(df_ODa_biomass_GDW.columns)
    for index, row in df_ODa.iterrows():
        x = row.at[ODa_columns[1]]
        df_ODa_biomass_GDW.at[index, ODa_columns[-1]] = (m * x) + b

    columns_names_ODa_biomass_GDW = list(df_ODa_biomass_GDW.columns)
    name_ODa_biomass_GDW = 'Biomass gDCWL Vs Time - ' + name
    fig_ODa_biomass_GDW = plt.figure(name_ODa_biomass_GDW)
    ax_ODa_biomass_GDW = fig_ODa_biomass_GDW.gca()

    df_ODa_biomass_GDW.plot(kind='line', x=columns_names_ODa_biomass_GDW[0], y=columns_names_ODa_biomass_GDW[-1],
                            ax=ax_ODa_biomass_GDW)
    plt.axis([0, df_ODa_biomass_GDW[columns_names_ODa_biomass_GDW[0]].max() + 5, 0,
              df_ODa_biomass_GDW[columns_names_ODa_biomass_GDW[-1]].max() + 5])
    ax_ODa_biomass_GDW.legend_ = None
    xticks_ODa_biomass_GDW = ax_ODa_biomass_GDW.xaxis.get_major_ticks()
    xticks_ODa_biomass_GDW[0].label1.set_visible(False)
    slash = r'/'
    plt.ylabel('Biomass (gDCW ' + slash + ' L)')
    # plt.grid()
    save_plot(name_ODa_biomass_GDW, paths, '.pdf')

    return df_ODa_biomass_GDW


def mu(df_gDCW, name, paths):
    """ This function is responsible for calculating mu, i.e., gDCWL = a ODa + b
         :param df_gDCW: Main DataFrame which should contain the following columns: Time (h), Oda, gDCWL
         :param name: This variable contains ID name, which will the ID of the plot generated
         :param paths: This is the paths in which all the files generated will be saved """

    df_gDCW_mu = df_gDCW.copy()
    df_gDCW_mu['mu'] = 0.0
    i = 0
    for index, row in df_gDCW.iterrows():
        if i == len(df_gDCW) - 1:
            break
        x1 = row.at['gDCWL']
        x2 = df_gDCW.at[index + 1, 'gDCWL']
        t1 = row.at['Time (h)']
        t2 = df_gDCW.at[index + 1, 'Time (h)']

        df_gDCW_mu.at[index, 'mu'] = (ln(x2) - ln(x1)) / (t2 - t1)
        i = i + 1

    name_mu = ' gDCW_mu vs Time (All data) -' + name
    fig_bio, ax1_bio = plt.subplots()
    fig_bio.canvas.set_window_title(name_mu)
    t = df_gDCW_mu['Time (h)']
    mu = df_gDCW_mu['mu']
    bio = df_gDCW_mu['gDCWL']

    ax1_bio.set_xlabel('Time (H)')
    ax1_bio.set_ylabel('Biomass (gdcw/L)')
    ax1_bio.plot(t, bio)  # in case you want to change the plot color, add color= 'some color'
    ax1_bio.tick_params(axis='y')
    xticks_bio = ax1_bio.xaxis.get_major_ticks()
    xticks_bio[0].label1.set_visible(False)

    ax2_mu = ax1_bio.twinx()  # instantiate a second axes that shares the same x-axis
    ax2_mu.set_ylabel('μ (1/h)')  # we already handled the x-label with ax1
    ax2_mu.scatter(t, mu, color='peru', s=15)
    ax2_mu.tick_params(axis='y')
    fig_bio.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1_bio.set_xlim(0.0, len(t))
    ax1_bio.set_ylim(ymin=0)
    ax2_mu.set_ylim(ymin=0)
    plt.grid()

    save_plot(name_mu, paths, '.pdf')

    # plot all without outlier
    name_mu = ' gDCW_mu vs Time -' + name
    fig_bio, ax1_bio = plt.subplots()
    fig_bio.canvas.set_window_title(name_mu)
    t = df_gDCW_mu['Time (h)']
    mu = df_gDCW_mu['mu']
    bio = df_gDCW_mu['gDCWL']

    ax1_bio.set_xlabel('Time (H)')
    ax1_bio.set_ylabel('Biomass (gdcw/L)')
    ax1_bio.plot(t, bio)  # in case you want to change the plot color, add color= 'some color'
    ax1_bio.tick_params(axis='y')
    xticks_bio = ax1_bio.xaxis.get_major_ticks()
    xticks_bio[0].label1.set_visible(False)

    ax2_mu = ax1_bio.twinx()  # instantiate a second axes that shares the same x-axis
    ax2_mu.set_ylabel('μ (1/h)')  # we already handled the x-label with ax1
    ax2_mu.scatter(t[2:], mu[2:], color='peru', s=15)
    ax2_mu.tick_params(axis='y')
    fig_bio.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1_bio.set_xlim(0.0, len(t))
    ax1_bio.set_ylim(ymin=0)
    ax2_mu.set_ylim(ymin=0)
    plt.grid()

    plt.savefig(name_mu)

    return df_gDCW_mu


def Kmeans(X, n_clusters):
    model = KMeans(n_clusters)
    model.fit(X)
    cluster_labels = model.predict(X)
    cent = model.cluster_centers_
    return cluster_labels, cent


def mu_Max(df_gDCW_mu, name, paths):
    """ This function is responsible for calculating in-direct biomass based on ODa measurement, i.e., gDCWL = a ODa + b
     :param df_gDCW_mu: Main DataFrame which should contain the following columns: Time (h), Oda, gDCWL, and mu
     :param name: This variable contains ID name, which will the ID of the plot generated
     :param paths: This is the paths in which all the files generated will be saved """
    print(df_gDCW_mu.keys())

    log10_Oda = df_gDCW_mu.copy()
    log10_Oda['gDCWL-ln'] = 0.0

    for index, row in log10_Oda.iterrows():
        a = row.at['gDCWL']
        if a != 0:
            log10_Oda.at[index, 'gDCWL-ln'] = np.abs(ln(a))

        else:
            continue
    log10_Oda = log10_Oda.fillna(0)
    log10_Oda.drop(log10_Oda[log10_Oda['Time (h)'] == 0].index, inplace=True)

    columns_names = list(log10_Oda.columns)
    name = 'Biomass gDCW_L Vs Time (log10) - ' + name
    fig = plt.figure(name)
    ax = fig.gca()

    log10_Oda.plot(kind='line', x=columns_names[0], y=columns_names[2], ax=ax)
    ax.set_yscale("log", nonposy='clip')
    plt.axis([0, log10_Oda[columns_names[0]].max(), 0, log10_Oda[columns_names[2]].max()])
    ax.legend_ = None
    xticks_ = ax.xaxis.get_major_ticks()
    xticks_[0].label1.set_visible(False)
    slash = r'/'
    plt.ylabel('Biomass (gDCW ' + slash + ' L)')
    # plt.grid()
    save_plot(name, paths, '.pdf')

    n_clusters = 3
    cluster_labels, cent = Kmeans(log10_Oda[['gDCWL', 'mu']], n_clusters)
    kmeans = pd.DataFrame(cluster_labels)
    log10_Oda.insert((log10_Oda.shape[1]), 'regions', kmeans)

    targets = [0, 1, 2]
    ax.set_xlabel('Time (h)')
    colors = ['coral', 'g', 'y']
    leg = ['Biomass', 'Reg 1', 'Reg 2', 'Reg 3']
    for target, colors in zip(targets, colors):
        if target == 0:
            m = '.'
        if target == 1:
            m = 'v'
        if target == 2:
            m = '<'

        indicesToKeep = log10_Oda['regions'] == target
        ax.scatter(log10_Oda.loc[indicesToKeep, 'Time (h)']
                   , log10_Oda.loc[indicesToKeep, 'gDCWL']
                   , color=colors,
                   marker=m)
        ax.set_yscale("log", nonposy='clip')
    for i in range(0, len(targets)):
        targets[i] = targets[i] + 1
    ax.legend(leg)

    reg1 = log10_Oda.copy()
    reg1.drop(reg1[reg1['regions'] != 0].index, inplace=True)
    y = reg1['gDCWL-ln'].array
    y_exp = reg1['gDCWL-ln']
    x = reg1['Time (h)'].to_numpy()
    x = x.reshape((-1, 1))
    x_exp = reg1['Time (h)']
    model = LinearRegression()
    model.fit(x, y)
    new_y_reg1 = model.predict(x)
    ax.scatter(x, new_y_reg1, color='r', s=40, marker='.')

    reg1_r_sq = model.score(x, y)
    reg1_inter = model.intercept_
    reg1_slope = model.coef_
    print('Region 1 Rsq', reg1_r_sq)
    print('Region 1 intercept', reg1_inter)
    print('Region 1 slope', reg1_slope)

    reg2 = log10_Oda.copy()
    reg2.drop(reg2[reg2['regions'] != 1].index, inplace=True)
    x_reg2 = reg2['Time (h)'].to_numpy()
    x_reg2_exp = reg2['Time (h)']
    x_reg2 = x_reg2.reshape((-1, 1))
    y_reg2 = reg2['gDCWL-ln'].array
    y_reg2_exp = reg2['gDCWL-ln']
    model = LinearRegression()
    model.fit(x_reg2, y_reg2)

    reg2_r_sq = model.score(x_reg2,y_reg2)
    reg2_inter = model.intercept_
    reg2_slope = model.coef_
    new_y_reg2 = model.predict(x_reg2)
    ax.scatter(x_reg2, new_y_reg2, color='r', s=40, marker='.')
    print('Region 2 Rsq', reg2_r_sq)
    print('Region 2 intercept', reg2_inter)
    print('Region 2 slope', reg2_slope)

    reg3 = log10_Oda.copy()
    reg3.drop(reg3[reg3['regions'] != 2].index, inplace=True)
    x_reg3 = reg3['Time (h)'].to_numpy()
    x_reg3_exp = reg3['Time (h)']
    x_reg3 = x_reg3.reshape((-1, 1))
    y_reg3 = reg3['gDCWL-ln'].array
    y_reg3_exp = reg3['gDCWL-ln']
    model = LinearRegression()
    model.fit(x_reg3, y_reg3)
    new_y_reg3 = model.predict(x_reg3)
    ax.scatter(x_reg3, new_y_reg3, color='r', s=40, marker='.')
    ax.set_yscale('log')
    reg3_r_sq = model.score(x_reg3, y_reg3)
    reg3_inter = model.intercept_
    reg3_slope = model.coef_
    print('Region 3 Rsq', reg3_r_sq)
    print('Region 3 intercept', reg3_inter)
    print('Region 3 slope', reg3_slope)

    popt1, pcov = curve_fit(exponential2, x_exp, y_exp)
    a1, b1 =popt1

    popt2, pcov2 = curve_fit(exponential2, x_reg2_exp, y_reg2_exp)
    print(popt2)
    a2, b2 = popt2

    popt3, pcov3 = curve_fit(exponential2, x_reg3_exp, y_reg3_exp)
    a3, b3 = popt3

    fig2 = plt.figure('Exp fit')
    ax2 = fig2.gca()
    ax2.plot(x_exp, y_exp, color='coral')
    ax2.plot(x_reg2_exp, y_reg2_exp, color='g')
    ax2.plot(x_reg3_exp, y_reg3_exp, color='y')
    ax2.scatter(log10_Oda['Time (h)'], log10_Oda['gDCWL'], color='r', s=40, marker='.')
    ax2.set_yscale("log", nonposy='clip')

    print('Reg 1:', a1, b1)
    print('Reg 2:', a2, b2)
    print('Reg 3:', a3, b3)
    # plt.plot(x_exp, exponential2(x_exp, a, b), 'r-')
    #save_plot(name, paths, '.pdf')
    log10_Oda.to_csv('final2.csv')
    muMax = 0

    return muMax


def silh(df_gDCW_GR):
    range_n_cluster = list(range(2, 10))
    aux = []
    # fig_kelbow = plt.figure('kelbow')
    # axK = fig_kelbow.gca()
    # kelbow_visualizer(KMeans(random_state=4), df_gDCW_GR[['Time (h)', 'gDCWL', 'mu']], k=(2, 10), ax=axK)
    # mod = KEL
    print()

    for n_clusters in range_n_cluster:
        k_means = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = k_means.fit_predict(df_gDCW_GR)
        silhouette_avg = silhouette_score(df_gDCW_GR[['Time (h)', 'gDCWL', 'mu']], cluster_labels)
        aux.append(silhouette_avg)
    print(aux)


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
            df_ODa_ODa = pd.read_csv(file_inODa)
            print(df_ODa_ODa.keys())
            df_ODa = df_ODa_ODa[['Time (h)', 'Oda']]
            # To get the file name
            if file_inBiomas.find('\\') != -1:
                name = file_inBiomas.split('\\')
                name = name[-1].split('.')
                name = name[0]

            else:
                name = file_inBiomas.split('/')
                name = name[-1]
                name = name.split('.')
                name = name[0]
                print(name)

            columns_names_oda = list(df_ODa.columns)
            for index, row in df_ODa.iterrows():
                if type(row[columns_names_oda[0]]) == str:
                    t = row.at[columns_names_oda[0]]
                    tt = get_h(t)
                    df_ODa.at[index, columns_names_oda[0]] = tt

            plotBIOMAS_ODA_vsTime(df_biomas, df_ODa, name, paths)  # Call the function to plot direct biomass vs time,
            # and also the in-direct biomass vs time

            m, b = biomassVSoda(df_biomas, df_ODa, name, paths)  # m and b are the coefficients of y = mX + b, that
            # are necessaries to convert ODa into biomass (g/L)

            df_gDCW_L = biomass_gDCW_L(df_ODa, m, b, name, paths)  # Call the function to convert ODa into biomass
            df_gDCW_L_mu = mu(df_gDCW_L, name, paths)  # Call the function to compute μ (1/h) by using ln(y2-y1)/t2-t1
            # and plot μ (1/h) vs time (h), and also biomass (gDCW/l) vs time

            df_gDCW_muMax = mu_Max(df_gDCW_L_mu, name, paths)  # Call the function to compute mu (growth rate) based on
            # in-direct biomass(df_gDW_L)

            # Finding Growth Phases

            # silh(df_gDCW_GR)

            plt.show()

        else:
            print('I do not know what you want, push (d) for loading multiple file or (f) for loding just one file :D')
main()
