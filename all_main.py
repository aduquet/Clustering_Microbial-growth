import os
import pathlib
import warnings
import glob as gl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import log as ln
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.metrics import silhouette_samples, silhouette_score

matplotlib_axes_logger.setLevel('ERROR')
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
    if a < 2:
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

    print('Plot saved in: ', name)
    plt.grid()
    plt.savefig(name)


def plotBIOMAS_ODA_vsTime(df_biomas, df_oda, name, paths):
    """This function is responsible for plotting the directly biomass measurement vs Time, and biomass values based
    in absorbance from the online probe (ODa) vs Time
    :param paths: This is the paths in which all the files generated will be saved
    :param name: This variable contains ID name, which will the ID of the plot generated
    :param df_oda: main DataFrame of the biomass values based in absorbance from the online probe
    :type df_biomas: main DataFrame of the direct biomass measurement"""

    print('\n*** Plotting and saving the directly Biomass vs Time  *** ')

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
    print('\n*** Plotting ans saving biomass values based in absorbance from the online probe (ODa) vs Time  *** ')
    name_oda = 'ODa Vs Time - ' + name
    fig_oda = plt.figure(name_oda)
    ax_oda = fig_oda.gca()
    df_oda.plot(kind='scatter', x=columns_names_oda[0], y=columns_names_oda[1], ax=ax_oda)
    df_oda.plot(kind='line', x=columns_names_oda[0], y=columns_names_oda[1], ax=ax_oda)
    plt.axis([0, df_oda[columns_names_oda[0]].max() + 5, 0, df_oda[columns_names_oda[1]].max() + 5])
    ax_oda.legend_ = None
    xticks_oda = ax_oda.xaxis.get_major_ticks()
    xticks_oda[0].label1.set_visible(False)

    save_plot(name_oda, paths, '.pdf')


def lineal(x, a, b):
    """This function fits a set of points to a lineal model (y= Xm +b)"""
    return a * x + b


def exponential(x, a, k):
    """This function fits a set of points to an Exp model (y = a*Exp(bx)"""
    return a * np.exp(x * k)


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

    y = df_biomas_oda[columns_names_bio[1]]
    poppet, _ = curve_fit(lineal, x, y)
    a, b = poppet  # Coefficients of y = a x + b
    print('\n*** Calibration of ODa x measured biomass (X) *** \n', '\n y = %.5f * x + %.5f' % (a, b), '\n')
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
    save_plot(name_bioVSoda, paths, '.pdf')

    return a, b


def biomass_gDCW_L(df_ODa, m, b, name, paths):
    """ This function is responsible for calculating in-direct biomass based on ODa measurement, i.e., gDCWL = a ODa + b
    :param m: coefficient y = m X+b
    :param b: Coefficient y = m X+b
    :param df_ODa: DataFrame of the biomass values based in absorbance from the online probe
    :param paths: This is the paths in which all the files generated will be saved
    :param name: This variable contains ID name, which will the ID of the plot generated """
    print('\n*** Computing in-direct biomass based on ODa (gDCWL = x ODa + b) *** \n')
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
    save_plot(name_ODa_biomass_GDW, paths, '.pdf')

    return df_ODa_biomass_GDW


def mu(df_gDCW, name, paths):
    """ This function is responsible for calculating mu, i.e., gDCWL = a ODa + b
         :param df_gDCW: Main DataFrame which should contain the following columns: Time (h), Oda, gDCWL
         :param name: This variable contains ID name, which will the ID of the plot generated
         :param paths: This is the paths in which all the files generated will be saved """

    print('\n*** Computing mu *** \n')
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

    ax1_bio.set_xlabel('Time (h)')
    ax1_bio.set_ylabel('Biomass (gdcw/L)')
    ax1_bio.plot(t, bio)  # in case you want to change the plot color, add color= 'some color'
    ax1_bio.tick_params(axis='y')
    xticks_bio = ax1_bio.xaxis.get_major_ticks()
    xticks_bio[0].label1.set_visible(False)

    ax2_mu = ax1_bio.twinx()  # instantiate a second axes that shares the same x-axis
    ax2_mu.set_ylabel('μ (1/h)')  # we already handled the x-label with ax1
    ax2_mu.scatter(t, abs(mu), color='peru', s=15)
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

    ax1_bio.set_xlabel('Time (h)')
    ax1_bio.set_ylabel('Biomass (gdcw/L)')
    ax1_bio.plot(t, bio)  # in case you want to change the plot color, add color= 'some color'
    ax1_bio.tick_params(axis='y')
    xticks_bio = ax1_bio.xaxis.get_major_ticks()
    xticks_bio[0].label1.set_visible(False)

    ax2_mu = ax1_bio.twinx()  # instantiate a second axes that shares the same x-axis
    ax2_mu.set_ylabel('μ (1/h)')  # we already handled the x-label with ax1
    ax2_mu.scatter(t[2:], abs(mu[2:]), color='peru', s=15)
    ax2_mu.tick_params(axis='y')
    fig_bio.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1_bio.set_xlim(0.0, len(t))
    ax1_bio.set_ylim(ymin=0)
    ax2_mu.set_ylim(ymin=0)
    plt.grid()
    save_plot(name_mu, paths, '.pdf')

    return df_gDCW_mu


def Kmeans(X, n_clusters):
    model = KMeans(n_clusters)
    model.fit(X)
    cluster_labels = model.predict(X)
    cent = model.cluster_centers_
    inertia = model.inertia_

    return cluster_labels, cent, inertia


def k_optimal_silhouetteCoffin(x, ini):
    range_n_clusters = np.arange(ini, 15)  # Possibles K optimos
    aux = []
    for n_clusters in range_n_clusters:
        label, cent, inert = Kmeans(x, n_clusters)
        silhouette_avg = silhouette_score(x, label)
        aux.append(silhouette_avg)
    return range_n_clusters, aux


def k_optimal_inertia(x):
    range_n_clusters = np.arange(1, 15)
    inertia_vec = []
    for i in range(1, len(range_n_clusters)):
        label, cent, inertia = Kmeans(x[['Time (h)', 'mu']], i)
        inertia_vec.append(inertia)
    return inertia_vec


def R_sq_optimiser(df, R_sq_old):
    global a, r_df, n_df, inter, slope
    model = LinearRegression()
    ind = 0
    for i in range(1, len(df)):
        n_df = df.copy()
        n_df = n_df[i:]

        y = n_df['gDCWL-ln'].array
        x = n_df['Time (h)'].to_numpy()
        x = x.reshape((-1, 1))
        model.fit(x, y)
        R_sq = model.score(x, y)
        if R_sq > R_sq_old:
            R_sq_old = R_sq
            r_df = n_df[ind:].copy()
            inter = model.intercept_
            slope = model.coef_
            a = i
        else:
            break

    for j in reversed(np.arange(a, len(r_df))):
        n_df = r_df.copy()
        n_df = n_df[:j]
        y = n_df['gDCWL-ln'].array
        x = n_df['Time (h)'].to_numpy()
        x = x.reshape((-1, 1))
        model.fit(x, y)
        R_sq = model.score(x, y)

        if R_sq > R_sq_old:
            R_sq_old = R_sq
            inter = model.intercept_
            slope = model.coef_
        else:
            break

    return R_sq_old, slope, inter, n_df


def mu_Max(df_gDCW_mu, name, paths):
    """ This function is responsible for calculating in-direct biomass based on ODa measurement, i.e., gDCWL = a ODa + b
     :param df_gDCW_mu: Main DataFrame which should contain the following columns: Time (h), Oda, gDCWL, and mu
     :param name: This variable contains ID name, which will the ID of the plot generated
     :param paths: This is the paths in which all the files generated will be saved """

    print('\n*** Computing mu_Max *** \n')

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
    plt.ylabel('Biomass (gDCW ' + slash + ' L) (log_10)')
    plt.grid()
    print('Computing and saving Log_10 Biomass (gDCW\L)')

    range_n_clusters, opt_k = k_optimal_silhouetteCoffin(log10_Oda[['Time (h)', 'gDCWL']], 2)
    n_clusters = range_n_clusters[opt_k.index(np.max(opt_k))]

    cluster_labels, cent, inert = Kmeans(log10_Oda[['gDCWL', 'mu']], n_clusters)
    kmeans = pd.DataFrame(cluster_labels)
    log10_Oda.insert((log10_Oda.shape[1]), 'regions', kmeans)

    targets = np.arange(0, log10_Oda['regions'].max() + 1)
    ax.set_xlabel('Time (h)')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))
    f_leg = 'Biomass (gDCW' + slash + ' L)'
    leg = [f_leg]

    for target, colors in zip(targets, colors):
        leg_name = 'Region ' + str(int(target + 1))
        leg.append(leg_name)
        indicesToKeep = log10_Oda['regions'] == target
        ax.scatter(log10_Oda.loc[indicesToKeep, 'Time (h)']
                   , log10_Oda.loc[indicesToKeep, 'gDCWL']
                   , color=colors,
                   marker='.', alpha=0.8)
        ax.set_yscale("log", nonposy='clip')

    ax.legend(leg)

    save_plot(name, paths, '.pdf')
    # Lineal Model
    model = LinearRegression()
    R_sq = []
    inter = []
    slope = []

    for i in range(0, n_clusters):
        reg = log10_Oda.copy()
        reg.drop(reg[reg['regions'] != i].index, inplace=True)  # Sub-select the rows belonging to the region 1 only
        y = reg['gDCWL-ln'].array
        x = reg['Time (h)'].to_numpy()
        x = x.reshape((-1, 1))
        model.fit(x, y)

        R_sq.append(model.score(x, y))
        inter.append(model.intercept_)
        slope.append(model.coef_)

    region = slope.index(np.max(slope))
    R_sq_final = R_sq[region]
    bestRegion = log10_Oda.copy()
    bestRegion.drop(bestRegion[bestRegion['regions'] != region].index, inplace=True)

    R_sq_opt, slope_opt, inter_opt, df_final = R_sq_optimiser(bestRegion, R_sq_final)
    # print(R_sq_opt, slope_opt, inter_opt, df_final)
    # print(R_sq, slope, inter)
    # print(region, R_sq_final, slope_final, inter_final)
    print('\n*** Finding optimal mu_Max *** \n', '\n y = %.5f * x + %.5f' % (slope_opt, inter_opt), '\n')
    print('mu_Max: ', slope_opt)
    print('\nR-squared: ', R_sq_opt, '\n')
    columns_names = list(log10_Oda.columns)
    name = 'mu_Max' + name
    fig_muMax = plt.figure(name)
    ax_muMax = fig_muMax.gca()
    bestRegion.plot(kind='line', x=columns_names[0], y=columns_names[2], ax=ax_muMax, label='Biomass')
    df_final.plot(kind='scatter', x=columns_names[0], y=columns_names[2], ax=ax_muMax, marker='.', label='y = %.5f '
                                                                                                         '* x + '
                                                                                                         '%.5f '
                                                                                                         '\n\nR'
                                                                                                         '-squared '
                                                                                                         '= %.5f '
                                                                                                         '\n'
                                                                                                         '\nmu_Max '
                                                                                                         '= %.5f '
                                                                                                         % (slope_opt,
                                                                                                            inter_opt,
                                                                                                            R_sq_opt,
                                                                                                            slope_opt),
                  color='red', s=40)
    ax_muMax.set_yscale("log", nonposy='clip')
    ax_muMax.set_xlabel('Time (H)')
    ax_muMax.set_ylabel('Biomass (gdcw/L) (log_10)')
    plt.axis([0, bestRegion[columns_names[0]].max() + 5, 0,
              bestRegion[columns_names[2]].max() + 5])
    xticks_ = ax_muMax.xaxis.get_major_ticks()
    xticks_[0].label1.set_visible(False)

    save_plot(name, paths, '.pdf')
    name = paths + '\\' + name.replace(' ', '') + '.csv'
    print('saving data in csv format: ', name)
    log10_Oda.to_csv(name)
    muMax = slope_opt

    return muMax, log10_Oda


def growthPhases_mu(muMax, df_gDCW_final, name, paths):
    print('\n*** Computing growth phases *** \n')
    df_gDCW_final = df_gDCW_final[1:]
    df_gDCW_final = df_gDCW_final.fillna(0)
    k_opt = k_optimal_inertia(df_gDCW_final)
    k2_opt, vec = k_optimal_silhouetteCoffin(df_gDCW_final[['Time (h)', 'mu']], 4)
    n_clusters = k2_opt[vec.index(np.max(vec))]
    cluster_labels, cent, inert = Kmeans(df_gDCW_final[['Time (h)', 'mu', 'gDCWL']], n_clusters)
    kmeans = pd.DataFrame(cluster_labels)
    df_gDCW_final.insert((df_gDCW_final.shape[1]), 'GR-Phase', kmeans)

    columns_names = list(df_gDCW_final.columns)
    name = 'Growth Phases based on mu - ' + name
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(name)
    slash = r'/'
    targets = np.arange(0, df_gDCW_final['GR-Phase'].max() + 1)
    ax.set_xlabel('Time (h)')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))

    t = df_gDCW_final['Time (h)']
    bio = df_gDCW_final['gDCWL']

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Biomass (gdcw/L)')
    ax.plot(t, bio, label='Biomass (gDCW ' + slash + ' L)')  # in case you want to change the plot color, add color=
    # 'some color'
    leg1 = ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(0., 1.02, 1., .102), ncol=1, borderaxespad=0)
    ax.tick_params(axis='y')
    xticks_bio = ax.xaxis.get_major_ticks()
    xticks_bio[0].label1.set_visible(False)

    ax2_mu = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2_mu.set_ylabel('μ (1/h)')  # we already handled the x-label with ax1
    leg = []
    for target, colors in zip(targets, colors):
        # leg_name = 'Phase ' + str(int(target + 1))
        # leg.append(leg_name)
        indicesToKeep = df_gDCW_final['GR-Phase'] == target
        ax2_mu.scatter(df_gDCW_final.loc[indicesToKeep, 'Time (h)']
                       , df_gDCW_final.loc[indicesToKeep, 'mu']
                       , color=colors,
                       marker='.', alpha=0.8)
    #    ax.set_yscale("log", nonposy='clip')
    ax2_mu.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax.set_xlim(0.0, len(t))
    ax2_mu.set_ylim(ymin=0)

    # ax.legend('Biomass (gDCW ' + slash + ' L)')
    # ax2_mu.legend(leg, loc='lower right', frameon=False, bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, borderaxespad=0)
    plt.grid()
    save_plot(name, paths, '.pdf')
    name = paths + '\\' + name.replace(' ', '') + '.csv'
    print('saving final data in csv format : ', name)
    df_gDCW_final.to_csv(name)


if __name__ == '__main__':

    import click


    @click.command()
    @click.option('-bio', '--Bio', 'file_inBiomas', help='Path for getting the data of Biomas')
    @click.option('-ODa', '--ODa', 'file_inODa', help='Path for getting the data of ODa')
    @click.option('-d', '-- One file or directory', 'd', help='write (dir) for loading multiple files or (f) for '
                                                              'loading just one file :D')
    @click.option('-o', '--out', 'file_out', help='Name of the file in which data will be stored')
    def main(file_inBiomas, file_inODa, d, file_out):
        print('\n*** Reading Data ***')
        paths = str(pathlib.Path().absolute()) + '\\' + file_out

        if not os.path.exists(paths):
            os.mkdir(file_out)

        if d == 'dir':
            dataPath_bio = gl.glob(file_inBiomas)
            dataPath_ODa = gl.glob(file_inODa)

            for i in range(0, len(dataPath_bio)):
                df_biomas = pd.read_csv(dataPath_bio[i])
                df_ODa_ODa = pd.read_csv(dataPath_ODa[i])

                name = dataPath_bio[i].split('\\')
                name_bio = name[-1].split('.')[0]

                name = dataPath_ODa[i].split('\\')
                name_ODa = name[-1].split('.')[0]
                name_ODa_name = name_ODa.split('_ODa')[0]

                if name_ODa_name == name_bio:
                    paths_f = paths + '\\' + name_bio

                    if not os.path.exists(paths_f):
                        os.mkdir(file_out + '\\' + name_bio)

                    df_ODa = df_ODa_ODa[['Time (h)', 'Oda']]

                    columns_names_oda = list(df_ODa.columns)
                    for index, row in df_ODa.iterrows():
                        if type(row[columns_names_oda[0]]) == str:
                            t = row.at[columns_names_oda[0]]
                            tt = get_h(t)
                            df_ODa.at[index, columns_names_oda[0]] = tt

                    plotBIOMAS_ODA_vsTime(df_biomas, df_ODa, name_bio,
                                          paths_f)  # Call the function to plot direct biomass vs time,
                    # and also the in-direct biomass vs time

                    m, b = biomassVSoda(df_biomas, df_ODa, name_bio,
                                        paths_f)  # m and b are the coefficients of y = mX + b, that
                    # are necessaries to convert ODa into biomass (g/L)

                    df_gDCW_L = biomass_gDCW_L(df_ODa, m, b, name_bio,
                                               paths_f)  # Call the function to convert ODa into biomass
                    df_gDCW_L_mu = mu(df_gDCW_L, name_bio,
                                      paths_f)  # Call the function to compute μ (1/h) by using ln(y2-y1)/t2-t1
                    # and plot μ (1/h) vs time (h), and also biomass (gDCW/l) vs time

                    muMax, df_gDCW_final = mu_Max(df_gDCW_L_mu, name_bio,
                                                  paths_f)  # Call the function to compute mu (growth rate)
                    # based on in-direct biomass(df_gDW_L)
                    # Finding Growth Phases
                    growthPhases_mu(muMax, df_gDCW_final, name_bio, paths_f)

        elif d == 'f':
            df_biomas = pd.read_csv(file_inBiomas)
            df_ODa_ODa = pd.read_csv(file_inODa)

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

            muMax, df_gDCW_final = mu_Max(df_gDCW_L_mu, name, paths)  # Call the function to compute mu (growth rate)
            # based on in-direct biomass(df_gDW_L)
            # Finding Growth Phases
            growthPhases_mu(muMax, df_gDCW_final, name, paths)
            # silh(df_gDCW_GR)

            # plt.show()

        else:
            print('Push (d) for loading multiple file or (f) for loding just one file :D')
main()
