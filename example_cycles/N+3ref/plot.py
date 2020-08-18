# --- Python 3.8 ---
# FileName: pressureSweepPlot.py
# Created by: alamkin
# Date: 7/16/20
# Last Updated: 2:21 PM

# --- Imports ---
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import openmdao.api as om
import ast
import pandas as pd

def readData(basefolder):
    """
    Reads the data created by the N3Ref engine pressure sweeps.  Reads all SQL
    files in the directory and then creates and saves a dictionary with the
    necessary values for quicker access.

    Returns
    -------
    None.

    """
    # --- Initialize the data dictionary ---
    data = dict()
    sweep = np.array([0.0, 100.0e3, 200.0e3, 300.0e3, 400.0e3, 500.0e3,
                      600.0e3, 700.0e3, 800.0e3, 900.0e3, 1000.0e3])
    sweep = np.multiply(sweep, 0.0009478171208703)
    sweep = list(sweep)

    data['sweep'] = sweep

    for pt in ['TOC', 'CRZ', 'SLS', 'RTO']:
        # TSFC
        data[pt + '.TSFC'] = []

        #dPqP
        data[pt + '.dPqP'] = []

    # --- iterate through the directory to find SQL files ---
    data_dir = basefolder+r'/pressure_sweeps/raw/'
    if os.path.exists(data_dir):
        dirList = os.listdir(data_dir)
    else:
        print('Path to data directory does not exist')
        return
    # dirList.append(dirList.pop(dirList.index('pressureSweep_1000000.sql')))
    # print(dirList)
    for file in dirList:
        if file.endswith(".sql"):
            print(file)
            cr = om.CaseReader(data_dir + file)
            if 'pressureSweep' in file:
                case = cr.get_case('pressureSweep')
                # loop through design and off-design conditions
                for pt in ['TOC', 'CRZ', 'SLS', 'RTO']:
                    # TSFC
                    data[pt + '.TSFC'].append(float(case.get_val(pt + '.perf.TSFC')))

                    #dPqP
                    data[pt + '.dPqP'].append(float(case.get_val('duct17:dPqP')))

    # --- Create file for the pareto front dictionary data ---
    fp = basefolder+r'/pressure_sweeps/pressureSweepDict.txt'

    # --- Check if the file exists and delete to make new file ---
    if os.path.exists(fp):
        os.remove(fp)

    # --- Make new file and write dictionary data to file ---
    file = open(fp, "w+")
    file.write(str(data))
    file.close()

def plotData():
    """
    Plots the data stored in the pressure sweep dictionary file
    Returns
    -------

    """
    # --- Initialize the directory for plotting ---
    pltDir = r"/home/ben/Documents/data_output/pressure_sweeps"

    # --- initialize filename for the data dictionary ---
    fpData = r"/home/ben/Documents/data_output/pressure_sweeps/pressureSweepDict.txt"

    # --- Check if the data file path exists before attempting to plot ---
    if os.path.exists(fpData):
        # --- create dictionary using ast.literal_eval ---
        dataFile = open(fpData, "r")
        content = dataFile.read()
        dataFile.close()
        data = ast.literal_eval(content)
        # --- Create plot for TSFC ---
        for pt in ['CRZ']:
            fig1, ax1 = plt.subplots()

            ax1.plot(np.array(data[pt + '.dPqP']) - 0.015, data[pt + '.TSFC'], '.')
            ax1.plot(np.array([0.0, 0.06]), np.array([0.45405, 0.45405]), '-')
            ax1.set(xlabel='Total pressure loss fraction (incremental vs baseline)',
                    ylabel='TSFC',
                    title='dPqP vs. TSFC')

            fig1.tight_layout()

            plt.savefig(pltDir + pt + '_tsfc.eps')
            plt.close(pltDir + pt + '_tsfc.eps')


if __name__ == "__main__":
    # read the data and create a new file each time
    # to ensure content is up to date for plots
    # for pwr in [100, 200, 300]:
    #     basefolder = r'/home/ben/Documents/data_output_mdp_'+str(pwr)
    #     readData(basefolder)

    fig1, ax1 = plt.subplots()
    ax1.set(xlabel='Overall bypass (cold) HX total pressure loss factor',
            ylabel='Cruise TSFC (lbf/lbm/hr)',
            title='Cold Side Heat Exchanger Pressure Drop Breakevens')

    colordict = {'100':'r','200':'k','300':'b'}
    for pwr in [100, 200, 300]:
        basefolder = r'/home/ben/Documents/data_output_mdp_'+str(pwr)

        fpData = basefolder+r"/pressure_sweeps/pressureSweepDict.txt"
        # --- Check if the data file path exists before attempting to plot ---
        if os.path.exists(fpData):
            # --- create dictionary using ast.literal_eval ---
            dataFile = open(fpData, "r")
            content = dataFile.read()
            dataFile.close()
            data = ast.literal_eval(content)
            # --- Create plot for TSFC ---
            
            for pt in ['CRZ']:
                df = pd.DataFrame(data={'dPqp':np.array(data[pt + '.dPqP']) - 0.015, 'TSFC':data[pt + '.TSFC']})
                df = df.sort_values('dPqp')
                ax1.plot(df['dPqp'], df['TSFC'], '-', color=colordict[str(pwr)])

    ax1.plot(np.array([0.0, 0.03]), np.array([0.45795, 0.45795]), '-', color=colordict[str(300)])
    ax1.plot(np.array([0.0, 0.03]), np.array([0.45238, 0.45238]), '-', color=colordict[str(200)])
    ax1.plot(np.array([0.0, 0.03]), np.array([0.44682, 0.44682]), '-', color=colordict[str(100)])

    ax1.plot(np.array([0.00523, 0.00523]), np.array([0.437, 0.44682]), '--', color=colordict[str(100)])
    ax1.plot(np.array([0.0103885, 0.0103885]), np.array([0.437, 0.45238]), '--', color=colordict[str(200)])
    ax1.plot(np.array([0.01547, 0.01547]), np.array([0.437, 0.45795]), '--', color=colordict[str(300)])

    ax1.text(0.032, 0.44682, '100kW', color=colordict['100'], verticalalignment='center')
    ax1.text(0.032, 0.45238, '200kW extracted from shaft', color=colordict['200'], verticalalignment='center')
    ax1.text(0.032, 0.45795, '300kW', color=colordict['300'], verticalalignment='center')

    ax1.annotate('Thermoacoustic power ext.', xy=(0.029, 0.485),  xycoords='data',
            xytext=(0.0, 0.51), textcoords='data',
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            horizontalalignment='left', verticalalignment='center',
            )

    ax1.annotate('200kW breakeven', xy=(0.012, 0.44),  xycoords='data',
            xytext=(0.02, 0.44), textcoords='data',
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            horizontalalignment='left', verticalalignment='center',
            )
    # ax1.text(0.0, 0.49, 'Power extracted thermoacoustically', color=colordict['200'], verticalalignment='center')

    # ax1.text(0.0103885+0.001, 0.439, '200kW breakeven', color=colordict['200'], verticalalignment='center')

    fig1.tight_layout()
    plt.ylim(0.437, 0.54)

    plt.show()