#! usr/bin/env python

import pandas as pd
import numpy as np
import scipy.stats.stats as sci_stats
import matplotlib.pyplot as plt
from tkinter import Button, mainloop, X, Tk
from tkinter.filedialog import askopenfilename

'''
FindFile:
    provides a tkinter popup dialog box that makes getting filepaths easy

    Inputs:
        text: a string of text that appears on the button for the file dialog
              box. This is useful for remebering what file that needs selected

    Outputs:
        Fname: a string of the filepath for the chosen file.
'''


def FindFile(text):
    def openFile():
        global Fname
        Fname = askopenfilename()
        print(Fname)
        root.destroy()

    root = Tk()
    root.attributes("-topmost", True)
    Button(root, text=text, command=openFile).pack(fill=X)
    mainloop()

    return Fname


'''
grouping:
    Sort the data in a Pandas DataFrame into histogram type bins then find the
    mean and standar deviation of the data in the bins.

    Inputs:

        data:         a Pandas DataFrame on which the operations are preformed
        group_column: the column name of the data to be used for sorting
        bins:         a list or array that defines the histogram bin intervals

    Outputs:

        processed:    a dictionary containing with three data sets defined by
                      the keys; data, mean, std. 'data' contains the sorted raw
                      data. 'mean' contains the average of the bin values.
                      'std' contains the standard deviation of the bin values
'''


def grouping(data, group_column='Diluent (CO2)', bins=np.linspace(0, 1, 25)):
    data['groups'] = pd.cut(data[group_column], bins=bins)
    columns = ['Dilution', 'Velocity', 'Error']
    mean_vals = data.groupby(['groups']).mean()
    std_vals = data.groupby(['groups']).std()

    mean_vals.columns = columns
    std_vals.columns = columns
    data.columns = columns + ['groups']

    processed = {'data': data, 'mean': mean_vals, 'std': std_vals}
    return processed

'''
read_csv:
    Take in data from the desired csv file and place the data in a Pandas
    DataFrame. Then keep only the desired columns trim the data baed on the
    input limits.
'''


def read_csv(file_name=None, trim=False, trim_limits=(500, 3000), plot=True,
             axis_limits=True, desired_columns=['Diluent', 'V1', 'R1'],
             trim_column='V'):

    try:
        # Read the data from the csv into a Pandas DataFrame
        data = pd.read_csv(file_name)
    except:
        file_name = FindFile('csv file to plot')
        # Read the data from the csv into a Pandas DataFrame
        data = pd.read_csv(file_name)

    keep_columns = []
    for i in desired_columns:
        [keep_columns.append(loc) for loc,
         idx in enumerate(data.columns) if i in idx]
    # Remove the columns that do not contain any useful information

    data = data.take(keep_columns, axis=1)
    # Separate the data into the respectiv categories for easier use later on
    base_data = data[data.columns[0::3]]
    base_data.replace(base_data[base_data.columns[0]])
    CO2_data = data[data.columns[1::3]]
    N2_data = data[data.columns[2::3]]

    # Remove all of the data that does not reside inside the limits imposed
    # by the user
    if trim is True:
        base_data = base_data[base_data['V1'] > min(trim_limits)]
        base_data = base_data[base_data['V1'] < max(trim_limits)]

        CO2_data = CO2_data[CO2_data['V1.1'] > min(trim_limits)]
        CO2_data = CO2_data[CO2_data['V1.1'] < max(trim_limits)]

        N2_data = N2_data[N2_data['V1.2'] > min(trim_limits)]
        N2_data = N2_data[N2_data['V1.2'] < max(trim_limits)]
    # Plot the CO2 and N2 data. The non dilution data is not plotted since
    # it creates a large cluster of data on the LHS of the plot that is not
    # really useful
    if plot is True:
        fig = plt.figure()
        p1, = plt.plot(CO2_data['Diluent (CO2)'], CO2_data['V1.1'], 'k^')
        p2, = plt.plot(N2_data['Diluent (N2)'], N2_data['V1.2'], 'o',
                       markerfacecolor='None', markeredgecolor='blue')

        if axis_limits is True:
            plt.xlim([0.0, 0.5])
            plt.ylim(trim_limits)
        plt.xlabel(r'$Y_{diluent}$', fontsize=14)
        plt.ylabel('Detonation Velocity (m/s)', fontsize=14)
        fig.legend(handles=[p1, p2], labels=[r'$CO_{2}$', r'$N_{2}$'], loc=0,
                   numpoints=1)
        plt.show()

    # How many test values reside within the trim limits on velocity
    successful_tests = {'None': sum(((trim_limits[0] < data['V1']) &
                                     (data['V1'] < trim_limits[1]))),
                        'CO2': sum(((trim_limits[0] < data['V1.1']) &
                                    (data['V1.1'] < trim_limits[1]))),
                        'N2': sum(((trim_limits[0] < data['V1.2']) &
                                   (data['V1.2'] < trim_limits[1])))}
    print(successful_tests)
    return base_data, CO2_data, N2_data

def error_analysis():
    return None

if __name__ == '__main__':
    trim_limits = (1200, 2100)
    # Create the bins to sort the dilution species into
    bins = np.linspace(0, 0.5, 50)
    file_name = '/home/aero-10/Dropbox/PDE Codes/Compiled test data.csv'
#    file_name = 'C:/Users/beande.ONID/Dropbox/PDE Codes/Compiled test data.csv'
    # Get the raw data from the csv_file
    base_data, CO2_data, N2_data = read_csv(file_name, axis_limits=False,
                                            trim=True,
                                            trim_limits=trim_limits,
                                            plot=False)
    # Output the data to a nested dictionary that can be accessed using the
    # following syntax
    # prcessed_data[<dilution species ('CO2', 'N2', or 'No_dil')>][<what data
    # you want ('data', 'mean', 'std')>]
    processed_data = {'CO2': grouping(CO2_data, group_column='Diluent (CO2)',
                                      bins=bins),
                      'N2': grouping(N2_data, group_column='Diluent (N2)',
                                     bins=bins),
                      'No_dil': [base_data['V1'].mean(), base_data['V1'].std()]
                      }

    # Get the fit data in an array
    fig_path = '/home/aero-10/Dropbox/Apps/ShareLaTeX/Dilution Manuscript/Figures/'
#    fig_path = 'C:/Users/beande.ONID/Dropbox/Apps/ShareLaTeX/Dilution Manuscript/Figures/'
    correction = 0.95
    CO2_x = np.array(processed_data['CO2']['mean']['Dilution'])
    CO2_y = np.array(processed_data['CO2']['mean']['Velocity'])
    N2_x = np.array(processed_data['N2']['mean']['Dilution'])*(1/correction)
    N2_y = np.array(processed_data['N2']['mean']['Velocity'])
    # Remove nan values
    CO2_x = CO2_x[~np.isnan(CO2_x)]
    CO2_y = CO2_y[~np.isnan(CO2_y)]
    chi_CO2 = CO2_x
    N2_x = N2_x[~np.isnan(N2_x)]
    N2_y = N2_y[~np.isnan(N2_y)]
    chi_N2 = N2_x/(28*(N2_x/28+(1-N2_x)/44))
#    # Create the curve fit
#    CO2_fit = sci_stats.linregress(CO2_x, CO2_y)
#    N2_fit = sci_stats.linregress(N2_x, N2_y)
#    # Plot the data with curve fits
#    fig = plt.figure(1)
#    plt.plot(CO2_x, CO2_y, 'k^')
#    plt.plot(N2_x, N2_y, 'ko', markerfacecolor='None')
#    plt.plot(CO2_x, CO2_x*CO2_fit.slope+CO2_fit.intercept, '--k')
#    plt.plot(N2_x, N2_x*N2_fit.slope+N2_fit.intercept, '--k')
#    plt.xlim([0.0, 0.5])
#    plt.xlabel(r'$Y_{N_{2}}$ equivalent', fontsize=14)
#    plt.ylabel('Detonation Velocity (m/s)', fontsize=14)
#    plt.legend([r'$CO_{2}$', r'$N_{2}$'], loc=0, numpoints=1)
#    plt.show()
#    plt.savefig(fig_path+'avg_plot')
#    # Plot the deviation from the mean measured value
#    CO2_dev = abs(processed_data['No_dil'][0]-CO2_y)
#    N2_dev = abs(processed_data['No_dil'][0]-N2_y)
#    CO2dev_fit = sci_stats.linregress(CO2_x, CO2_dev)
#    N2dev_fit = sci_stats.linregress(N2_x, N2_dev)
#
#    plt.figure(2)
#    plt.plot(CO2_x, CO2_dev, 'k^')
#    plt.plot(N2_x, N2_dev, 'ko', markerfacecolor='None')
#
#    plt.xlim([0.0, 0.5])
#    plt.xlabel(r'$Y_{N_{2}}$ equivalent', fontsize=14)
#    plt.ylabel('Velocity Suppression (m/s)', fontsize=14)
#    plt.legend([r'$CO_{2}$', r'$N_{2}$'], loc=0, numpoints=1)
#    plt.show()
#    plt.savefig(fig_path+'depression_plot')
    print(processed_data['No_dil'])
#    f = open(fig_path+'avg_nondil_vel.tex', 'w')
#    f.write('{0}'.format(int(processed_data['No_dil'][0])))
#    f.close()
#    f = open(fig_path+'percent_CJ.tex', 'w')
#    f.write('{0}'.format(int((processed_data['No_dil'][0]/2188)*100)))
#    f.close()
#    f = open(fig_path+'fit_slope.tex', 'w')
#    f.write('{0}'.format(int(CO2_fit.slope/N2_fit.slope)))
#    f.close()
