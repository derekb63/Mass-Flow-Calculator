#! usr/bin/env python

import pandas as pd
import numpy as np
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
        p1, = plt.plot(CO2_data['Diluent (CO2)'], CO2_data['V1.1'], 'o')
        p2, = plt.plot(N2_data['Diluent (N2)'], N2_data['V1.2'], 'x')

        if axis_limits is True:
            plt.xlim([0.0, 0.5])
            plt.ylim(trim_limits)
        plt.xlabel(r'$Y_{diluent}$')
        plt.ylabel('Detonation Velocity (m/s)')
        fig.legend(handles=[p1, p2], labels=[r'$CO_{2}$', r'$N_{2}$'], loc=5)
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


if __name__ == '__main__':
    # Create the bins to sort the dilution species into
    bins = np.linspace(0, 1, 150)
    file_name = 'C:/Users/beande.ONID/Dropbox/PDE Codes/Compiled test data.csv'
    # Get the raw data from the csv_file
    base_data, CO2_data, N2_data = read_csv(file_name, axis_limits=False,
                                            trim=True,
                                            trim_limits=(500, 3000),
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
    fig = plt.figure()
    plt.plot(processed_data['CO2']['mean']['Dilution'],
             processed_data['CO2']['mean']['Velocity'], 'o')

    plt.plot(processed_data['N2']['mean']['Dilution'],
             processed_data['N2']['mean']['Velocity'], 'x')
    plt.show()
