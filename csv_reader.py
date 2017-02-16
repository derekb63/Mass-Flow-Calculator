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
    grouped = data.groupby(['groups'])
    columns = ['Dilution', 'Velocity', 'Error']
    mean_vals = grouped.mean()
    std_vals = grouped.std()

    mean_vals.columns = columns
    std_vals.columns = columns
    data.columns = columns + ['groups']

    processed = {'data': data, 'mean': mean_vals, 'std': std_vals}
    return processed

'''
read_csv:
    Take in data from the desired csv file.

'''


def read_csv(trim=False, trim_limits=(500, 3000), plot=True, axis_limits=True):

    file_name = FindFile('csv file to plot')

    # Read the data from the csv into a Pandas DataFrame
    data = pd.read_csv(file_name)

    # Remove the columns that do not contain any useful information
    drop_cols = ['Test Number', 'Test Number.1', 'Test Number.2',
                 'Unnamed: 9', 'Unnamed: 20', 'Unnamed: 19']

    data = data.drop(drop_cols, axis=1)

    # Separate the data into the respectiv categories for easier use later on
    base_data = data[['Diluent (None)', 'V1', 'R1']]
    base_data.replace(base_data['Diluent (None)'])
    CO2_data = data[['Diluent (CO2)', 'V1.1', 'R1.1']]
    N2_data = data[['Diluent (N2)', 'V1.2', 'R1.2']]

    # Remove all of the data that does not reside inside the limits imposed
    # by the user
    if trim is True:
        base_data = base_data[base_data['V1'] > trim_limits[0]]
        base_data = base_data[base_data['V1'] < trim_limits[1]]

        CO2_data = CO2_data[CO2_data['V1.1'] > trim_limits[0]]
        CO2_data = CO2_data[CO2_data['V1.1'] < trim_limits[1]]

        N2_data = N2_data[N2_data['V1.2'] > trim_limits[0]]
        N2_data = N2_data[N2_data['V1.2'] < trim_limits[1]]

    # Plot the CO2 and N2 data. The non dilution data is not plotted since
    # it creates a large cluster of data on the LHS of the plot that is not
    # really useful
    if plot is True:
        fig = plt.figure()
        p1, = plt.plot(CO2_data['Diluent (CO2)'], CO2_data['V1.1'], 'o')
        p2, = plt.plot(N2_data['Diluent (N2)'], N2_data['V1.2'], 'x')

        if axis_limits is True:
            plt.xlim([0.0, 0.5])
            plt.ylim([500, 3000])
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
    bins = np.linspace(0.125, 0.3, 25)

    # Get the raw data from the csv_file
    base_data, CO2_data, N2_data = read_csv(axis_limits=False,
                                            trim=True,
                                            trim_limits=(1500, 2300),
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
