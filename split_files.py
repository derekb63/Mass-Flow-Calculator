'''
The goal if this code is to open the desired TDMS files ad split the data
into data for individual tests and then save the data to new individual files
'''

import re, os, glob
import numpy as np
import nptdms


def split_file(file):
    if isinstance(file, nptdms.tdms.TdmsFile):
        return [file.group_channels(g) for g in file.groups()]
    elif isinstance(file, list):
        return [split_file(f) for f in file]
    else:
        raise TypeError('The input was not a list of TdmsFiles or a single Tdms file')
        return None


def save_files(tdms_object, directory, filename):
    # Generate the next available filename
    i = 0
    while os.path.exists(os.path.join(directory, filename)):
        i += 1
        filename = filename.split('.')
        filename.insert(-1, '_' + str(i))
        filename.insert(-1, '.')
        filename = ''.join(filename)
    i = 0


    # save the data to the file
    with nptdms.TdmsWriter(os.path.join(directory, filename), 'w') as tdms_writer:
        tdms_writer.write_segment(tdms_object)
    return None

def get_files(filepath):
    cur_dur = os.getcwd()
    os.chdir(filepath)
    filenames = glob.glob('**/*.tdms', recursive=True)
    os.chdir(cur_dur)
    files = [nptdms.TdmsFile(os.path.join(filepath, file)) for file in filenames]
    return files, filenames

def get_filenames(filepath):
    cur_dur = os.getcwd()
    os.chdir(filepath)
    filenames = glob.glob('**/*.tdms', recursive=True)
    os.chdir(cur_dur)
    return filenames

def get_file(filepath, filename):
    return nptdms.TdmsFile(os.path.join(filepath, filename))

if __name__ == "__main__":
    filepath = 'C:\\Users\\derek\\Desktop\\OxyPDE_08132019'
    filenames = get_filenames(filepath)
    for f in filenames:
        data_file = get_file(filepath, f)
        object_list = split_file(data_file)
        for o in object_list:
            save_files(o, filepath, f)

    # try:
    #     type(data_files)
    # except NameError:
    #     data_files, filenames = get_files(filepath)
    # object_list = split_file(data_files)
    # for idx, file in enumerate(object_list):
    #     save_files(file, filepath, filenames[idx])
