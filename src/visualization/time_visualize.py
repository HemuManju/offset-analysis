import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd


def describe_helper(series):
    splits = str(series.describe()[['mean', 'std', 'max']]).split()
    keys, values = "", ""
    for i in range(0, len(splits) - 2, 2):
        keys += "{:8}\n".format(splits[i] + ':')
        values += "{:>8}\n".format(splits[i + 1])

    print(values)
    return keys, values


def plot_time_delays(config):
    # Read the data
    data = dd.io.load(config['time_offset_dataset'])

    plt.style.use('clean_box')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    streams = ['eeg', 'eye', 'game']
    offset = {'eeg': [], 'eye': [], 'game': []}

    for subject in config['subjects']:
        for session in config['sessions']:
            temp = data['sub-OFS_' + subject][session]
            for i, stream in enumerate(streams):
                offset[stream].append(temp[stream]['time_offsets'] -
                                      np.mean(temp[stream]['time_offsets']))

    title = {'eeg': 'EEG Data', 'eye': 'Eye Data', 'game': 'Game Data'}
    for i, stream in enumerate(streams):
        data = np.concatenate(offset[stream], axis=0)
        ax[i].set_title(title[stream])
        ax[i].hist(data, edgecolor='black')
        ax[i].grid()
        ax[i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax[i].set_xlabel('Time offsets (s)')
    plt.autoscale(enable=True, axis='x')
    plt.tight_layout(pad=0)
    plt.savefig('time_offsets.pdf')
    plt.show()
