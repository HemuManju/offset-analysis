import collections

import mne
from .mne_import_xdf import read_raw_xdf
import matplotlib.pyplot as plt
from mne.time_frequency import psd_multitaper


def get_engagement_workload(psds, nfreqs, freq_bands):
    beta_mask = (nfreqs >= freq_bands[2][0]) & (nfreqs <= freq_bands[2][1])
    beta_data = psds[:, beta_mask]
    alpha_mask = (nfreqs >= freq_bands[1][0]) & (nfreqs <= freq_bands[1][1])
    alpha_data = psds[:, alpha_mask]
    theta_mask = (nfreqs >= freq_bands[0][0]) & (nfreqs <= freq_bands[0][1])
    theta_data = psds[:, theta_mask]
    # Engagement
    engagement = beta_data.mean(axis=1) / (alpha_data.mean(axis=1) +
                                           theta_data.mean(axis=1))
    workload = (theta_data.mean(axis=1) / alpha_data.mean(axis=1))
    return engagement, workload


# Calculate the psd values
def animate(epochs, config):
    n_epochs = len(epochs.events)
    labels = [r'$\beta/(\alpha + \theta)$', r'$\theta/(\alpha)$']
    title = ['Engagement', 'Workload']
    info = mne.pick_info(epochs.info, mne.pick_types(epochs.info, eeg=True))
    freq_bands = config['freq_bands']
    psds, nfreqs = psd_multitaper(epochs,
                                  fmin=1.0,
                                  fmax=64.0,
                                  n_jobs=6,
                                  verbose=False,
                                  normalization='full')
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    for epoch in range(n_epochs):
        # Engagement index beta/(alpha + theta)
        engagement, workload = get_engagement_workload(psds[epoch, :, :],
                                                       nfreqs, freq_bands)

        for i in range(len(labels)):
            if i == 0:
                mne.viz.plot_topomap(engagement,
                                     pos=info,
                                     axes=ax[i],
                                     show=False,
                                     cmap='viridis')
                ax[i].set_ylabel(labels[i])
                ax[i].title.set_text(title[i])
            else:
                mne.viz.plot_topomap(workload,
                                     pos=info,
                                     axes=ax[i],
                                     show=False,
                                     cmap='viridis')
                ax[i].set_ylabel(labels[i])
                ax[i].title.set_text(title[i])

        plt.pause(0.01)
        for i in range(len(labels)):
            ax[i].cla()


def read_xdf_eeg_data(config, subject):
    read_path = config['raw_eeg_path'] + 'S_' + subject + '/eeg.xdf'
    raw = read_raw_xdf(read_path)
    raw = raw.drop_channels(
        ['ACC30', 'ACC31', 'ACC32', 'Packet Counter', 'TRIGGER'])
    raw.set_montage(montage="standard_1020", set_dig=True, verbose=False)
    ch_names = [
        'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6',
        'T7', 'T8', 'C3', 'Cz', 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz',
        'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1', 'O2', 'A2'
    ]
    epoch_length = config['epoch_length']
    events = mne.make_fixed_length_events(raw, duration=epoch_length)
    epochs = mne.Epochs(raw, events, picks=ch_names, verbose=False)
    animate(epochs, config)

    return raw


def create_eeg_data(config):
    plt.rcParams.update({'font.size': 22})
    eeg_data = collections.defaultdict(dict)
    for subject in config['subjects']:
        eeg_data['S_' + subject] = read_xdf_eeg_data(config, subject)
    return None
