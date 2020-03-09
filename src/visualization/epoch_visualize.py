import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_multitaper

from .utils import plot_settings


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
def topo_visualize(epochs, config):
    # Default plot settings
    plot_settings()

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
