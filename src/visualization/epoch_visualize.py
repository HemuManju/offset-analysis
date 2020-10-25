import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_multitaper

from features.eeg_features import (_clean_with_ica, _filter_eeg,
                                   read_xdf_eeg_data, construct_time_kd_tree,
                                   find_nearest_time_stamp, _compute_coherence)
from features.game_features import compute_option_type

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
    fig, ax = plt.subplots(2, 1, figsize=[5, 10])
    for epoch in range(n_epochs):
        if epoch >= 23:
            # Engagement index beta/(alpha + theta)
            engagement, workload = get_engagement_workload(
                psds[epoch, :, :], nfreqs, freq_bands)

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

            plt.pause(1)
            for i in range(len(labels)):
                ax[i].cla()


def topo_map(subject, config, session):

    plt.style.use('clean')
    plt.rcParams.update({'font.size': 12})
    options = ['target_option', 'engage_option', 'caution_option']
    option_dict = {'target_option': 2, 'engage_option': 1, 'caution_option': 0}
    option_label = {
        'target_option': 'Target Search Tactic',
        'engage_option': 'Offensive Tactic',
        'caution_option': 'Cautious Tactic'
    }
    fig, ax = plt.subplots(1, len(options), figsize=[13, 5])
    freq_band = [38, 42]

    # Read subjects eye data
    eeg_data, time_stamps = read_xdf_eeg_data(config, subject, 'S002')
    time_kd_tree = construct_time_kd_tree(np.array(time_stamps, ndmin=2).T)

    # Filter and clean the EEG data
    filtered_eeg = _filter_eeg(eeg_data, config)
    cleaned_eeg, ica = _clean_with_ica(filtered_eeg, config)

    # Compute user option
    option_type, option_time = compute_option_type(config, subject, session)

    for option, time in zip(option_type, option_time):
        i = option_dict[option]
        ax[i].cla()
        # Find the nearest time stamp
        nearest_time_stamp = find_nearest_time_stamp(time_kd_tree, time)

        # Get the start and end time of the epoch
        if config['option_type'] == 'pre_option':
            start_time = nearest_time_stamp['time'] - time_stamps[0] - config[
                'cropping_length']
            end_time = start_time + config['cropping_length']

        else:
            start_time = nearest_time_stamp['time'] - time_stamps[0]
            end_time = start_time + config['cropping_length']

        # Adjust the start time
        if start_time < 0:
            start_time = 0

        temp_eeg = cleaned_eeg.copy()
        cropped_eeg = temp_eeg.crop(tmin=start_time, tmax=end_time)
        coherence = _compute_coherence(cropped_eeg, config)

        # Make the epochs
        epoch_length = config['epoch_length']
        events = mne.make_fixed_length_events(cropped_eeg,
                                              duration=epoch_length)
        epochs = mne.Epochs(cropped_eeg,
                            events,
                            picks=['eeg'],
                            tmin=0,
                            tmax=config['epoch_length'],
                            baseline=(0, 0),
                            verbose=False)
        picks = mne.pick_types(epochs.info, eeg=True)
        info = mne.pick_info(epochs.info, mne.pick_types(epochs.info,
                                                         eeg=True))
        psds, freqs = psd_multitaper(epochs,
                                     fmin=1.0,
                                     fmax=64.0,
                                     picks=picks,
                                     n_jobs=6,
                                     verbose=False,
                                     normalization='full')
        power = psds[:, :, (freqs >= freq_band[0]) &
                     (freqs < freq_band[1])].mean(axis=0)
        img, cn = mne.viz.plot_topomap(power.mean(axis=1) * 10e12,
                                       pos=info,
                                       axes=ax[i],
                                       show=False,
                                       cmap='viridis')
        ax[i].set_title('Coherence Value ={:.2f}'.format(
            coherence['fz_o1_gamma']))
        ax[i].set_xlabel(option_label[option])
    plt.savefig('coherence.pdf')
    plt.show()
    return None
