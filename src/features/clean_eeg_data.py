import mne
from autoreject import AutoReject, compute_thresholds

from data.extract_data import read_xdf_eeg_data

import collections


def _autoreject_repair_epochs(epochs, reject_plot=False):
    """Rejects the bad epochs with AutoReject algorithm

    Parameter
    ----------
    epochs : Epoched, filtered eeg data
    Returns
    ----------
    epochs : Epoched data after rejection of bad epochs

    """
    # Cleaning with autoreject
    picks = mne.pick_types(epochs.info, eeg=True)  # Pick EEG channels
    ar = AutoReject(n_interpolate=[1, 4, 8],
                    n_jobs=6,
                    picks=picks,
                    thresh_func='bayesian_optimization',
                    cv=10,
                    random_state=42,
                    verbose=False)

    # Apply autoreject
    repaired_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    if reject_plot:
        reject_log.plot_epochs(epochs, scalings=dict(eeg=40e-6))

    return repaired_epochs


def _append_eog_index(epochs, ica):
    """Detects the eye blink aritifact indices and adds that information to ICA

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne
    Returns
    ----------
    ICA : ICA object with eog indices appended

    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp1',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.85]
    ica.exclude += id_eog

    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp2',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.85]
    ica.exclude += id_eog
    return ica


def _filter_eeg(raw_eeg, config):
    # Drop auxillary channels
    try:
        raw_eeg = raw_eeg.drop_channels(['ECG', 'AUX1', 'AUX2', 'AUX3'])
    except ValueError:
        pass

    # Filtering
    raw_eeg.notch_filter(60, filter_length='auto', phase='zero',
                         verbose=False)  # Line noise
    raw_eeg.filter(l_freq=0.5, h_freq=50, fir_design='firwin',
                   verbose=False)  # Band pass filter

    # Channel information
    raw_eeg.set_montage(montage="standard_1020", set_dig=True, verbose=False)
    ch_info = {
        'Fp1': 'eeg',
        'F7': 'eeg',
        'F8': 'eeg',
        'T4': 'eeg',
        'T6': 'eeg',
        'T5': 'eeg',
        'T3': 'eeg',
        'Fp2': 'eeg',
        'O1': 'eeg',
        'P3': 'eeg',
        'Pz': 'eeg',
        'F3': 'eeg',
        'Fz': 'eeg',
        'F4': 'eeg',
        'C4': 'eeg',
        'P4': 'eeg',
        'POz': 'eeg',
        'C3': 'eeg',
        'Cz': 'eeg',
        'O2': 'eeg'
    }
    raw_eeg.set_channel_types(ch_info)
    return raw_eeg


def _clean_with_ica(raw_eeg, config, show_ica=False, apply_on_epoch=False):
    """Clean epochs with ICA.

    Parameter
    ----------
    epochs : Filtered raw EEG

    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs

    """
    try:
        raw_eeg = raw_eeg.drop_channels(['ECG', 'AUX1', 'AUX2', 'AUX3'])
    except ValueError:
        pass
    raw_eeg.set_montage(montage="standard_1020", set_dig=True, verbose=False)
    picks = mne.pick_types(raw_eeg.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False,
                           exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="picard",
                                verbose=False)

    # Epoch the data and get the global rejection threshold
    epoch_length = config['epoch_length']
    events = mne.make_fixed_length_events(raw_eeg, duration=epoch_length)
    epochs = mne.Epochs(raw_eeg,
                        events,
                        picks=['eeg'],
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=(0, 0),
                        verbose=False)

    # Get the rejection threshold using autoreject
    reject_threshold = compute_thresholds(epochs.load_data(),
                                          method='bayesian_optimization',
                                          random_state=42,
                                          n_jobs=10)
    ica.fit(epochs, picks=picks, reject=reject_threshold, tstep=epoch_length)

    # Extra caution to detect the eye blinks
    ica = _append_eog_index(epochs, ica)  # Append the eog index to ICA

    # mne pipeline to detect artifacts
    ica.detect_artifacts(epochs, eog_criterion=range(2))

    if show_ica:
        ica.plot_components(inst=epochs)

    if apply_on_epoch:
        cleaned_eeg = ica.apply(epochs)  # Apply the ICA on Epoch EEG
    else:
        cleaned_eeg = ica.apply(raw_eeg)  # Apply the ICA on raw EEG

    return cleaned_eeg, ica


def clean_eeg_data(config):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.

    Parameter
    ----------
    config : The configuration file

    Returns
    ----------
    cleaned_eeg_dataset : dataset of all the subjects with different conditions

    """
    cleaned_eeg_dataset = {}

    for subject in config['subjects']:
        data = collections.defaultdict(dict)
        for session in config['sessions']:
            # Read only the eeg data
            raw_eeg, time_stamps = read_xdf_eeg_data(config, subject, session)

            # Clean the EEG epochs
            cleaned_eeg, ica = _clean_with_ica(raw_eeg, config, show_ica=False)

            # Populate the dictionary
            data[session]['cleaned_eeg'] = cleaned_eeg
            data[session]['ica'] = ica
            data[session]['time_stamps'] = time_stamps

        # Append the data
        cleaned_eeg_dataset['sub-OFS_' + subject] = data
    return cleaned_eeg_dataset
