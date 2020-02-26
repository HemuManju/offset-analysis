from pathlib import Path

import mne
import deepdish as dd
from autoreject import AutoReject, compute_thresholds
import collections


def autoreject_repair_epochs(epochs, reject_plot=False):
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


def append_eog_index(epochs, ica):
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


def clean_with_ica(raw_eeg, config, show_ica=False):
    """Clean epochs with ICA.

    Parameter
    ----------
    epochs : Filtered raw EEG

    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs

    """
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
    epochs.plot(block=True)

    # Extra caution to detect the eye blinks
    ica = append_eog_index(epochs, ica)  # Append the eog index to ICA

    # mne pipeline to detect artifacts
    ica.detect_artifacts(epochs, eog_criterion=range(2))

    if show_ica:
        ica.plot_components(inst=epochs)

    cleaned_eeg = ica.apply(raw_eeg)  # Apply the ICA on raw EEG
    return cleaned_eeg, ica


def clean_eeg_data(subjects, sessions, config):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    sessions  : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    cleaned_eeg_dataset : dataset of all the subjects with different conditions

    """
    cleaned_eeg_dataset = {}
    read_path = Path(__file__).parents[2] / config['raw_offset_dataset']

    for subject in subjects:
        data = collections.defaultdict(dict)
        for session in sessions:
            print(subject, session)

            # Read only the eeg data
            group = '/sub_OFS_' + '/'.join([subject, session, 'eeg'])
            eeg_data = dd.io.load(str(read_path), group=group)
            raw_eeg = eeg_data['data']
            time_stamps = eeg_data['time_stamps']

            # Clean the EEG epochs
            cleaned_eeg, ica = clean_with_ica(raw_eeg, config, show_ica=False)
            cleaned_eeg.plot(block=True)

            # Populate the dictionary
            data[session]['cleaned_eeg'] = cleaned_eeg
            data[session]['ica'] = ica
            data[session]['time_stamps'] = time_stamps

        cleaned_eeg_dataset['sub_OFS_' + subject] = data
    return cleaned_eeg_dataset
