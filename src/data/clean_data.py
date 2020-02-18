from pathlib import Path

import mne
import deepdish as dd
from autoreject import AutoReject, get_rejection_threshold
import collections

# from .eeg_utils import *


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


def clean_with_ica(epochs, show_ica=False):
    """Clean epochs with ICA.

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs

    """
    picks = mne.pick_types(epochs.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False,
                           exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="picard",
                                verbose=False)

    # Get the rejection threshold using autoreject
    reject_threshold = get_rejection_threshold(epochs)
    ica.fit(epochs, picks=picks, reject=reject_threshold)

    ica = append_eog_index(epochs, ica)  # Append the eog index to ICA

    # mne pipeline to detect artifacts
    ica.detect_artifacts(epochs, eog_criterion=range(2))
    if show_ica:
        ica.plot_components(inst=epochs)

    cleaned_epochs = ica.apply(epochs)  # Apply the ICA
    return cleaned_epochs, ica


def clean_eeg_data(subjects, sessions, config):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    sessions  : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    clean_eeg_dataset : dataset of all the subjects with different conditions

    """
    clean_eeg_dataset = {}
    read_path = Path(__file__).parents[2] / config['raw_eeg_dataset']
    raw_eeg = dd.io.load(str(read_path))  # load the raw eeg

    # Parameters

    for subject in subjects:
        data = collections.defaultdict(dict)
        for session in sessions:
            print(subject, session)
            epochs = raw_eeg['sub_OFS_' + subject][session]['eeg']

            repaired_eeg_epoch = autoreject_repair_epochs(epochs.load_data())
            ica_epochs, ica = clean_with_ica(repaired_eeg_epoch,
                                             show_ica=False)

            data[session]['clean_eeg'] = ica_epochs
            data[session]['ica'] = ica
        clean_eeg_dataset[subject] = data
    return clean_eeg_dataset
