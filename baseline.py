import numpy as np
from physics import get_peak_values

def get_baseline_mean_current(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Calculate the mean baseline current excluding the peak region. This is the average background current.
    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak calculation
        end_idx (int): ending index for peak calculation
    Returns:
        float: The baseline mean current.
    '''
    Ibaseline = np.concatenate((I[:start_idx], I[end_idx:]))
    return np.mean(Ibaseline)

def get_prepeak_baseline_mean_current(E: np.ndarray, I: np.ndarray, start_idx: int=18) -> float:
    '''
    Calculate the mean baseline current from the pre-peak part of the signal. This is the average background current.
    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak calculation
    Returns:
        float: The baseline mean current.
    '''
    Ibaseline = I[:start_idx]
    return np.mean(Ibaseline)

def get_postpeak_baseline_mean_current(E: np.ndarray, I: np.ndarray, end_idx: int=70) -> float:
    '''
    Calculate the mean baseline current from the post-peak part of the signal. This is the average background current.
    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        end_idx (int): ending index for peak calculation
    Returns:
        float: The baseline mean current.
    '''
    Ibaseline = I[end_idx:]
    return np.mean(Ibaseline)

def get_baseline_slope(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Calculate the slope of the baseline current excluding the peak region.
    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak calculation
        end_idx (int): ending index for peak calculation
    Returns:
        float: The baseline slope.
    '''
    Ibaseline = np.concatenate((I[:start_idx], I[end_idx:]))
    Ebaseline = np.concatenate((E[:start_idx], E[end_idx:]))
    return np.polyfit(Ebaseline, Ibaseline, 1)[0]

def get_prepeak_baseline_slope(E: np.ndarray, I: np.ndarray, start_idx: int=18) -> float:
    '''
    Calculate the slope of the baseline current before the peak.
    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak calculation
        end_idx (int): ending index for peak calculation
    Returns:
        float: The pre-peak baseline slope.
    '''
    E_left = np.array(E[:start_idx], dtype=float)
    I_left = np.array(I[:start_idx], dtype=float)
    return np.polyfit(E_left, I_left, 1)[0]

def get_postpeak_baseline_slope(E: np.ndarray, I: np.ndarray, end_idx: int=70) -> float:
    '''
    Calculate the slope of the baseline current after the peak.
    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        end_idx (int): ending index for peak calculation
    Returns:
        float: The post-peak baseline slope.
    '''
    E_right = np.array(E[end_idx:], dtype=float)
    I_right = np.array(I[end_idx:], dtype=float)
    return np.polyfit(E_right, I_right, 1)[0]

def get_baseline_RMS_noise(I: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Calculate the RMS noise of the baseline current excluding the peak region.
    Arguments:
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak calculation
        end_idx (int): ending index for peak calculation
    Returns:
        float: The baseline RMS noise.
    '''
    Ibaseline = np.concatenate((I[:start_idx], I[end_idx:]))
    Inoise = Ibaseline - np.mean(Ibaseline)
    baselineRMS = np.sqrt(np.mean(Inoise**2))
    return baselineRMS

def get_SNR(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Signal to noise ratio (SNR).
    Essential for low concentrations. Noise is estimated from a region without active signal.
    Both early potrentials and late potentials will be used for noise estimation. Then RMS is computed for the signal.

    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak calculation
        end_idx (int): ending index for peak calculation
    Returns:
        float: The signal to noise ratio.
    '''
    Ip, _, _ = get_peak_values(E, I)

    Ibaseline = np.concatenate((I[:start_idx], I[end_idx:]))
    Inoise = Ibaseline - np.mean(Ibaseline)
    baselineRMS = np.sqrt(np.mean(Inoise**2))
    SNR = Ip / baselineRMS
    
    return float(SNR)

