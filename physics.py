import numpy as np

def get_peak_current(current: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Peak current (Ip)
    Arguments:
        current (np.ndarray): list of current values
        potential (np.ndarray): list of potential values
        start_idx (int): starting index for peak search
        end_idx (int): ending index for peak search
    Returns (float):
        The maximum current value of the peak.

    This is the amount of electroactive species being oxidized or reduced. This is the analytical signal and the strongest signal for ML.
    '''
    return float(np.max(current[start_idx:end_idx]))

def get_peak_potential(potential: np.ndarray, current: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Peak potential (Ep)
    Arguments:
        potential (np.ndarray): list of potential values
        current (np.ndarray): list of current values
        start_idx (int): starting index for peak search
        end_idx (int): ending index for peak search
    Returns (float):
        The potential (voltage) at which the current reaches its maximum.

    This is the redox potential of pyocyanin under your experimental conditions.
    This confirms that the signal is actually from pyocyanin. If this value is somehow stable accross concentrations means that the sensor is a good one.
    Small shifts in this value can indicate changes in the chemical environment: absorption, kinetics or interference.
    Usually weak for concentration prediction, but useful for sensor and signal validation.
    '''
    slice_idx = int(np.argmax(current[start_idx:end_idx]))
    global_idx = start_idx + slice_idx

    return float(potential[global_idx])

def get_peak_width_HM(potential: np.ndarray, current: np.ndarray, peak_current: float, threshold_ratio: float=0.5) -> float:
    '''
    Peak width at half maximum (FWHM)
    Arguments:
        potential (np.ndarray): list of potential values
        current (np.ndarray): list of current values
        peak_current (float): the maximum current value of the peak
        threshold_ratio (float): ratio of peak current to define width (default is 0.5 for FWHM - half maximum)
    Returns (float):
        The voltage difference between the two points where the current is half of Ip.

    This indicates how sharp or broad the peak is.
    Peak broadening ofte increases at low concentration, noisy regimes and surface effects.
    '''
    threshold = peak_current * threshold_ratio
    indices_above_threshold = np.where(current >= threshold)[0]
    
    if len(indices_above_threshold) < 2:
        return 0.0  # Peak is too narrow or not well-defined
    
    left_idx = indices_above_threshold[0]
    right_idx = indices_above_threshold[-1] + 1
    
    peak_width = potential[right_idx] - potential[left_idx]
    return float(peak_width)

def get_peak_area(potential: np.ndarray, current: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Peak area (AUC)
    Arguments:
        potential (np.ndarray): list of potential values
        current (np.ndarray): list of current values
        start_idx (int): starting index for peak area calculation
        end_idx (int): ending index for peak area calculation
    Returns (float):
        The area under the peak curve (numerical integration).
    
    This represents the total charge transferred during the redox event.
    It can be more robust than peak current alone, especially in noisy data.
    Corelates better than peak current with concentration in some cases.
    '''
    peak_area = np.trapz(current[start_idx:end_idx], potential[start_idx:end_idx])
    return float(peak_area)

def get_left_slope(potential: np.ndarray, current: np.ndarray, start_idx: int=18, end_idx: int=70) -> float:
    '''
    Left slope (pre-peak slope)
    Arguments:
        current (np.ndarray): list of current values
        potential (np.ndarray): list of potential values
        peak_idx (int): index of the peak current
        points (int): number of points to consider for slope calculation
    Returns (float):
        The slope of the left side of the peak.

    The slope of current increase before the peak. This can indicate how fast the electrochemical reaction turns on.
    This can be sensitive to diffusion, adsorption and surface kinetics.
    '''
    # defining the local window
    alpha = 0.1
    current_threshold = alpha * get_peak_current(current, start_idx, end_idx)
    pass

def get_right_slope(potential: np.ndarray, current: np.ndarray, peak_idx: int, points: int=5) -> float:
    '''
    Right slope (post-peak slope)
    Arguments:
        current (np.ndarray): list of current values
        potential (np.ndarray): list of potential values
        peak_idx (int): index of the peak current
        points (int): number of points to consider for slope calculation
    Returns (float):
        The slope of the right side of the peak.
    The slope of current decrease after the peak. This can indicate how fast the electrochemical reaction turns off.
    This can be sensitive to diffusion, adsorption and surface kinetics.
    '''
    pass