import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def get_peak_values(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70) -> tuple[float, float, int]:
    '''
    Gets the peak signal values: the peak current, the peak potential and the index value of the corresponding signals.
    The peak represents the amount of electroactive species being oxidized or reduced.
    The potential corresponding for that peak, represents the redox potential of pyocyanin under your experimental conditions.
    This confirms that the signal is actually from pyocyanin.
    Small shifts in the potential signal can indicate changes in the chemical environment: absorption, kinetics or interference.
    
    Arguments:
        E (np.ndarray): the potential signal values
        I (np.ndarray): the current singnal values
        start_idx (int): starting index for peak search
        end_idx (int): ending index for peak search
    Returns:
        float: The peak value of the current signal.
        float: The potential at which the current reaches its maximum.
        int: The index of the maximum current value.
    '''
    peak_current = float(np.max(I[start_idx:end_idx]))
    peak_idx = int(np.argmax(I[start_idx:end_idx])) + start_idx
    peak_potential = float(E[peak_idx])
    return peak_current, peak_potential, peak_idx

def get_peak_FWHM(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70, threshold_ratio: float=0.5) -> float:
    '''
    Computes the Full width at half maximum (FWHM).
    This indicates how sharp or broad the peak is.
    Peak broadening often increases at low concentration, noisy regimes and surface effects.

    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak width calculation
        end_idx (int): ending index for peak width calculation
        threshold_ratio (float): ratio of peak current to define width (default is 0.5 for FWHM - half maximum)
    
    Returns:
        float: The voltage difference between the two points where the current is half of Ip.
    '''
    peak_current = np.max(I[start_idx:end_idx])
    threshold = peak_current * threshold_ratio
    indexes_above_threshold = np.where(I >= threshold)[0]
    
    if len(indexes_above_threshold) < 2:
        print("Warning: Not enough points above threshold to calculate FWHM.")
        return 0.0  # Peak is too narrow or not well-defined
    
    left_idx = indexes_above_threshold[0]
    right_idx = indexes_above_threshold[-1] + 1
    peak_width = E[right_idx] - E[left_idx]

    return float(peak_width)

def get_peak_area(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70, plotting: bool=False) -> float:
    '''
    Peak area under the curve.
    This represents the total charge transferred during the redox event.
    It can be more robust than peak current alone, especially in noisy data.
    Corelates better than peak current with concentration in some cases.

    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for peak area calculation
        end_idx (int): ending index for peak area calculation
        plotting (bool): whether to plot the area under the curve (default is False)
    
    Returns:
        float: The area under the peak curve (numerical integration).
    '''
    peak_area = np.trapz(I[start_idx:end_idx], E[start_idx:end_idx])
    
    if plotting:
        plt.plot(E, I, label='Current vs Potential')
        plt.fill_between(E[start_idx:end_idx], I[start_idx:end_idx], alpha=0.3, label='Peak Area')
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        plt.title('Peak Area Under the Curve')
        plt.legend()
        plt.grid()
        plt.show()
    
    return float(peak_area)

def get_left_slope(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70, plotting: bool=False) -> np.ndarray[float]:
    '''
    Left slope, i.e. the pre-peak slope.
    The slope of current increase before the peak. This can indicate how fast the electrochemical reaction turns on.
    This can be sensitive to diffusion, adsorption and surface kinetics.

    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for slope calculation
        end_idx (int): ending index for slope calculation
        plotting (bool): whether to plot the linear approximation (default is False)
    
    Returns:
        NDArray[float64]: The coefficients of the linear approximation
    '''
    # defining the local window for the left slope
    # we need the signals E_left and I_left, with values between threshold and peak
    Ip, _, peak_idx = get_peak_values(E, I, start_idx, end_idx)
    
    alpha = 0.1
    Ith = alpha * Ip
    idx_threshold = start_idx + np.where(I[start_idx:peak_idx] >= Ith)[0][0]

    # Ensure arrays are float type for polyfit
    I_left = np.array(I[idx_threshold:peak_idx], dtype=float)
    E_left = np.array(E[idx_threshold:peak_idx], dtype=float)

    if len(I_left) < 2 or len(E_left) < 2:
        print("Warning: Not enough points to calculate left slope.")
        return 0.0, 0.0
    
    # performing linear regression to approcimate the slope
    coefficients = np.polyfit(E_left, I_left, deg=1)

    # plotting for verification
    if plotting:
        y = coefficients[0] * E_left + coefficients[1]
        plt.plot(
            E, I,
            E_left, y, 'r--',
        )
        plt.grid()
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        plt.title('Left Slope Linear Approximation')
        plt.show()
    
    return coefficients

def get_right_slope(E: np.ndarray, I: np.ndarray, start_idx: int=18, end_idx: int=70, plotting: bool=False) -> np.ndarray[float]:
    '''
    Right slope, i.e. the post-peak slope.
    The slope of current decrease after the peak. This can indicate how fast the electrochemical reaction turns off.
    This can be sensitive to diffusion, adsorption and surface kinetics.

    Arguments:
        E (np.ndarray): list of potential values
        I (np.ndarray): list of current values
        start_idx (int): starting index for slope calculation
        end_idx (int): ending index for slope calculation
        plotting (bool): whether to plot the linear approximation (default is False)
    
    Returns:
        NDArray[float64]: The coefficients of the linear approximation
    '''
    # Defining the local window for the right slope
    # we need the signals E_right and I_right, with values between peak and end_idx
    Ip, _, peak_idx = get_peak_values(E, I, start_idx, end_idx)
    
    alpha = 0.1
    Ith = alpha * Ip
    idx_threshold = peak_idx + np.where(I[peak_idx:end_idx] >= Ith)[0][-1]

    # Ensure arrays are float type for polyfit
    I_right = np.array(I[peak_idx:idx_threshold], dtype=float)
    E_right = np.array(E[peak_idx:idx_threshold], dtype=float)

    if len(I_right) < 2 or len(E_right) < 2:
        print("Warning: Not enough points to calculate right slope.")
        return 0.0, 0.0
    
    # 2. performing linear regression to approcimate the slope
    coefficients = np.polyfit(E_right, I_right, deg=1)
    
    # plotting for verification
    if plotting:
        y = coefficients[0] * E_right + coefficients[1]
        plt.plot(
            E, I,
            E_right, y, 'r--',
        )
        plt.grid()
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        plt.title('Right Slope Linear Approximation')
        plt.show()

    return coefficients
