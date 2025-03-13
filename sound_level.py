"""Sound level calculations according to IEC 61672-1 (2002)

Based on the publicly available reprint IS 15575 (Part 1 ) : 2005
https://law.resource.org/pub/in/bis/S04/is.15575.1.2005.pdf

References to sections, figures, etc in the function docs refer to the IS document.
"""

import numpy as np
from scipy import signal


def define_time_weighting_filter(fs, weighting):
    """Define the time-weighting filter in sos format.

    Figure 1 describes the filter as "low-pass filter with one real pole at -1/tau".

    Where tau is defined in section 5.7.1 as 0.125s for F- (Fast-) and 1s for S- (Slow-) weighting.

    Args:
        fs (int): Sampling rate of the system in Hz.
        weighting (str): Time weighting to apply ("F" for Fast- or "S" for Slow-weighting).

    Returns:
        sos (ndarray): Array of second-order filter coefficients.
    """

    tau = 0.125 if weighting == "F" else 1

    # https://github.com/python-acoustics/python-acoustics/issues/210#issuecomment-368898437
    b = np.poly1d([1])
    a = np.poly1d([tau, 1])

    b, a = signal.bilinear(b, a, fs=fs)
    sos = signal.tf2sos(b, a)

    return sos


def time_weight(fs, weighting, in_data):
    """Apply time weighting.

    Figure 1 describes the process of time weighting a signal:
        input -> [square] -> [time-weighting filter] -> [sqrt] -> [log] -> output

    Where "time-weighting filter" is realised in `time_weighting_filter()`.

    Args:
        fs (int): Sampling rate of the system in Hz.
        weighting (str): Time weighting to apply ("F" for Fast- or "S" for Slow-weighting).
        in_data (ndarray): Input signal to be time weighted.

    Returns:
        out_data (ndarray): Output time weighted signal.
    """

    sos = define_time_weighting_filter(fs, weighting)

    out_data = np.square(in_data)
    out_data = signal.sosfilt(sos, out_data)
    out_data = np.sqrt(out_data)
    out_data = 20 * np.log10(out_data)

    return out_data
