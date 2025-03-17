"""Sound level calculations according to IEC 61672-1 (2002)

Based on the publicly available reprint IS 15575 (Part 1): 2005
https://law.resource.org/pub/in/bis/S04/is.15575.1.2005.pdf

References to sections, figures, etc in the function docs refer to the IS 15575 document.
"""

import numpy as np
from scipy import signal


def define_time_weighting_filter(fs, weighting):
    """Define a time weighting filter in sos format.

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
    """Time weight an input signal.

    Figure 1 describes the process of time weighting a signal:
        input -> [square] -> [time-weighting filter] -> [sqrt] -> [log] -> output

    Where "time-weighting filter" is realised in `define_time_weighting_filter()`.

    Args:
        fs (int): Sampling rate of the system in Hz.
        weighting (str): Time weighting to apply where the options are:
            - "F" for Fast weighting, or
            - "S" for Slow weighting
        in_data (ndarray): Input signal to be time weighted.

    Returns:
        out_data (ndarray): Output time weighted signal.
    """

    sos = define_time_weighting_filter(fs, weighting)

    out_data = np.square(in_data)
    out_data = signal.sosfilt(sos, out_data)
    out_data = np.sqrt(out_data)
    out_data = 20 * np.log10(out_data)  # ie, to dBFS

    return out_data


def time_weighting_pole_frequencies():
    """Calculate the pole frequencies from their definitions.

    Sections 5.4.6 to 5.4.11 give the pole frequencies and their definitions.

    Returns:
        f1 (float): Pole frequency f1 in Hz.
        f2 (float): Pole frequency f2 in Hz.
        f3 (float): Pole frequency f3 in Hz.
        f4 (float): Pole frequency f4 in Hz.
    """

    # TODO: implement the pole frequency calculations to improve the below approximations

    # Approximate values are given in section 5.4.11
    f1 = 20.60
    f2 = 107.7
    f3 = 737.9
    f4 = 12194

    return f1, f2, f3, f4


# TODO: implement the equation based filter and normalisation process


def define_frequency_weighting_filter(fs, weighting):
    """Define a frequency weighting filter in sos format.

    Section 5.4.6 describes the realisation of the frequency weighting filters:
        - Z-weighting requires no filter; z(f)=0
        - C-weighting:
            - 2 low frequency poles at f1
            - 2 high frequency poles at f4
            - 2 zeros at 0Hz
        - A-weighting extends the C-weighting filter by adding:
            - 2 coupled first-order high-pass filters with a cut-off at fA

    Where, section 5.4.10 describes the implementaiton of the high-pass filters:
        - a pole at f2
        - a pole at f3

    Args:
        fs (int): Sampling rate of the system in Hz.
        weighting (str): Frequency weighting to apply where the options are:
            - "Z" for Z-weighting (ie, no weighting), or
            - "C" for C-weighting, or
            - "A" for A-weighting, or
            - "CtoA" for only the additional filter to take C-weighted to A-weighted

    Returns:
        sos (ndarray): Array of second-order filter coefficients.
    """

    f1, f2, f3, f4 = time_weighting_pole_frequencies()

    # https://github.com/endolith/waveform-analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py#L29

    # Z-weighting (no filter) default starting case
    z = []
    p = []
    k = 1

    if weighting == "C" or weighting == "A":
        # 2x zeros, 2x poles at f1, 2x poles at f4
        z.extend([0, 0])
        p.extend([-2 * np.pi * f1, -2 * np.pi * f1, -2 * np.pi * f4, -2 * np.pi * f4])

    if weighting == "A" or weighting == "CtoA":
        # 1x pole at f2, 1x pole at f3
        z.extend([0, 0])
        p.extend([-2 * np.pi * f2, -2 * np.pi * f3])

    # Normalise the filter to 0dB at 1kHz (fr)
    b, a = signal.zpk2tf(z, p, k)
    k /= abs(signal.freqs(b, a, [2 * np.pi * 1000])[1][0])

    # Transform into sos representation
    dz, dp, dk = signal.bilinear_zpk(z, p, k, fs)
    sos = signal.zpk2sos(dz, dp, dk)

    return sos


def frequency_weight(fs, weighting, in_data):
    """Frequency weight an input signal.

    The frequency weighting filters are realised in `define_frequency_weighting_filter().

    Args:
        fs (int): Sampling rate of the system in Hz.
        weighting (str): Frequency weighting to apply where the options are:
            - "Z" for Z-weighting (ie, no weighting), or
            - "C" for C-weighting, or
            - "A" for A-weighting, or
            - "CtoA" for only the additional filter to take C-weighted to A-weighted
        in_data (ndarray): Input signal to be time weighted.

    Returns:
        out_data (ndarray): Output time weighted signal.
    """

    sos = define_frequency_weighting_filter(fs, weighting)

    out_data = signal.sosfilt(sos, in_data)

    return out_data
