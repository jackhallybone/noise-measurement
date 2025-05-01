import cmsisdsp
import numpy as np

from sound_level import define_frequency_weighting_filter


def sos_to_cmsis_biquad(sos):
    """'Convert' an sos filter definition into a format for the CMSIS biquad filter.

    "SciPy and CMSIS-DSP do not save the coefficients in the same way in memory. In CMSIS-DSP, the a0 coefficient is assumed to be 1 and is not saved in memory. Also, in CMSIS-DSP the “a” coefficients are negative compared to the SciPy conventions." - https://developer.arm.com/documentation/102463/0100/How-to-implement-biquads-to-filter-an-Electrocardiography-signal

    Reference implementation: https://github.com/ARM-software/CMSIS-DSP/blob/main/PythonWrapper/examples/example.py#L58

    Args:
        sos (ndarray): Array of second-order filter coefficients.

    Returns:
        num_stages (int): The number of filter stages
        coeffs (ndarray): Array of filter coefficiants for the CMSIS biquad filter
        state (ndarray): Array of filter state values
    """

    # https://arm-software.github.io/CMSIS-DSP/main/structarm__biquad__cascade__df2T__instance__f32.html
    num_stages = sos.shape[0]  # overall order is 2*num_stages
    num_coeffs = sos.shape[1] - 1  # == 5*num_stages
    state = np.zeros(2 * num_stages)  # array of length 2*num_stages

    # In CMSIS, a0 assumed to be 1 and not included, and the 'a' coeffs are negated
    coeffs = np.reshape(np.hstack((sos[:, :3], -sos[:, 4:])), num_stages * num_coeffs)

    return num_stages, coeffs, state


def frequency_weight(fs, weighting, in_data):
    """Frequency weight an input signal.

    The frequency weighting filters are defined in python and 'converted' to CMSIS format.

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

    # Define the cmsis filter based on the sos filter
    sos = define_frequency_weighting_filter(fs, weighting)
    num_stages, coeffs, state = sos_to_cmsis_biquad(sos)

    # If the embedded system requires, the filter can be implemented one stage a time by setting num_stages=1 and looping over the coeffs in groups of 5.

    # Init the cmsis biquad filter with the filter settings
    biquad_f32 = cmsisdsp.arm_biquad_cascade_df2T_instance_f32()
    cmsisdsp.arm_biquad_cascade_df2T_init_f32(biquad_f32, num_stages, coeffs, state)

    out_data = cmsisdsp.arm_biquad_cascade_df2T_f32(biquad_f32, in_data)

    return out_data


fs = 48000

test_signal_unit_impulse = np.array([1] + [0] * (fs - 1))

cmsis_out = frequency_weight(48000, "A", test_signal_unit_impulse)

from sound_level import frequency_weight

scipy_out = frequency_weight(48000, "A", test_signal_unit_impulse)

import matplotlib.pyplot as plt

plt.plot(cmsis_out)
plt.plot(scipy_out, ":")
plt.show()
