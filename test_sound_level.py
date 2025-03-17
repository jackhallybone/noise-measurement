"""Sound level calculation validation according to IEC 61672-1 (2002)

Based on the publicly available reprint IS 15575 (Part 1): 2005
https://law.resource.org/pub/in/bis/S04/is.15575.1.2005.pdf

References to sections, figures, etc in the function docs refer to the IS 15575 document.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Polygon
from scipy import fft

import sound_level

# GLOBAL SETTINGS
fs = 48000
plot = False


if plot:

    @pytest.fixture(scope="session", autouse=True)
    def show():
        yield  # run tests first, then show plots all at once
        plt.show()


testdata = [
    ("F", "tab:blue"),
    ("S", "tab:orange"),
]


@pytest.mark.parametrize("weighting, color", testdata)
def test_time_weighting_decay_rate(weighting, color):
    """Validate time weighting decay rate.

    Section 5.7.2 defines the decay rate requirements following sudden cessation of a 4kHz sine.
        - At least 25dB/s for F-weighting
        - Between 3.4dB/s and 5.3dB.s for S-weighting

    TODO: "this applies for any level range"
    """

    # Set up test signal
    test_signal_4kHz = np.concatenate(
        (
            np.sin(2 * np.pi * 4000 * (np.arange(5 * fs) / fs)),  # 5 seconds at 0dBFS
            np.zeros(10 * fs),  # 10 seconds of silence to view decay rate
        )
    )
    test_signal_cessation_time = 5

    # Time weight the test signal
    weighted_signal = sound_level.time_weight(fs, weighting, test_signal_4kHz)

    # Calculate the decay rate following cessation of the signal
    decay_period = 5  # seconds to measure decay rate over
    start_idx = int(round(test_signal_cessation_time * fs))
    end_idx = int(round((test_signal_cessation_time + decay_period) * fs))
    decay = weighted_signal[start_idx] - weighted_signal[end_idx]
    decay_rate = decay / decay_period

    if weighting == "F":
        assert decay_rate >= 25
    elif weighting == "S":
        assert decay_rate > 3.4 and decay_rate < 5.3

    if plot:
        fig, ax = plt.subplots()

        ax.plot(weighted_signal, color=color, label=f"{weighting}-weighted signal")

        # Plot the range of permissible decay rates
        top_dB = -3
        bottom_dB = -50
        fastest_diff_delay = 0 if weighting == "F" else (top_dB - bottom_dB) / 3.4
        slowest_diff_delay = (top_dB - bottom_dB) / (25 if weighting == "F" else 5.3)
        ax.add_patch(
            Polygon(
                [
                    ((test_signal_cessation_time) * fs, top_dB),
                    ((test_signal_cessation_time + fastest_diff_delay) * fs, bottom_dB),
                    ((test_signal_cessation_time + slowest_diff_delay) * fs, bottom_dB),
                ],
                fill=True,
                color=color,
                alpha=0.1,
                label=f"Permissible {weighting}-weighting decay range",
            )
        )

        ax.set_ylim(-40, 0)

        fig.suptitle(f"{weighting}- Time Weighting Decay Rate")
        fig.tight_layout()


def test_time_weighting_steady_state():
    """Validate time weighting steady state response.

    Section 5.7.3 defines the maximum deviation between F- and S-weighted signals.
        - Shall not exceed +/-0.3dB

    TODO: for A-weighted signals and at reference level
    """

    test_signal_1kHz = np.sin(2 * np.pi * 1000 * (np.arange(10 * fs) / fs))  # 10 seconds

    F_weighted_signal = sound_level.time_weight(fs, "F", test_signal_1kHz)
    S_weighted_signal = sound_level.time_weight(fs, "S", test_signal_1kHz)

    # Assert against the final 1 second at steady state
    assert np.max(np.abs(F_weighted_signal[-fs:] - S_weighted_signal[-fs:])) < 0.3

    if plot:
        fig, ax = plt.subplots()

        ax.plot(F_weighted_signal, color="tab:blue", label="F-weighted signal")
        ax.plot(S_weighted_signal, color="tab:orange", label="S-weighted signal")

        ax.set_ylim(-20, 0)

        fig.suptitle("Time Weighting Steady State")
        fig.tight_layout()


testdata = [
    ("Z", "tab:blue"),
    ("C", "tab:orange"),
    ("A", "tab:purple"),
    ("CtoA", "tab:brown"),
]


@pytest.mark.parametrize("weighting, color", testdata)
def test_frequency_weighting_and_tolerance(weighting, color):
    """Validate the frequency weighting is within the tolerance limits.

    The weighting targets and tolerance limits are defined in section 5.4.1 and Table 2.
    """

    weighting_and_tolerance_limits = np.array(
        [
            # Data from Table 2: Frequency Hz, A-weight, C-weight, Z-weight, C1 lower, C1 upper, C2 lower, C2 upper
            [10, -70.4, -14.3, 0.0, +3.5, -np.inf, +5.5, -np.inf],
            [12.5, -63.4, -11.2, 0.0, +3.0, -np.inf, +5.5, -np.inf],
            [16, -56.7, -8.5, 0.0, +2.5, -4.5, +5.5, -np.inf],
            [20, -50.5, -6.2, 0.0, +2.5, -2.5, +3.5, -3.5],
            [25, -44.7, -4.4, 0.0, +2.5, -2.0, +3.5, -3.5],
            [31.5, -39.4, -3.0, 0.0, +2.0, -2.0, +3.5, -3.5],
            [40, -34.6, -2.0, 0.0, +1.5, -1.5, +2.5, -2.5],
            [50, -30.2, -1.3, 0.0, +1.5, -1.5, +2.5, -2.5],
            [63, -26.2, -0.8, 0.0, +1.5, -1.5, +2.5, -2.5],
            [80, -22.5, -0.5, 0.0, +1.5, -1.5, +2.5, -2.5],
            [100, -19.1, -0.3, 0.0, +1.5, -1.5, +2.0, -2.0],
            [125, -16.1, -0.2, 0.0, +1.5, -1.5, +2.0, -2.0],
            [160, -13.4, -0.1, 0.0, +1.5, -1.5, +2.0, -2.0],
            [200, -10.9, 0.0, 0.0, +1.5, -1.5, +2.0, -2.0],
            [250, -8.6, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [315, -6.6, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [400, -4.8, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [500, -3.2, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [630, -1.9, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [800, -0.8, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [1000, 0, 0, 0, +1.1, -1.1, +1.4, -1.4],  # 1kHz
            [1250, +0.6, 0.0, 0.0, +1.4, -1.4, +1.9, -1.9],
            [1600, +1.0, -0.1, 0.0, +1.6, -1.6, +2.6, -2.6],
            [2000, +1.2, -0.2, 0.0, +1.6, -1.6, +2.6, -2.6],
            [2500, +1.3, -0.3, 0.0, +1.6, -1.6, +3.1, -3.1],
            [3150, +1.2, -0.5, 0.0, +1.6, -1.6, +3.1, -3.1],
            [4000, +1.0, -0.8, 0.0, +1.6, -1.6, +3.6, -3.6],
            [5000, +0.5, -1.3, 0.0, +2.1, -2.1, +4.1, -4.1],
            [6300, -0.1, -2.0, 0.0, +2.1, -2.6, +5.1, -5.1],
            [8000, -1.1, -3.0, 0.0, +2.1, -3.1, +5.6, -5.6],
            [10000, -2.5, -4.4, 0.0, +2.6, -3.6, +5.6, -np.inf],
            [12500, -4.3, -6.2, 0.0, +3.0, -6.0, +6.0, -np.inf],
            [16000, -6.6, -8.5, 0.0, +3.5, -17.0, +6.0, -np.inf],
            [20000, -9.3, -11.2, 0.0, +4.0, -np.inf, +6.0, -np.inf],
        ]
    )

    weighting_index = 1 if weighting == "A" or weighting == "CtoA" else 2 if weighting == "C" else 3
    class_1_upper_limits = weighting_and_tolerance_limits[:, weighting_index] + weighting_and_tolerance_limits[:, 4]
    class_1_lower_limits = weighting_and_tolerance_limits[:, weighting_index] + weighting_and_tolerance_limits[:, 5]

    test_signal_unit_impulse = np.array([1] + [0] * (fs - 1))

    if weighting == "CtoA":
        weighted_signal = sound_level.frequency_weight(fs, "C", test_signal_unit_impulse)
        weighted_signal = sound_level.frequency_weight(fs, "CtoA", weighted_signal)
    else:
        weighted_signal = sound_level.frequency_weight(fs, weighting, test_signal_unit_impulse)

    # Get the frequency response
    y = np.abs(fft.rfft(weighted_signal))
    y = 20 * np.log10(y)  # to dBFS
    x = np.linspace(0, fs / 2, num=y.shape[0])  # frequency bin numbering

    # Filter the frequency response for only the frequencies defined in Table 2
    bin_indices = [(np.abs(x - f)).argmin(axis=0) for f in weighting_and_tolerance_limits[:, 0]]
    y_defined = y[bin_indices]

    assert np.all(y_defined < class_1_upper_limits) and np.all(y_defined > class_1_lower_limits)

    if plot:
        fig, ax = plt.subplots()

        # Plot target and tolerance limits
        ax.semilogx(
            weighting_and_tolerance_limits[:, 0],
            weighting_and_tolerance_limits[:, weighting_index],
            linestyle="",
            marker="x",
            color="k",
            label=f"{weighting}-weighting target",
        )
        ax.semilogx(
            weighting_and_tolerance_limits[:, 0],
            class_1_upper_limits,
            linestyle="",
            marker=7,
            color="k",
            label=f"Class 1 {weighting}-weighting upper limit",
        )
        ax.semilogx(
            weighting_and_tolerance_limits[:, 0],
            class_1_lower_limits,
            linestyle="",
            marker=6,
            color="k",
            label=f"Class 1 {weighting}-weighting lower limits",
        )

        ax.vlines(fs / 2, ymin=-75, ymax=10, color="k", linestyle=":", label="fs/2")

        ax.semilogx(x, y, color=color, label=f"{weighting}-weighted signal")

        ax.set_xlim(9, (fs / 2) * 1.1)
        ax.set_ylim(-75, 10)
        ax.grid()
        ax.legend(loc="lower center")
        fig.suptitle(f"{weighting}- Frequency Weighting")
        fig.tight_layout()


def test_frequency_weighting_steady_state():
    """Validate the frequency weighting steady state response.

    Section 5.4.14 defines the maximum deviation between frequency weighted signals.
        - shall not exceed +/-0.4dB from the A-weighted
    """

    test_signal_1kHz = np.sin(2 * np.pi * 1000 * (np.arange(10 * fs) / fs))  # 10 seconds

    Z_weighted_signal = sound_level.frequency_weight(fs, "Z", test_signal_1kHz)
    C_weighted_signal = sound_level.frequency_weight(fs, "C", test_signal_1kHz)
    A_weighted_signal = sound_level.frequency_weight(fs, "A", test_signal_1kHz)
    CtoA_weighted_signal = sound_level.frequency_weight(fs, "CtoA", C_weighted_signal)

    # TODO: using RMS for now, but correct this
    Z_level = np.sqrt(np.mean(np.square(Z_weighted_signal)))
    C_level = np.sqrt(np.mean(np.square(C_weighted_signal)))
    A_level = np.sqrt(np.mean(np.square(A_weighted_signal)))
    CtoA_level = np.sqrt(np.mean(np.square(CtoA_weighted_signal)))

    assert np.all(np.abs([Z_level, C_level, CtoA_level] - A_level) <= 0.4)

    if plot:
        fig, ax = plt.subplots()

        ax.plot(Z_weighted_signal, color="tab:blue", label=f"Z-weighted signal (level={Z_level:.4f})")
        ax.plot(C_weighted_signal, color="tab:orange", label=f"C-weighted signal (level={C_level:.4f})")
        ax.plot(A_weighted_signal, color="tab:purple", label=f"A-weighted signal (level={A_level:.4f})")
        ax.plot(
            CtoA_weighted_signal,
            color="tab:brown",
            linestyle="--",
            label=f"CtoA-weighted signal (level={CtoA_level:.4f})",
        )

        ax.set_xlim(0, fs / 10)
        ax.legend(loc="lower center")
        fig.suptitle("Frequency Weighting Steady State")
        fig.tight_layout()
