"""Sound level calculation validation according to IEC 61672-1 (2002)

Based on the publicly available reprint IS 15575 (Part 1 ) : 2005
https://law.resource.org/pub/in/bis/S04/is.15575.1.2005.pdf

References to sections, figures, etc in the function docs refer to the IS document.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Polygon

import sound_level

# GLOBAL SETTINGS
fs = 48000
plot = False


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

        ax.plot(weighted_signal, color=color)

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
            )
        )

        ax.set_ylim(-40, 0)

        fig.suptitle(f"{weighting}- Time Weighting Decay Rate")
        fig.tight_layout()

        plt.show()


def test_time_weighting_steady_state():
    """Validate time weighting steady state response.

    Section 5.7.3 defines the maximum deviation between F- and S-weighted signals.
        - Shall not exceed +/-0.3dB

    TODO: for A-weighted signals and at reference level
    """

    test_signal_1kHz = np.sin(2 * np.pi * 1000 * (np.arange(10 * fs) / fs))  # 10 seconds

    print(test_signal_1kHz.shape)

    F_weighted_signal = sound_level.time_weight(fs, "F", test_signal_1kHz)
    S_weighted_signal = sound_level.time_weight(fs, "S", test_signal_1kHz)

    # Assert against the final 1 second at steady state
    assert np.max(np.abs(F_weighted_signal[-fs:] - S_weighted_signal[-fs:])) < 0.3

    if plot:
        fig, ax = plt.subplots()

        ax.plot(F_weighted_signal, color="tab:blue")
        ax.plot(S_weighted_signal, color="tab:orange")

        ax.set_ylim(-20, 0)

        fig.suptitle("Time Weighting Steady State")
        fig.tight_layout()

        plt.show()
