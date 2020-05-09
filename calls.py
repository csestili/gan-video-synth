"""calls.py: Example calls for GanVideoSynth.generate_in_tempo().

Usage:
1. Start up the GAN (only need to do this once):
    [in Python interpreter]
    >>> from example import *
    >>> import numpy as np
    >>> gvs = GanVideoSynth()

2. Copy and paste all the parameters below into the interpreter. Tweak to your liking.
"""

# Tempo of the song in beats per minute.
bpm = 144

# Number of beats to generate an animation for.
num_beats = 4

# ImageNet class IDs to include in the y vector.
classes = [835]

# Magnitude to scale the y vector to. If this goes higher than 7-8, the images get quite abstract.
y_scale = 1

# If true, then ignore the `classes` setting and generate a random y vector.
random_label = False

# If true, then quantize y vector entries to 1 if greater than 0.5, or 0 otherwise.
quantize_label = False

# The indices in the range [0, 128) to change in the z vector.
axis_sets = [
    range(0, 8),
    range(8, 16),
    range(16, 24),
    range(24, 32),
    range(32, 40),
    range(40, 48),
    range(48, 56),
    range(56, 64),
    range(64, 72),
    range(72, 80)
]

# How much to change each set of indices. Greater values make more noticeable change.
magnitudes = [
    0.15,
    0.15,
    0.5,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
]

# The periodic functions that change the z vector index sets. `ramp` is a saw wave.
# rampy
# funcs = [
#     ramp,
#     np.sin,
#     ramp,
#     np.sin,
#     ramp,
#     np.sin,
#     np.cos,
#     np.sin,
#     np.cos,
#     np.sin
# ]

# smooth
funcs = [
    np.cos,
    np.sin,
    np.cos,
    np.sin,
    np.cos,
    np.sin,
    np.cos,
    np.sin,
    np.cos,
    np.sin
]

# The periods of the preceding functions. 1 repeats every beat, 2 repeats every 2 beats, etc.
periods = [
    1,
    1,
    2,
    2,
    4,
    4,
    8,
    8,
    16,
    16
]

generate_in_tempo(
    gvs, bpm=bpm, magnitudes=magnitudes, funcs=funcs, axis_sets=axis_sets, random_label=random_label, classes=classes,
    quantize_label=quantize_label, y_scale=y_scale, num_beats=num_beats, periods=periods)