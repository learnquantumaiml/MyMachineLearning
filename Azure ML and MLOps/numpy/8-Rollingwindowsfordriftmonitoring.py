import numpy as np

# Simulate a stream of 20 incoming feature values (e.g., daily model input averages)
stream = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 20, 22, 21, 23, 25, 24, 26, 28, 27, 29], dtype=float)

window_size = 5

# Use stride tricks to build overlapping windows without copying data repeatedly
shape = (stream.size - window_size + 1, window_size)
strides = (stream.strides[0], stream.strides[0])
windows = np.lib.stride_tricks.as_strided(stream, shape=shape, strides=strides)

rolling_mean = windows.mean(axis=1)   # mean per window → "moving average"
rolling_std = windows.std(axis=1)     # std per window → could flag drift spikes

print("Rolling mean:\n", np.round(rolling_mean, 2))
print("Rolling std:\n", np.round(rolling_std, 2))
