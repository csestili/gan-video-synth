### data exploration
from example import *
audio_fname = "audio/<fname>"

def plot_side_mid_ratio():
    fps = 30
    # data shape: [num_samples, num_channels]
    rate, data = scipy.io.wavfile.read(audio_fname)

    # Length of window. We'll use a window that covers the duration of 2 video frames, because windows overlap by half.
    # This *should* make the number of STFT frames equal to the number of video frames.
    # It's apparently off by one, not sure why.
    nperseg = 2 * (rate // fps)
    # zxx is a complex array with shape: [num_bins, num_channels, num_windows]
    _, _, zxx = scipy.signal.stft(data, axis=0, nperseg=nperseg)
    # Discard phase. Shape is still [num_bins, num_channels, num_windows]
    spectrogram = np.abs(zxx)

    # Get mid/side energy in each bin and window.
    if spectrogram.shape[1] != 2:
        raise NotImplementedError(
            "Mid-side processing is only implemented for stereo signals, but was given a non-stereo signal.")
    # Shape of each is [num_bins, num_windows]
    mid = 0.5 * np.sum(spectrogram, axis=1)
    side = 0.5 * (spectrogram[:, 1, :] - spectrogram[:, 0, :])
    # Get the ratio, summing across bins.
    # When this quantity is close to 0, the signal is nearly mono.
    # The greater this quantity, the more stereo the signal is.
    side_mid_ratio = np.sum(np.abs(side), axis=0) / (np.sum(mid, axis=0) + 0.00001)
    # Rescale to [0, 1]
    normalized_smr = side_mid_ratio - np.min(side_mid_ratio)
    normalized_smr = normalized_smr / np.max(normalized_smr)

    # mid-side plots
    import matplotlib.pyplot as plt

    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    num_frames = mid.shape[1]
    seconds = np.arange(num_frames) / fps
    plt.scatter(seconds, normalized_smr)
    plt.title("Side/Mid Ratio")
    plt.xlabel("Seconds")
    plt.ylabel("Side/Mid Ratio")
    # Save figure using 72 dots per inch
    plt.savefig("side_mid.png", dpi=72)