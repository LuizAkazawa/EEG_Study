from pyedflib import highlevel
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

path = "S001R03.edf"
signals, signal_headers, header = highlevel.read_edf(path)
fs = signal_headers[0]['sample_frequency']  # Sampling rate (Hz)
channel_names = [h['label'] for h in signal_headers]  # List of channel names

print(f"Sampling rate: {fs} Hz")
print(f"Channels: {channel_names}")
print(f"Signal shape: {signals.shape}")  # (n_channels, n_samples)

# Find indices of C3 and C4
c3_idx = channel_names.index('C3..')
c4_idx = channel_names.index('C4..')

c3_signal = signals[c3_idx, :]
c4_signal = signals[c4_idx, :]

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Filter C3 and C4
c3_filtered = bandpass_filter(c3_signal, 0.5, 40, fs)
c4_filtered = bandpass_filter(c4_signal, 0.5, 40, fs)

# Optional: Notch filter (50 Hz)
def notch_filter(data, notch_freq, fs, quality=30):
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = butter(2, [freq - 0.5/quality, freq + 0.5/quality], btype='bandstop')
    return filtfilt(b, a, data)

c3_filtered = notch_filter(c3_filtered, 50, fs)
c4_filtered = notch_filter(c4_filtered, 50, fs)

time = np.arange(0, len(c3_signal)) / fs

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
#plt.plot(time, c3_signal, label='Raw C3')
plt.plot(time, c3_filtered, label='Filtered C3', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend()

'''plt.subplot(2, 1, 2)
plt.plot(time, c4_signal, label='Raw C4')
plt.plot(time, c4_filtered, label='Filtered C4', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend()'''
plt.tight_layout()
plt.show()