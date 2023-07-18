import numpy as np
import scipy.io.wavfile as wav
import scipy.linalg
import math
import soundfile as sf
from scipy import signal

# Load the audio file
audio, sr = sf.read('14.wav')

frame_size = 500  # Number of samples per frame

padding = np.zeros(frame_size)  # Create silent padding frames

audio_with_padding = np.concatenate((padding, audio, padding))

# Save the modified audio with added silence
sf.write('output_audio14.wav', audio_with_padding, sr)

# Load the speech segment from a WAV file
sampling_frequency, speech_segment = wav.read('14.wav')

# Downsampling factor
downsampling_factor = 44100 // 16000

# Downsample the speech segment
downsampled_segment = signal.resample_poly(speech_segment, 1, downsampling_factor)

# Update the sampling frequency
sampling_frequency = sampling_frequency // downsampling_factor

# Determine the window length for a 20 ms segment
window_length = int(0.02 * sampling_frequency)

# Apply the Hamming window to the speech segment
hamming_window = np.hamming(window_length)
windowed_speech = speech_segment[:window_length] * hamming_window

# Perform the FFT on the windowed speech segment
spectrum = np.fft.fft(windowed_speech)

# Calculate the short-term power spectrum
power_spectrum = np.abs(spectrum) ** 2

# Calculate the Bark frequency corresponding to each frequency bin
frequency_bins = np.fft.fftfreq(window_length, d=1.0 / sampling_frequency)
bark_freq = 6 * np.log(frequency_bins / (1200.0*np.pi) + ((frequency_bins / (1200.0*np.pi))**2 + 1)**0.5)

# Create the critical band masking curve
critical_band_curve = np.zeros_like(power_spectrum)
masking_curve = np.where(bark_freq < -1.3, 0,
                         np.where(bark_freq < -0.5, 10 ** (2.5 * (bark_freq + 0.5)),
                                  np.where(bark_freq < 0.5, 1,
                                           np.where(bark_freq < 2.5, 10 ** (-1 * (bark_freq - 0.5)), 0))))

# Convolve the critical band masking curve with the power spectrum
critical_band_power_spectrum = np.convolve(power_spectrum, masking_curve, mode='same')

# Define the simulated equal loudness curve
equal_loudness_curve = np.array([
    ((frequency / 600.0) ** 2 + 56.8 * 10 ** 6) ** 0.94 /
    (((frequency ** 2 + 6.3 * 10 ** 6) ** 2) * (frequency ** 2 + 0.38 * 10 ** 9))
    for frequency in frequency_bins
])

# Preemphasize the critical band power spectrum
preemphasized_spectrum = critical_band_power_spectrum * equal_loudness_curve

# Make the first and last samples equal to the values of their nearest neighbors
preemphasized_spectrum[0] = preemphasized_spectrum[1]
preemphasized_spectrum[-1] = preemphasized_spectrum[-2]

# Apply cube root amplitude compression
compressed_spectrum = np.cbrt(preemphasized_spectrum)

# Compute the cepstral coefficients
cepstral_coefficients = np.fft.ifft(compressed_spectrum)

# Keep desired number of cepstral coefficients
cepstral_coefficients = np.real(cepstral_coefficients)[:12]

# Save cepstral coefficients to a text file
np.savetxt('cepstral_coefficients1.txt', cepstral_coefficients)

# Print the cepstral coefficients
print("Cepstral coefficients:")
print(cepstral_coefficients)
