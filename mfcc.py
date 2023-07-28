import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.fftpack import dct

# Load the audio file
audio, sr = sf.read('14.wav')

frame_size = 500  # Number of samples per frame

padding = np.zeros(frame_size)  # Create silent padding frames

audio_with_padding = np.concatenate((padding, audio, padding))

# Save the modified audio with added silence
sf.write('output_audio14.wav', audio_with_padding, sr)

#load the audio file
audio_file = "output_audio14.wav"
sample_rate, signal = wav.read(audio_file)

#preemphasis - x'[t] = x[t] - alpha*x[t-1]...0.95 < alpha < 0.99
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

##framming and windowing
frame_size = int(0.025 * sample_rate)  # 25ms
frame_stride = int(0.01 * sample_rate)  # 10ms
num_frames = int(np.ceil(float(np.abs(len(emphasized_signal) - frame_size)) / frame_stride))

# Pad the signal to ensure that all frames have equal length
pad_signal_length = num_frames * frame_stride + frame_size
padded_signal = np.pad(emphasized_signal, (0, pad_signal_length - len(emphasized_signal)), 'constant')

# Generate the indices for the frames
indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_size, 1)).T

# Extract the frames using the indices
frames = padded_signal[indices.astype(np.int32, copy=False)]

#apply hamming window so that the signals taper smoothly at the edges
#reduces spectral leakage and improves accuracy of the feature extraction process
frames *= np.hamming(frame_size)


NFFT = 512#no. of pts. in FFT for spectral analysis
magnitudes = np.abs(np.fft.rfft(frames, NFFT))#FFT applied to frames to compute magnitudes(amplitude spectrum)
power_spectrum = (1.0 / NFFT) * np.square(magnitudes)#calculate power spectrum by squaring mag spec. and scaling them by 1/NFFT

#MFB is used - to approximate non linear human auditory perception of sound
num_filters = 40#no. of MMFB used
#conversion of Hz to Mel scale
low_freq_mel = 0
high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

# Generate equally spaced Mel points(used as ref pts. for MFBs)
mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)

#Converts mel points back to corres. freq values 
hz_points = 700 * (10**(mel_points / 2595) - 1)

# Find the indices corresponding to the points in the power spectrum
# bin_points represent the bins in the power spectrum that will be used for filtering.
bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

#initialize an array to store FB coeff.
filter_bank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))

#iterate over each filter bank and fill the filter_back array with appropriate filter coeff. 
for m in range(1, num_filters + 1):
    filter_bank[m - 1, bin_points[m - 1]:bin_points[m]] = (bin_points[m] - bin_points[m - 1]) / (hz_points[m] - hz_points[m - 1])#calculates the slope of the rising edge of the filter
    filter_bank[m - 1, bin_points[m]:bin_points[m + 1]] = (bin_points[m + 1] - bin_points[m]) / (hz_points[m + 1] - hz_points[m])#calculates the slope of the falling edge of the filter

#apply the MFB coeff. to power spectrum using matrix multiplication to obtain filtered energy values for each FB
filter_bank_energy = np.dot(power_spectrum, filter_bank.T)
#this line replaces any 0 values by epsilon to avoid numerical instability issues
filter_bank_energy = np.where(filter_bank_energy == 0, np.finfo(float).eps, filter_bank_energy)

filter_bank_energy = 20 * np.log10(filter_bank_energy)  # dB conversion by log application

num_ceps = 13  # Number of MFCC coefficients to keep
#apply dct on the filter bank energies - gives mfcc
mfcc = dct(filter_bank_energy, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

output_file = "mfcc_feat.txt"
np.savetxt(output_file, mfcc)

