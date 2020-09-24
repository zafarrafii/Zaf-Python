"""
This Python module implements several functions for audio signal analysis.

Functions:
    stft - Short-time Fourier transform (STFT)
    istft - inverse STFT
    cqtkernel - Constant-Q transform (CQT) kernel
    cqtspectrogram - CQT spectrogram using a CQT kernel
    cqtchromagram - CQT chromagram using a CQT kernel
    mfcc - Mel frequency cepstrum coefficients (MFCCs)
    dct - Discrete cosine transform (DCT) using the fast Fourier transform (FFT)
    dst - Discrete sine transform (DST) using the FFT
    mdct - Modified discrete cosine transform (MDCT) using the FFT
    imdct - Inverse MDCT using the FFT

Other:
    wavread - Read a WAVE file (using Scipy)
    wavwrite - Write a WAVE file (using Scipy)
    sigplot - Plot an audio signal in seconds
    specshow - Display an audio spectrogram in dB, seconds, and Hz

Author:
    Zafar Rafii
    zafarrafii@gmail.com
    http://zafarrafii.com
    https://github.com/zafarrafii
    https://www.linkedin.com/in/zafarrafii/
    09/24/20
"""

import numpy as np
import scipy.sparse
import scipy.signal
import scipy.fftpack
import scipy.io.wavfile
import matplotlib.pyplot as plt


def stft(audio_signal, window_function, step_length):
    """
    Short-time Fourier transform (STFT)

    Parameters:
        audio_signal: audio signal [number_samples,]
        window_function: window function [window_length,]
        step_length: step length in samples
    Returns:
        audio_stft: audio STFT [window_length, number_frames]

    Example: compute and display the spectrogram from an audio file
        # Import modules
        import numpy as np
        import scipy.signal
        import zaf
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz, and average it over its channels
        audio_signal, sampling_frequency = zaf.wavread('audio_file.wav')
        audio_signal = np.mean(audio_signal, 1)

        # Set the window duration in seconds (audio is stationary around 40 milliseconds)
        window_duration = 0.04

        # Derive the window length in samples (use powers of 2 for faster FFT and constant overlap-add (COLA))
        window_length = pow(2, int(np.ceil(np.log2(window_duration * sampling_frequency))))

        # Compute the window function (use SciPy's periodic Hamming window for COLA as NumPy's Hamming window is symmetric)
        window_function = scipy.signal.hamming(window_length, False)

        # Set the step length in samples (half of the window length for COLA)
        step_length = int(window_length/2)

        # Compute the STFT
        audio_stft = zaf.stft(audio_signal, window_function, step_length)

        # Derive the magnitude spectrogram without the DC component and the mirrored frequencies
        audio_spectrogram = np.absolute(audio_stft[1:int(window_length/2+1), :])

        # Display the spectrogram in dB, second, and Hz (with a resolution of 1 second and 1 kHz)
        plt.figure(figsize=(17, 10))
        zaf.specshow(audio_spectrogram, len(audio_signal), sampling_frequency, xtick_resolution=1, ytick_resolution=1000)
        plt.show()
    """

    # Get the number of samples and the window length in samples
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Derive the zero-padding length at the start and at the end of the signal to center the windows
    padding_length = int(np.floor(window_length / 2))

    # Derive the number of time frames given the zero-padding at the start and at the end of the signal
    number_times = (
        int(
            np.ceil(
                ((number_samples + 2 * padding_length) - window_length) / step_length
            )
        )
        + 1
    )

    # Zero-pad the start and the end of the signal to center the windows
    audio_signal = np.pad(
        audio_signal,
        (
            padding_length,
            (
                number_times * step_length
                + (window_length - step_length)
                - padding_length
            )
            - number_samples,
        ),
        "constant",
        constant_values=0,
    )

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times))

    # Loop over the time frames
    i = 0
    for j in range(0, number_times):

        # Window the signal
        audio_stft[:, j] = audio_signal[i : i + window_length] * window_function
        i = i + step_length

    # Compute the Fourier transform of the frames
    audio_stft = np.fft.fft(audio_stft, axis=0)

    return audio_stft


def istft(audio_stft, window_function, step_length):
    """
    Inverse short-time Fourier transform (STFT)

    Parameters:
        audio_stft: audio STFT [window_length, number_frames]
        window_function: window function [window_length,]
        step_length: step length in samples
    Returns:
        audio_signal: audio signal [number_samples,]

    Example: estimate the center and the sides from a stereo audio file
        # Import modules
        import numpy as np
        import scipy.signal
        import zaf
        import matplotlib.pyplot as plt

        # Read the (stereo) audio signal with its sampling frequency in Hz
        audio_signal, sampling_frequency = zaf.wavread('audio_file.wav')

        # Set the parameters for the STFT
        window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
        window_function = scipy.signal.hamming(window_length, False)
        step_length = int(window_length/2)

        # Compute the STFTs for the left and right channels
        audio_stft1 = zaf.stft(audio_signal[:, 0], window_function, step_length)
        audio_stft2 = zaf.stft(audio_signal[:, 1], window_function, step_length)

        # Derive the magnitude spectrograms (with DC component) for the left and right channels
        audio_spectrogram1 = abs(audio_stft1[0:int(window_length/2)+1, :])
        audio_spectrogram2 = abs(audio_stft2[0:int(window_length/2)+1, :])

        # Estimate the time-frequency masks for the left and right channels for the center
        center_mask1 = np.minimum(audio_spectrogram1, audio_spectrogram2)/audio_spectrogram1
        center_mask2 = np.minimum(audio_spectrogram1, audio_spectrogram2)/audio_spectrogram2

        # Derive the STFTs for the left and right channels for the center (with mirrored frequencies)
        center_stft1 = np.multiply(np.concatenate((center_mask1, center_mask1[int(window_length/2)-1:0:-1, :])), audio_stft1)
        center_stft2 = np.multiply(np.concatenate((center_mask2, center_mask2[int(window_length/2)-1:0:-1, :])), audio_stft2)

        # Synthesize the signals for the left and right channels for the center
        center_signal1 = zaf.istft(center_stft1, window_function, step_length)
        center_signal2 = zaf.istft(center_stft2, window_function, step_length)

        # Derive the final stereo center and sides signals
        center_signal = np.stack((center_signal1, center_signal2), axis=1)
        center_signal = center_signal[0:len(audio_signal), :]
        sides_signal = audio_signal-center_signal

        # Write the center and sides signals
        zaf.wavwrite(center_signal, sampling_frequency, 'center_file.wav')
        zaf.wavwrite(sides_signal, sampling_frequency, 'sides_file.wav')

        # Original, center, and sides signals displayed in s
        plt.figure(figsize=(17, 10))
        plt.subplot(311),
        zaf.sigplot(audio_signal, sampling_frequency, xtick_resolution=1), plt.ylim(-1, 1), plt.title("Original Signal")
        plt.subplot(312)
        zaf.sigplot(center_signal, sampling_frequency, xtick_resolution=1), plt.ylim(-1, 1), plt.title("Center Signal")
        plt.subplot(313)
        zaf.sigplot(sides_signal, sampling_frequency, xtick_resolution=1), plt.ylim(-1, 1), plt.title("Sides Signal")
        plt.show()
    """

    # Get the window length in samples and the number of time frames
    window_length, number_times = np.shape(audio_stft)

    # Derive the number of samples for the signal
    number_samples = number_times * step_length + (window_length - step_length)

    # Initialize the signal
    audio_signal = np.zeros(number_samples)

    # Compute the inverse Fourier transform of the frames and take the real part to ensure real values
    audio_stft = np.real(np.fft.ifft(audio_stft, axis=0))

    # Loop over the time frames
    i = 0
    for j in range(0, number_times):

        # Perform a constant overlap-add (COLA) of the signal (with proper window function and step length)
        audio_signal[i : i + window_length] = (
            audio_signal[i : i + window_length] + audio_stft[:, j]
        )
        i = i + step_length

    # Remove the zero-padding at the start and at the end of the signal
    audio_signal = audio_signal[
        window_length - step_length : number_samples - (window_length - step_length)
    ]

    # Normalize the signal by the gain introduced by the COLA (if any)
    audio_signal = audio_signal / sum(window_function[0:window_length:step_length])

    return audio_signal


def cqtkernel(
    sampling_frequency, frequency_resolution, minimum_frequency, maximum_frequency
):
    """
    Constant-Q transform (CQT) kernel

    Parameters:
        sampling_frequency: sampling frequency in Hz
        frequency_resolution: frequency resolution in number of frequency channels per semitone
        minimum_frequency: minimum frequency in Hz
        maximum_frequency: maximum frequency in Hz
    Returns:
        cqt_kernel: CQT kernel [number_frequencies, fft_length]

    Example: compute and display a CQT kernel
        # Import modules
        import numpy as np
        import zaf
        import matplotlib.pyplot as plt

        # Set the parameters for the CQT kernel
        sampling_frequency = 44100
        frequency_resolution = 2
        minimum_frequency = 55
        maximum_frequency = sampling_frequency/2

        # Compute the CQT kernel
        cqt_kernel = zaf.cqtkernel(sampling_frequency, frequency_resolution, minimum_frequency, maximum_frequency)

        # Display the magnitude CQT kernel
        plt.figure(figsize=(17,10))
        plt.imshow(np.absolute(cqt_kernel).toarray(), aspect='auto', cmap='jet', origin='lower')
        plt.title('Magnitude CQT kernel')
        plt.xlabel('FFT length')
        plt.ylabel('CQT frequency')
        plt.show()
    """

    # Derive th number of frequency channels per octave
    octave_resolution = 12 * frequency_resolution

    # Derive the constant ratio of frequency to resolution (= fk/(fk+1-fk))
    quality_factor = 1 / pow(2, (1 / octave_resolution) - 1)

    # Compute the number of frequency channels for the CQT
    number_frequencies = int(
        round(octave_resolution * np.log2(maximum_frequency / minimum_frequency))
    )

    # Compute the window length for the FFT (= window length of the minimum frequency = longest window)
    fft_length = int(
        pow(
            2, np.ceil(np.log2(quality_factor * sampling_frequency / minimum_frequency))
        )
    )

    # Initialize the kernel
    cqt_kernel = np.zeros((number_frequencies, fft_length), dtype=complex)

    # Loop over the frequency channels
    for frequency_index in range(0, number_frequencies):

        # Frequency value (in Hz)
        frequency_value = minimum_frequency * 2 ** (frequency_index / octave_resolution)

        # Window length (nearest odd value because the complex exponential will have an odd length, in samples)
        window_length = (
            2 * round(quality_factor * sample_rate / frequency_value / 2) + 1
        )

        # Temporal kernel (without zero-padding, odd and symmetric)
        temporal_kernel = (
            np.hamming(window_length)
            * np.exp(
                2
                * np.pi
                * 1j
                * quality_factor
                * np.arange(-(window_length - 1) / 2, (window_length - 1) / 2 + 1)
                / window_length
            )
            / window_length
        )

        # Pre zero-padding to center FFTs (fft does post zero-padding; temporal kernel still odd but almost symmetric)
        temporal_kernel = np.pad(
            temporal_kernel,
            (int((fft_length - window_length + 1) / 2), 0),
            "constant",
            constant_values=0,
        )

        # Spectral kernel (mostly real because temporal kernel almost symmetric)
        spectral_kernel = np.fft.fft(temporal_kernel, fft_length)

        # Save the spectral kernels
        cqt_kernel[frequency_index, :] = spectral_kernel

    # Energy threshold for making the kernel sparse
    energy_threshold = 0.01

    # Make the CQT kernel sparser
    cqt_kernel[np.absolute(cqt_kernel) < energy_threshold] = 0

    # Make the CQT kernel sparse
    cqt_kernel = scipy.sparse.csc_matrix(cqt_kernel)

    # From Parseval's theorem
    cqt_kernel = np.conjugate(cqt_kernel) / fft_length

    return cqt_kernel


def cqtspectrogram(audio_signal, sampling_frequency, time_resolution, cqt_kernel):
    """
    Constant-Q transform (CQT) spectrogram using a kernel

    Parameters:
        audio_signal: audio signal [number_samples,]
        sampling_frequency: sampling frequency in Hz
        time_resolution: time resolution in number of time frames per second
        cqt_kernel: CQT kernel [number_frequencies, fft_length]
    Returns:
        audio_spectrogram: audio spectrogram in magnitude [number_frequencies, number_times]

    Example: compute and display the CQT spectrogram
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt

        # Audio file (normalized) averaged over the channels and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / ( 2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)

        # CQT kernel
        frequency_resolution = 2
        minimum_frequency = 55
        maximum_frequency = 3520
        cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)

        # CQT spectrogram
        time_resolution = 25
        audio_spectrogram = z.cqtspectrogram(audio_signal, sample_rate, time_resolution, cqt_kernel)

        # CQT spectrogram displayed in dB, s, and semitones
        plt.rc('font', size=30)
        plt.imshow(20*np.log10(audio_spectrogram), aspect='auto', cmap='jet', origin='lower')
        plt.title('CQT spectrogram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*time_resolution),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.arange(1, 6*12*frequency_resolution+1, 12*frequency_resolution),
                   ('A1 (55 Hz)','A2 (110 Hz)','A3 (220 Hz)','A4 (440 Hz)','A5 (880 Hz)','A6 (1760 Hz)'))
        plt.ylabel('Frequency (semitones)')
        plt.show()
    """

    # Number of time samples per time frame
    step_length = round(sampling_frequency / time_resolution)

    # Number of time frames
    number_times = int(np.floor(len(audio_signal) / step_length))

    # Number of frequency channels and FFT length
    number_frequencies, fft_length = np.shape(cqt_kernel)

    # Zero-padding to center the CQT
    audio_signal = np.pad(
        audio_signal,
        (
            int(np.ceil((fft_length - step_length) / 2)),
            int(np.floor((fft_length - step_length) / 2)),
        ),
        "constant",
        constant_values=(0, 0),
    )

    # Initialize the spectrogram
    audio_spectrogram = np.zeros((number_frequencies, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Magnitude CQT using the kernel
        sample_index = time_index * step_length
        audio_spectrogram[:, time_index] = abs(
            cqt_kernel
            * np.fft.fft(audio_signal[sample_index : sample_index + fft_length])
        )

    return audio_spectrogram


def cqtchromagram(
    audio_signal, sampling_frequency, time_resolution, frequency_resolution, cqt_kernel
):
    """
    Constant-Q transform (CQT) chromagram using a kernel

    Parameters:
        audio_signal: audio signal [number_samples,]
        sampling_frequency: sampling frequency in Hz
        time_resolution: time resolution in number of time frames per second
        frequency_resolution: frequency resolution in number of frequency channels per semitones
        cqt_kernel: CQT kernel [number_frequencies, fft_length]
    Returns
        audio_chromagram: audio chromagram [number_chromas, number_times]

    Example: compute and display the CQT chromagram
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt

        # Audio signal (normalized) averaged over its channels and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)

        # CQT kernel
        frequency_resolution = 2
        minimum_frequency = 55
        maximum_frequency = 3520
        cqt_kernel = z.cqtkernel(sample_rate, frequency_resolution, minimum_frequency, maximum_frequency)

        # CQT chromagram
        time_resolution = 25
        audio_chromagram = z.cqtchromagram(audio_signal, sample_rate, time_resolution, frequency_resolution, cqt_kernel)

        # CQT chromagram displayed in dB, s, and chromas
        plt.rc('font', size=30)
        plt.imshow(20*np.log10(audio_chromagram), aspect='auto', cmap='jet', origin='lower')
        plt.title('CQT chromagram (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*time_resolution),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.arange(1, 12*frequency_resolution+1, frequency_resolution),
                   ('A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'))
        plt.ylabel('Chroma')
        plt.show()
    """

    # CQT spectrogram
    audio_spectrogram = cqtspectrogram(
        audio_signal, sampling_frequency, time_resolution, cqt_kernel
    )

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Number of chroma bins
    number_chromas = 12 * frequency_resolution

    # Initialize the chromagram
    audio_chromagram = np.zeros((number_chromas, number_times))

    # Loop over the chroma bins
    for chroma_index in range(0, number_chromas):

        # Sum the energy of the frequency channels for every chroma
        audio_chromagram[chroma_index, :] = np.sum(
            audio_spectrogram[chroma_index:number_frequencies:number_chromas, :], 0
        )

    return audio_chromagram


def mfcc(audio_signal, sample_rate, number_filters, number_coefficients):
    """
    Mel frequency cepstrum coefficients (MFFCs)

    Parameters:
        audio_signal: audio signal [number_samples,]
        sampling_frequency: sampling frequency in Hz
        number_filters: number of filters
        number_coefficients: number of coefficients (without the 0th coefficient)
    Returns:
        audio_mfcc: audio MFCCs [number_times, number_coefficients]

    Example: compute and display the MFCCs, delta MFCCs, and delta-detla MFCCs
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt

        # Audio signal (normalized) averaged over its channels and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)

        # MFCCs for a given number of filters and coefficients
        number_filters = 40
        number_coefficients = 20
        audio_mfcc = z.mfcc(audio_signal, sample_rate, number_filters, number_coefficients)

        # Delta and delta-delta MFCCs
        audio_deltamfcc = np.diff(audio_mfcc, n=1, axis=1)
        audio_deltadeltamfcc = np.diff(audio_deltamfcc, n=1, axis=1)

        # MFCCs, delta MFCCs, and delta-delta MFCCs displayed in s
        step_length = 2**np.ceil(np.log2(0.04*sample_rate)) / 2
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1), plt.plot(np.transpose(audio_mfcc)), plt.autoscale(tight=True), plt.title('MFCCs')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.subplot(3, 1, 2), plt.plot(np.transpose(audio_deltamfcc)), plt.autoscale(tight=True), plt.title('Delta MFCCs')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.subplot(3, 1, 3), plt.plot(np.transpose(audio_deltadeltamfcc)), plt.autoscale(tight=True), plt.title('Delta-delta MFCCs')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.show()
    """

    # Window duration in seconds, length in samples, and function, and step length in samples
    window_duration = 0.04
    window_length = int(2 ** np.ceil(np.log2(window_duration * sample_rate)))
    window_function = scipy.signal.hamming(window_length, False)
    step_length = int(window_length / 2)

    # Magnitude spectrogram (without the DC component and the mirrored frequencies)
    audio_stft = stft(audio_signal, window_function, step_length)
    audio_spectrogram = abs(audio_stft[1 : int(window_length / 2) + 1, :])

    # Minimum and maximum mel frequencies
    mininum_melfrequency = 2595 * np.log10(1 + (sample_rate / window_length) / 700)
    maximum_melfrequency = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    # Indices of the overlapping filters (linearly spaced in the mel scale and log spaced in the linear scale)
    filter_width = (
        2 * (maximum_melfrequency - mininum_melfrequency) / (number_filters + 1)
    )
    filter_indices = np.arange(
        mininum_melfrequency, maximum_melfrequency + 1, filter_width / 2
    )
    filter_indices = np.round(
        700 * (np.power(10, filter_indices / 2595) - 1) * window_length / sample_rate
    ).astype(int)

    # Initialize the filter bank
    filter_bank = np.zeros((number_filters, int(window_length / 2)))

    # Loop over the filters
    for filter_index in range(0, number_filters):

        # Left and right sides of the triangular overlapping filters (linspace more accurate than triang or bartlett!)
        filter_bank[
            filter_index,
            filter_indices[filter_index] - 1 : filter_indices[filter_index + 1],
        ] = np.linspace(
            0,
            1,
            num=filter_indices[filter_index + 1] - filter_indices[filter_index] + 1,
        )
        filter_bank[
            filter_index,
            filter_indices[filter_index + 1] - 1 : filter_indices[filter_index + 2],
        ] = np.linspace(
            1,
            0,
            num=filter_indices[filter_index + 2] - filter_indices[filter_index + 1] + 1,
        )

    # Discrete cosine transform of the log of the magnitude spectrogram mapped onto the mel scale using the filter bank
    audio_mfcc = scipy.fftpack.dct(
        np.log(np.dot(filter_bank, audio_spectrogram) + np.spacing(1)),
        axis=0,
        norm="ortho",
    )

    # The first coefficients (without the 0th) represent the MFCCs
    audio_mfcc = audio_mfcc[1 : number_coefficients + 1, :]

    return audio_mfcc


def dct(audio_signal, dct_type):
    """
    Discrete cosine transform (DCT) using the fast Fourier transform (FFT)

    Parameters:
        audio_signal: audio signal [number_samples, number_frames] (number_frames > 0)
        dct_type: DCT type (1, 2, 3, or 4)
    Returns:
        audio_dct: audio DCT [number_frequencies, number_frames]

    Example: compute the 4 different DCTs and compare them to SciPy's DCTs
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import scipy.fftpack
        import matplotlib.pyplot as plt

        # Audio signal (normalized) averaged over its channels (expanded) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)
        audio_signal = np.expand_dims(audio_signal, axis=1)

        # Audio signal for a given window length, and one frame
        window_length = 1024
        audio_signal = audio_signal[0:window_length, :]

        # DCT-I, II, III, and IV
        audio_dct1 = z.dct(audio_signal, 1)
        audio_dct2 = z.dct(audio_signal, 2)
        audio_dct3 = z.dct(audio_signal, 3)
        audio_dct4 = z.dct(audio_signal, 4)

        # SciPy's DCT-I (orthogonalized), II, and III (SciPy does not have a DCT-IV!)
        audio_signal1 = np.concatenate((audio_signal[0:1, :]*np.sqrt(2), audio_signal[1:window_length-1, :],
                                       audio_signal[window_length-1:window_length, :]*np.sqrt(2)))
        scipy_dct1 = scipy.fftpack.dct(audio_signal1, axis=0, type=1)
        scipy_dct1[[0, window_length-1], :] = scipy_dct1[[0, window_length-1], :]/np.sqrt(2)
        scipy_dct1 = scipy_dct1*np.sqrt(2/(window_length-1))/2
        scipy_dct2 = scipy.fftpack.dct(audio_signal, axis=0, type=2, norm='ortho')
        scipy_dct3 = scipy.fftpack.dct(audio_signal, axis=0, type=3, norm='ortho')

        # DCT-I, II, III, and IV, SciPy's versions, and  errors displayed
        plt.rc('font', size=30)
        plt.subplot(4, 3, 1), plt.plot(audio_dct1), plt.autoscale(tight=True), plt.title("DCT-I")
        plt.subplot(4, 3, 2), plt.plot(scipy_dct1), plt.autoscale(tight=True), plt.title("SciPy's DCT-I")
        plt.subplot(4, 3, 3), plt.plot(audio_dct1-scipy_dct1), plt.autoscale(tight=True), plt.title("Error")
        plt.subplot(4, 3, 4), plt.plot(audio_dct2), plt.autoscale(tight=True), plt.title("DCT-II")
        plt.subplot(4, 3, 5), plt.plot(scipy_dct2), plt.autoscale(tight=True), plt.title("SciPy's DCT-II")
        plt.subplot(4, 3, 6), plt.plot(audio_dct2-scipy_dct2), plt.autoscale(tight=True), plt.title("Error")
        plt.subplot(4, 3, 7), plt.plot(audio_dct3), plt.autoscale(tight=True), plt.title("DCT-III")
        plt.subplot(4, 3, 8), plt.plot(scipy_dct3), plt.autoscale(tight=True), plt.title("SciPy's DCT-III")
        plt.subplot(4, 3, 9), plt.plot(audio_dct3-scipy_dct3), plt.autoscale(tight=True), plt.title("Error")
        plt.subplot(4, 3, 10), plt.plot(audio_dct4), plt.autoscale(tight=True), plt.title("DCT-IV")
        plt.show()
    """

    if dct_type == 1:

        # Number of samples per frame
        window_length = np.size(audio_signal, 0)

        # Pre-processing to make the DCT-I matrix orthogonal (concatenate to avoid the input to change!)
        audio_signal = np.concatenate(
            (
                audio_signal[0:1, :] * np.sqrt(2),
                audio_signal[1 : window_length - 1, :],
                audio_signal[window_length - 1 : window_length, :] * np.sqrt(2),
            )
        )

        # Compute the DCT-I using the FFT
        audio_dct = np.concatenate(
            (audio_signal, audio_signal[window_length - 2 : 0 : -1, :])
        )
        audio_dct = np.fft.fft(audio_dct, axis=0)
        audio_dct = np.real(audio_dct[0:window_length, :]) / 2

        # Post-processing to make the DCT-I matrix orthogonal
        audio_dct[[0, window_length - 1], :] = audio_dct[
            [0, window_length - 1], :
        ] / np.sqrt(2)
        audio_dct = audio_dct * np.sqrt(2 / (window_length - 1))

        return audio_dct

    elif dct_type == 2:

        # Number of samples and frames
        window_length, number_frames = np.shape(audio_signal)

        # Compute the DCT-II using the FFT
        audio_dct = np.zeros((4 * window_length, number_frames))
        audio_dct[1 : 2 * window_length : 2, :] = audio_signal
        audio_dct[2 * window_length + 1 : 4 * window_length : 2, :] = audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dct = np.fft.fft(audio_dct, axis=0)
        audio_dct = np.real(audio_dct[0:window_length, :]) / 2

        # Post-processing to make the DCT-II matrix orthogonal
        audio_dct[0, :] = audio_dct[0, :] / np.sqrt(2)
        audio_dct = audio_dct * np.sqrt(2 / window_length)

        return audio_dct

    elif dct_type == 3:

        # Number of samples and frames
        window_length, number_frames = np.shape(audio_signal)

        # Pre-processing to make the DCT-III matrix orthogonal (concatenate to avoid the input to change!)
        audio_signal = np.concatenate(
            (audio_signal[0:1, :] * np.sqrt(2), audio_signal[1:window_length, :])
        )

        # Compute the DCT-III using the FFT
        audio_dct = np.zeros((4 * window_length, number_frames))
        audio_dct[0:window_length, :] = audio_signal
        audio_dct[window_length + 1 : 2 * window_length + 1, :] = -audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dct[2 * window_length + 1 : 3 * window_length, :] = -audio_signal[
            1:window_length, :
        ]
        audio_dct[3 * window_length + 1 : 4 * window_length, :] = audio_signal[
            window_length - 1 : 0 : -1, :
        ]
        audio_dct = np.fft.fft(audio_dct, axis=0)
        audio_dct = np.real(audio_dct[1 : 2 * window_length : 2, :]) / 4

        # Post-processing to make the DCT-III matrix orthogonal
        audio_dct = audio_dct * np.sqrt(2 / window_length)

        return audio_dct

    elif dct_type == 4:

        # Number of samples and frames
        window_length, number_frames = np.shape(audio_signal)

        # Compute the DCT-IV using the FFT
        audio_dct = np.zeros((8 * window_length, number_frames))
        audio_dct[1 : 2 * window_length : 2, :] = audio_signal
        audio_dct[2 * window_length + 1 : 4 * window_length : 2, :] = -audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dct[4 * window_length + 1 : 6 * window_length : 2, :] = -audio_signal
        audio_dct[6 * window_length + 1 : 8 * window_length : 2, :] = audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dct = np.fft.fft(audio_dct, axis=0)
        audio_dct = np.real(audio_dct[1 : 2 * window_length : 2, :]) / 4

        # Post-processing to make the DCT-IV matrix orthogonal
        audio_dct = np.sqrt(2 / window_length) * audio_dct

        return audio_dct


def dst(audio_signal, dst_type):
    """
    Discrete sine transform (DST) using the fast Fourier transform (FFT)

    Parameters:
        audio_signal: audio signal [number_samples, number_frames] (number_frames > 0)
        dst_type: DST type (1, 2, 3, or 4)
    Returns:
        audio_dst: audio DST [number_frequencies, number_frames]

    Example: compute the 4 different DSTs and compare them to their respective inverses
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import scipy.fftpack
        import matplotlib.pyplot as plt

        # Audio signal (normalized) averaged over its channels (expanded) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)
        audio_signal = np.expand_dims(audio_signal, axis=1)

        # Audio signal for a given window length, and one frame
        window_length = 1024
        audio_signal = audio_signal[0:window_length, :]

        # DST-I, II, III, and IV
        audio_dst1 = z.dst(audio_signal, 1)
        audio_dst2 = z.dst(audio_signal, 2)
        audio_dst3 = z.dst(audio_signal, 3)
        audio_dst4 = z.dst(audio_signal, 4)

        # Respective inverses, i.e., DST-I, III, II, and IV
        audio_idst1 = z.dst(audio_dst1, 1)
        audio_idst2 = z.dst(audio_dst2, 3)
        audio_idst3 = z.dst(audio_dst3, 2)
        audio_idst4 = z.dst(audio_dst4, 4)

        # DST-I, II, III, and IV, corresponding inverses, and errors displayed
        plt.rc('font', size=30)
        plt.subplot(4, 3, 1), plt.plot(audio_dst1), plt.autoscale(tight=True), plt.title("DCT-I")
        plt.subplot(4, 3, 2), plt.plot(audio_idst1), plt.autoscale(tight=True), plt.title("Inverse DST-I = DST-I")
        plt.subplot(4, 3, 3), plt.plot(audio_signal-audio_idst1), plt.autoscale(tight=True), plt.title("Error")
        plt.subplot(4, 3, 4), plt.plot(audio_dst2), plt.autoscale(tight=True), plt.title("DST-II")
        plt.subplot(4, 3, 5), plt.plot(audio_idst2), plt.autoscale(tight=True), plt.title("Inverse DST-II = DST-III")
        plt.subplot(4, 3, 6), plt.plot(audio_signal-audio_idst2), plt.autoscale(tight=True), plt.title("Error")
        plt.subplot(4, 3, 7), plt.plot(audio_dst3), plt.autoscale(tight=True), plt.title("DST-III")
        plt.subplot(4, 3, 8), plt.plot(audio_idst3), plt.autoscale(tight=True), plt.title("Inverse DST-III = DST-II")
        plt.subplot(4, 3, 9), plt.plot(audio_signal-audio_idst3), plt.autoscale(tight=True), plt.title("Error")
        plt.subplot(4, 3, 10), plt.plot(audio_dst4), plt.autoscale(tight=True), plt.title("DST-IV")
        plt.subplot(4, 3, 11), plt.plot(audio_idst4), plt.autoscale(tight=True), plt.title("Inverse DST-IV = DST-IV")
        plt.subplot(4, 3, 12), plt.plot(audio_signal-audio_idst4), plt.autoscale(tight=True), plt.title("Error")
        plt.show()
    """
    if dst_type == 1:

        # Number of samples per frame
        window_length, number_frames = np.shape(audio_signal)

        # Compute the DST-I using the FFT
        audio_dst = np.concatenate(
            (
                np.zeros((1, number_frames)),
                audio_signal,
                np.zeros((1, number_frames)),
                -audio_signal[window_length - 1 :: -1, :],
            )
        )
        audio_dst = np.fft.fft(audio_dst, axis=0)
        audio_dst = -np.imag(audio_dst[1 : window_length + 1, :]) / 2

        # Post-processing to make the DST-I matrix orthogonal
        audio_dst = audio_dst * np.sqrt(2 / (window_length + 1))

        return audio_dst

    elif dst_type == 2:

        # Number of samples and frames
        window_length, number_frames = np.shape(audio_signal)

        # Compute the DST-II using the FFT
        audio_dst = np.zeros((4 * window_length, number_frames))
        audio_dst[1 : 2 * window_length : 2, :] = audio_signal
        audio_dst[2 * window_length + 1 : 4 * window_length : 2, :] = -audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dst = np.fft.fft(audio_dst, axis=0)
        audio_dst = -np.imag(audio_dst[1 : window_length + 1, :]) / 2

        # Post-processing to make the DST-II matrix orthogonal
        audio_dst[window_length - 1, :] = audio_dst[window_length - 1, :] / np.sqrt(2)
        audio_dst = audio_dst * np.sqrt(2 / window_length)

        return audio_dst

    elif dst_type == 3:

        # Number of samples and frames
        window_length, number_frames = np.shape(audio_signal)

        # Pre-processing to make the DST-III matrix orthogonal (concatenate to avoid the input to change!)
        audio_signal = np.concatenate(
            (
                audio_signal[0 : window_length - 1, :],
                audio_signal[window_length - 1 : window_length, :] * np.sqrt(2),
            )
        )

        # Compute the DST-III using the FFT
        audio_dst = np.zeros((4 * window_length, number_frames))
        audio_dst[1 : window_length + 1, :] = audio_signal
        audio_dst[window_length + 1 : 2 * window_length, :] = audio_signal[
            window_length - 2 :: -1, :
        ]
        audio_dst[2 * window_length + 1 : 3 * window_length + 1, :] = -audio_signal
        audio_dst[3 * window_length + 1 : 4 * window_length, :] = -audio_signal[
            window_length - 2 :: -1, :
        ]
        audio_dst = np.fft.fft(audio_dst, axis=0)
        audio_dst = -np.imag(audio_dst[1 : 2 * window_length : 2, :]) / 4

        # Post-processing to make the DST-III matrix orthogonal
        audio_dst = audio_dst * np.sqrt(2 / window_length)

        return audio_dst

    elif dst_type == 4:

        # Number of samples and frames
        window_length, number_frames = np.shape(audio_signal)

        # Compute the DST-IV using the FFT
        audio_dst = np.zeros((8 * window_length, number_frames))
        audio_dst[1 : 2 * window_length : 2, :] = audio_signal
        audio_dst[2 * window_length + 1 : 4 * window_length : 2, :] = audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dst[4 * window_length + 1 : 6 * window_length : 2, :] = -audio_signal
        audio_dst[6 * window_length + 1 : 8 * window_length : 2, :] = -audio_signal[
            window_length - 1 :: -1, :
        ]
        audio_dst = np.fft.fft(audio_dst, axis=0)
        audio_dst = -np.imag(audio_dst[1 : 2 * window_length : 2, :]) / 4

        # Post-processing to make the DST-IV matrix orthogonal
        audio_dst = audio_dst * np.sqrt(2 / window_length)

        return audio_dst


def mdct(audio_signal, window_function):
    """
    Modified discrete cosine transform (MDCT) using the fast Fourier transform (FFT)

    Parameters:
        audio_signal: audio signal [number_samples,]
        window_function: window function [window_length,]
    Returns:
        audio_mdct: audio MDCT [number_frequencies, number_times]

    Example: compute and display the MDCT as in the AC-3 audio coding format
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt

        # Audio signal (normalized) averaged over its channels (expanded) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))
        audio_signal = np.mean(audio_signal, 1)

        # Kaiser-Bessel-derived (KBD) window as used in the AC-3 audio coding format
        window_length = 512
        alpha_value = 5
        window_function = np.kaiser(int(window_length/2)+1, alpha_value*np.pi)
        window_function2 = np.cumsum(window_function[0:int(window_length/2)])
        window_function = np.sqrt(np.concatenate((window_function2, window_function2[int(window_length/2)::-1]))
                                  / np.sum(window_function))

        # MDCT
        audio_mdct = z.mdct(audio_signal, window_function)

        # MDCT displayed in dB, s, and kHz
        plt.rc('font', size=30)
        plt.imshow(20*np.log10(np.absolute(audio_mdct)), aspect='auto', cmap='jet', origin='lower')
        plt.title('MDCT (dB)')
        plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/(window_length/2)),
                   np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
        plt.xlabel('Time (s)')
        plt.yticks(np.round(np.arange(1e3, sample_rate/2+1, 1e3)/sample_rate*window_length),
                   np.arange(1, int(sample_rate/2*1e3)+1))
        plt.ylabel('Frequency (kHz)')
        plt.show()
    """

    # Number of samples and window length
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil(2 * number_samples / window_length) + 1)

    # Pre and post zero-padding of the signal
    audio_signal = np.pad(
        audio_signal,
        (
            int(window_length / 2),
            int((number_times + 1) * window_length / 2 - number_samples),
        ),
        "constant",
        constant_values=0,
    )

    # Initialize the MDCT
    audio_mdct = np.zeros((int(window_length / 2), number_times))

    # Pre and post-processing arrays
    preprocessing_array = np.exp(
        -1j * np.pi / window_length * np.arange(0, window_length)
    )
    postprocessing_array = np.exp(
        -1j
        * np.pi
        / window_length
        * (window_length / 2 + 1)
        * np.arange(0.5, window_length / 2 + 0.5)
    )

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Window the signal
        sample_index = time_index * int(window_length / 2)
        audio_segment = (
            audio_signal[sample_index : sample_index + window_length] * window_function
        )

        # FFT of the audio segment after pre-processing
        audio_segment = np.fft.fft(audio_segment * preprocessing_array)

        # Truncate to the first half before post-processing
        audio_mdct[:, time_index] = np.real(
            audio_segment[0 : int(window_length / 2)] * postprocessing_array
        )

    return audio_mdct


def imdct(audio_mdct, window_function):
    """
    Inverse modified discrete cosine transform (MDCT) using the fast Fourier transform (FFT)

    Parameters:
        audio_mdct: audio MDCT [number_frequencies, number_times]
        window_function: window function [window_length,]
    Returns:
        audio_signal: audio signal [number_samples,]

    Example: verify that the MDCT is perfectly invertible
        # Import modules
        import scipy.io.wavfile
        import numpy as np
        import z
        import matplotlib.pyplot as plt

        # Audio signal (normalized) averaged over its channels (expanded) and sample rate in Hz
        sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
        audio_signal = audio_signal / (2.0 ** (audio_signal.itemsize * 8 - 1))
        audio_signal = np.mean(audio_signal, 1)

        # MDCT with a slope function as used in the Vorbis audio coding format
        window_length = 2048
        window_function = np.sin(np.pi / 2
                                 * np.power(np.sin(np.pi / window_length * np.arange(0.5, window_length + 0.5)), 2))
        audio_mdct = z.mdct(audio_signal, window_function)

        # Inverse MDCT and error signal
        audio_signal2 = z.imdct(audio_mdct, window_function)
        audio_signal2 = audio_signal2[0:len(audio_signal)]
        error_signal = audio_signal - audio_signal2

        # Original, resynthesized, and error signals displayed in s
        plt.rc('font', size=30)
        plt.subplot(3, 1, 1), plt.plot(audio_signal), plt.autoscale(tight=True), plt.title("Original Signal")
        plt.xticks(np.arange(sample_rate, len(audio_signal), sample_rate),
                   np.arange(1, int(np.floor(len(audio_signal) / sample_rate)) + 1))
        plt.xlabel('Time (s)')
        plt.subplot(3, 1, 2), plt.plot(audio_signal2), plt.autoscale(tight=True), plt.title("Resynthesized Signal")
        plt.xticks(np.arange(sample_rate, len(audio_signal), sample_rate),
                   np.arange(1, int(np.floor(len(audio_signal) / sample_rate)) + 1))
        plt.xlabel('Time (s)')
        plt.subplot(3, 1, 3), plt.plot(error_signal), plt.autoscale(tight=True), plt.title("Error Signal")
        plt.xticks(np.arange(sample_rate, len(audio_signal), sample_rate),
                   np.arange(1, int(np.floor(len(audio_signal) / sample_rate)) + 1))
        plt.xlabel('Time (s)')
        plt.show()
    """

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_mdct)

    # Number of samples for the signal
    number_samples = number_frequencies * (number_times + 1)

    # Initialize the audio signal
    audio_signal = np.zeros(number_samples)

    # Pre and post-processing arrays
    preprocessing_array = np.exp(
        -1j
        * np.pi
        / (2 * number_frequencies)
        * (number_frequencies + 1)
        * np.arange(0, number_frequencies)
    )
    postprocessing_array = (
        np.exp(
            -1j
            * np.pi
            / (2 * number_frequencies)
            * np.arange(
                0.5 + number_frequencies / 2,
                2 * number_frequencies + number_frequencies / 2 + 0.5,
            )
        )
        / number_frequencies
    )

    # FFT of the frames after pre-processing
    audio_mdct = np.fft.fft(
        audio_mdct.T * preprocessing_array, n=2 * number_frequencies, axis=1
    )

    # Apply the window to the frames after post-processing
    audio_mdct = 2 * (np.real(audio_mdct * postprocessing_array) * window_function).T

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Recover the signal thanks to the time-domain aliasing cancellation (TDAC) principle
        sample_index = time_index * number_frequencies
        audio_signal[sample_index : sample_index + 2 * number_frequencies] = (
            audio_signal[sample_index : sample_index + 2 * number_frequencies]
            + audio_mdct[:, time_index]
        )

    # Remove the pre and post zero-padding
    audio_signal = audio_signal[number_frequencies : -number_frequencies - 1]

    return audio_signal


def wavread(audio_file):
    """
    Read a WAVE file (using SciPy)

    Parameters:
        audio_file: path to an audio file
    Returns:
        audio_signal: audio signal [number_samples, number_channels]
        sampling_frequency: sampling frequency in Hz
    """

    # Read the audio file and return the sampling frequency in Hz and the non-normalized signal using SciPy
    sampling_frequency, audio_signal = scipy.io.wavfile.read(audio_file)

    # Normalize the signal by the data range given the size of an item in bytes
    audio_signal = audio_signal / np.power(2, audio_signal.itemsize * 8 - 1)

    return audio_signal, sampling_frequency


def wavwrite(audio_signal, sampling_frequency, audio_file):
    """
    Write a WAVE file (using Scipy)

    Parameters:
        audio_signal: audio signal [number_samples, number_channels]
        sampling_frequency: sampling frequency in Hz
    Returns:
        audio_file: path to an audio file
    """

    # Write the audio signal using SciPy
    scipy.io.wavfile.write(audio_file, sampling_frequency, audio_signal)


def sigplot(
    audio_signal,
    sampling_frequency,
    xtick_resolution=1,
):
    """
    Plot an audio signal in seconds

    Parameters:
        audio_signal: audio signal (without DC and mirrored frequencies) [number_samples, number_channels]
        sampling_frequency: sampling frequency in Hz
        xtick_resolution: resolution for the x-axis ticks in seconds (default: 1 second)
    """

    # Get the number of samples
    number_samples = np.shape(audio_signal)[0]

    # Prepare the tick locations and labels for the x-axis
    x_ticks = np.arange(
        xtick_resolution * sampling_frequency,
        number_samples,
        xtick_resolution * sampling_frequency,
    )
    x_labels = np.arange(
        xtick_resolution, number_samples / sampling_frequency + 1, xtick_resolution
    ).astype(int)

    # Plot the signal in seconds
    plt.plot(audio_signal)
    plt.autoscale(tight=True)
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel("Time (s)")


def specshow(
    audio_spectrogram,
    number_samples,
    sampling_frequency,
    xtick_resolution=1,
    ytick_resolution=1000,
):
    """
    Display an audio spectrogram in dB, seconds, and Hz

    Parameters:
        audio_spectrogram: magnitude spectrogram (without DC and mirrored frequencies) [number_frequencies, number_times]
        number_samples: number of samples from the original signal
        sampling_frequency: sampling frequency from the original signal in Hz
        xtick_resolution: resolution for the x-axis ticks in seconds (default: 1 second)
        ytick_resolution: resolution for the y-axis ticks in Hz (default: 1000 Hz)
    """

    # Get the number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Derive the number of Hertz and seconds
    number_hertz = sampling_frequency / 2
    number_seconds = number_samples / sampling_frequency

    # Derive the frequency and time resolutions (i.e., the size of one time-frequency bin in Hz and second)
    frequency_resolution = number_hertz / number_frequencies
    time_resolution = number_seconds / number_times

    # Prepare the tick locations and labels for the x-axis
    x_ticks = np.arange(
        xtick_resolution / time_resolution,
        number_times,
        xtick_resolution / time_resolution,
    )
    x_labels = np.arange(xtick_resolution, number_seconds + 1, xtick_resolution).astype(
        int
    )

    # Prepare the tick locations and labels for the y-axis
    y_ticks = np.arange(
        ytick_resolution / frequency_resolution,
        number_frequencies,
        ytick_resolution / frequency_resolution,
    )
    y_labels = np.arange(ytick_resolution, number_hertz + 1, ytick_resolution).astype(
        int
    )

    # Display the spectrogram in dB, seconds, and Hz
    plt.imshow(
        20 * np.log10(audio_spectrogram), aspect="auto", cmap="jet", origin="lower"
    )
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")