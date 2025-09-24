import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from math import pi
w0 = 1.81e-5
lmda = 665e-9
z0 = pi*w0*w0/lmda
# z0 = 0.0033

def z_formulae(z, phi):
    """
    Models the normalized closed aperture signal as a function of distance.

    Args:
        z: Distance in meters.
        phi: A parameter to fit.

    Returns:
        The modeled normalized closed aperture signal.
    """
    x = z / z0
    return 1 + phi * 4 * x / ((1 + x**2) * (9 + x**2))

def analyze_data_subset(x, y, delta=None):
    """
    Analyzes a subset of data to find max/min y values, corresponding x values,
    their differences, and the midpoint between these two points.

    Args:
        x (array-like): The x-dataset.
        y (array-like): The y-dataset.

    Returns:
        tuple: A tuple containing:
            - y1 (float): Max y value in the subset.
            - y2 (float): Min y value in the subset.
            - x1 (float): x value corresponding to max y.
            - x2 (float): x value corresponding to min y.
            - x_diff (float): Difference between x1 and x2.
            - y_diff (float): Difference between y1 and y2.
            - (x_mid, y_mid) (tuple): Midpoint coordinates between (x1, y1) and (x2, y2).
    """
    # Range
    start = int(len(x)*(1/3))
    end = int(len(x)*(1/2))
    # Extract data in the specified range
    x_subset = x[start:end]
    y_subset = y[start:end]

    # Find max and min y values and their corresponding x values
    y1 = np.max(y_subset)
    y2 = np.min(y_subset)

    x1 = x_subset[np.argmax(y_subset)]
    x2 = x_subset[np.argmin(y_subset)]

    # Calculate differences
    x_diff = x1 - x2
    y_diff = y1 - y2
    if delta:
      return x_diff, y_diff
    # Midpoint between (x1, y1) and (x2, y2)
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    return x_diff, y_diff, x_mid, y_mid
import numpy as np
from scipy.optimize import curve_fit

def denoise_with_half_gaussian(freq, mag, sigma):
    """
    Fit a half-Gaussian to (freq[1:], mag[1:]) with mean fixed to freq[1],
    then multiply the Gaussian with mag[1:]. Finally prepend mag[0].

    Parameters
    ----------
    freq : array-like
        Frequency values
    mag : array-like
        Magnitude values

    Returns
    -------
    denoised_mag : np.ndarray
        Denoised magnitude array
    """
    x = np.array(freq[1:])
    y = np.array(mag[1:])
    # mu fixed from first 10 x
    mu = x[0]

    # Half-Gaussian model
    def half_gaussian(x, sigma):
        return np.exp(-(x - mu)**2 / (2 * sigma**2))

    # # Initial parameter guesses
    # p0 = [(x[-1] - x[0]) / 2]

    # # Fit Gaussian
    # params, _ = curve_fit(half_gaussian, x, y, p0=p0)
    # sigma_fit = params

    # Evaluate Gaussian at x
    gaussian_vals = half_gaussian(x, sigma)

    # Multiply Gaussian with mag[1:]
    denoised_part = gaussian_vals  * y

    # Combine with original mag[0]
    denoised_mag = np.concatenate(([mag[0]], denoised_part))

    return denoised_mag
def load_material_data(path):
    """
    Constructs a file path and reads a CSV file into a pandas DataFrame.

    Args:
        base_path (str): The base directory path.
        material_name (str): The name of the material subfolder.
        filename (str): The name of the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the data, or None if an error occurs.
    """
    file_path = os.path.join(path)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {path}")
        return None

    try:
        # Assuming the CSV file has no header and two columns named 'ca' and 'oa'
        df = pd.read_csv(file_path, header=None, usecols=[0, 1], names=['ca', 'oa'])
        df["T"] = df['ca']/df['oa']
        # df['T'] = df['T']/np.mean(df['T'])
        print(f"{path[-13:]}")
        return df
    except Exception as e:
        print(f"Error reading the file {path}: {e}")
        return df

def plot_denoised_signal(plot, plot_fft, material, power, original_signal, sampling_rate, sigma, cutoff_frequency=None):
    """
    Applies a low-pass filter using DFT/FFT and plots the DFT results,
    their logarithms, and the original and denoised signals.

    Args:
        cutoff_frequency (float): The cutoff frequency for the low-pass filter (in Hz).
        original_signal (pd.Series): The original signal data (e.g., df['norm']).
        sampling_rate (float): The sampling rate of the signal (in Hz).
    """
    # Calculate the Discrete Fourier Transform (DFT)
    dft_result = np.fft.rfft(original_signal)
    frequencies = np.fft.rfftfreq(len(original_signal), 1/sampling_rate)

    # Calculate the magnitude of the DFT
    dft_magnitude = np.abs(dft_result)

    # Preserve phase
    phase = np.angle(dft_result)

    #denoise magnitudes

    new_magnitudes = denoise_with_half_gaussian(frequencies, dft_magnitude, sigma)

    filtered_dft_result = new_magnitudes * np.exp(1j * phase)

    if plot_fft:
      # Plot the magnitude spectrum
      plt.figure(figsize=(12, 6))
      plt.plot(frequencies[1:600], dft_magnitude[1:600], label="Original")

      plt.plot(frequencies[1:600], new_magnitudes[1:600], label =f"Gaussian (Sigma: {sigma}")
      plt.xlabel('Frequency (Hz)')
      plt.ylabel('Magnitude')
      plt.title(f'DFT Magnitude Spectrum\n(excluding initial high amplitude) {material} ({str(power)})')
      plt.legend()
      plt.grid(True)
      # plt.show()

    # Create a filter mask
    # Frequencies above the cutoff will be set to zero


    # Apply the filter mask to the DFT result
    if cutoff_frequency == 0:
      filter_mask = frequencies <= cutoff_frequency
      filtered_dft_result = filtered_dft_result * filter_mask
      return np.fft.irfft(filtered_dft_result,  n=len(original_signal))[0]
    # filtered_dft_result = filtered_dft_result * filter_mask

    # Perform the Inverse Discrete Fourier Transform (IDFT) to get the denoised signal
    denoised_signal = np.fft.irfft(filtered_dft_result,  n=len(original_signal))
    z = np.arange(len(denoised_signal))*1e-05
    params_of_z = analyze_data_subset(z, denoised_signal)
    z = z - params_of_z[2]
    denoised_signal = denoised_signal - params_of_z[3]+1
    # print(denoised_signal)
    # Plot the original and denoised signals
    popt0, _ = curve_fit(z_formulae,z, original_signal)
    popt1, _ = curve_fit(z_formulae,z, denoised_signal)
    raw_del_pv = analyze_data_subset(z, original_signal, delta=True)
    raw_fit_del_pv = analyze_data_subset(z, z_formulae(z, popt0[0]), delta=True)
    filter_fit_del_pv = analyze_data_subset(z, z_formulae(z, popt1[0]), delta=True)

    if plot:
      plt.figure(figsize=(12, 6))
      plt.scatter(z,original_signal,s=5 ,label='Original Signal', alpha=0.7)
      plt.plot(z, z_formulae(z, popt0[0]), label = f"Raw Fit (Phi: {popt0[0]}")
      plt.plot(z, z_formulae(z, popt1[0]), label = f"Denoised Fit (Phi: {popt1[0]}")
      plt.plot(z,denoised_signal, color = 'red',label=f'Denoised Signal (Sigma: {sigma})')
      plt.xlabel('Sample Index')
      plt.ylabel('Normalized Value')
      plt.title(f'Original vs. Denoised Signal (Low-Pass Filter) \n{material} ({str(power)})')
      plt.legend()
      plt.grid(True)
      plt.show()

    return denoised_signal, popt0[0], popt1[0], params_of_z[0], params_of_z[1], raw_del_pv[0], raw_del_pv[1], raw_fit_del_pv[0], raw_fit_del_pv[1], filter_fit_del_pv[0], filter_fit_del_pv[1]
def fft_filtering(path, material, power, plot_fft=False, plot=True, sigma=1):
  df = load_material_data(path)
  baseline = plot_denoised_signal(plot = False, plot_fft = False, material = material, power = str(power), cutoff_frequency = 0, original_signal=df['T'], sampling_rate=len(df)/22, sigma=sigma)
  df['T'] = df['T']/baseline
  final = plot_denoised_signal(plot = plot, plot_fft = plot_fft, material = material, power = str(power), original_signal=df['T'], sampling_rate=len(df)/22, sigma = sigma)
  return final



# --- terminal interface ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FFT filtering")
    
    # required parameters
    parser.add_argument("--path", type=str, required=True, help="Path for data")
    parser.add_argument("--material", type=str, required=True, help="Material name")
    parser.add_argument("--power", type=int, required=True, help="Power value")

    # optional parameters
    parser.add_argument("--plot_fft", action="store_true", help="Whether to plot FFT (flag, default=False)")
    parser.add_argument("--plot", action="store_false", help="Disable plotting (flag, default=True)")
    parser.add_argument("--sigma", type=float, default=1, help="Sigma value (default=1)")

    args = parser.parse_args()

    fft_filtering(
        path=args.path,
        material=args.material,
        power=args.power,
        plot_fft=args.plot_fft,
        plot=args.plot,
        sigma=args.sigma
    )
