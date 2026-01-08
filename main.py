#!/usr/bin/env python3
"""
zscan_cleanup.py

Algorithm for cleaning up Z-scan analysis.

Steps implemented:
1. Data loading (uses df['T'] as original signal)
2. Baseline normalization using a low-pass (cutoff) via rFFT
3. Denoising magnitudes by multiplying a half-Gaussian on the positive-frequency magnitudes
   while preserving phase, then inverse rFFT to get the denoised signal
4. SNR estimation (signal power vs noise power)
5. Fit theoretical Z-scan formula to denoised data (fit parameters phi and a; a bounded [0,0.01])
6. Plot original, denoised, and fitted signals in one figure and save

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple

# -----------------------------
# Utilities / helpers
# -----------------------------

def load_material_data(base_path: str, material_name: str, filename: str) -> pd.DataFrame:
    """Constructs a file path and reads a CSV file into a pandas DataFrame.
    Assumes CSV has no header and the first two columns are the ones we need.
    """
    material_path = os.path.join(base_path, material_name)
    file_path = os.path.join(material_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path, header=None, usecols=[0, 1], names=['ca', 'oa'])
    # df['T'] is defined as requested
    df['T'] = df['ca'] / df['oa']
    return df


# -----------------------------
# Normalization (baseline) step
# -----------------------------

def compute_baseline_and_normalize(original_signal: np.ndarray,
                                   sampling_rate: float,
                                   cutoff_frequency: float) -> Tuple[np.ndarray, float]:
    """
    Compute a baseline by low-pass filtering the rFFT (hard cutoff) and
    dividing the original signal by the baseline value (value at index 0 of the IFFT of filtered spectrum).

    Returns (normalized_signal, baseline_value)
    """
    N = len(original_signal)
    # rFFT of signal
    dft = np.fft.rfft(original_signal)
    freqs = np.fft.rfftfreq(N, 1.0 / sampling_rate)

    # hard low-pass mask
    mask = freqs <= cutoff_frequency

    filtered_dft = dft * mask
    baseline_time = np.fft.irfft(filtered_dft, n=N)
    baseline_value = baseline_time[0]
    if baseline_value == 0:
        raise ValueError("Baseline computed as zero; cannot normalize. Check cutoff_frequency or input signal.")

    normalized = original_signal / baseline_value
    return normalized, baseline_value


# -----------------------------
# Half-Gaussian denoising of magnitudes
# -----------------------------

def denoise_with_half_gaussian(freq: np.ndarray, mag: np.ndarray, sigma: float) -> np.ndarray:
    """
    Multiply mag[3:] by a half-Gaussian centered at mu = freq[3] with provided sigma.
    Prepend original mag[:3] unchanged (as requested in the algorithm description).

    Note: this function operates only on the positive-frequency magnitudes (as returned by rfft).
    """
    mag = np.asarray(mag)
    freq = np.asarray(freq)
    if len(freq) < 4:
        # nothing to denoise
        return mag.copy()

    x = freq[3:]
    y = mag[3:]
    mu = x[0]

    # half-Gaussian (same shape but centered at mu)
    gaussian_vals = np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))

    denoised_part = gaussian_vals * y
    denoised_mag = np.concatenate((mag[:3], denoised_part))
    return denoised_mag


# -----------------------------
# Full denoising pipeline
# -----------------------------

def denoise_signal(original_signal: np.ndarray, sampling_rate: float, sigma: float) -> Tuple[np.ndarray, dict]:
    """
    Denoise original_signal by:
      - computing rfft -> magnitudes and phase
      - denoising magnitudes with half-Gaussian
      - reconstructing complex spectrum and irfft

    Returns: denoised_signal, metadata dict (contains freqs, orig_mag, new_mag, phase, SNR, etc.)
    """
    N = len(original_signal)
    dft = np.fft.rfft(original_signal)
    freqs = np.fft.rfftfreq(N, 1.0 / sampling_rate)

    orig_mag = np.abs(dft)
    phase = np.angle(dft)

    new_mag = denoise_with_half_gaussian(freqs, orig_mag, sigma=sigma)

    # Avoid division by zero when orig_mag is zero --- if zero, set ratio to zero
    # Reconstruct complex spectrum
    filtered_dft = new_mag * np.exp(1j * phase)

    denoised_time = np.fft.irfft(filtered_dft, n=N)

    # Estimate noise as residual (time domain)
    residual_time = original_signal - denoised_time

    P_signal = np.mean(denoised_time ** 2)
    P_noise = np.mean(residual_time ** 2)
    snr_linear = np.inf if P_noise == 0 else P_signal / P_noise
    snr_db = np.inf if P_noise == 0 else 10.0 * np.log10(snr_linear)

    meta = {
        'freqs': freqs,
        'orig_mag': orig_mag,
        'new_mag': new_mag,
        'phase': phase,
        'filtered_dft': filtered_dft,
        'P_signal': P_signal,
        'P_noise': P_noise,
        'snr_linear': snr_linear,
        'snr_db': snr_db,
    }

    return denoised_time, meta


# -----------------------------
# Theoretical z-scan model
# -----------------------------

# Constants (from the user)
w0 = 1.81e-5
lmda = 665e-9
z0 = np.pi * w0 * w0 / lmda


def z_formulae(z: np.ndarray, phi: float, a: float) -> np.ndarray:
    """Models the normalized closed aperture signal as a function of distance z."""

    x = (z + a) / z0
    return 1.0 + phi * 4.0 * x / ((1.0 + x ** 2) * (9.0 + x ** 2))


# -----------------------------
# Fit routine
# -----------------------------

def fit_zscan(z: np.ndarray, signal: np.ndarray, a_bounds=(0.0, 0.01)) -> Tuple[np.ndarray, dict]:
    """
    Fit z_formulae to (z, signal). Returns fitted_curve and fit_info dict.
    Bounds: a in [a_bounds[0], a_bounds[1]], phi unbounded (but we give a wide bound).
    """
    # reasonable initial guesses
    phi0 = -0.02
    a0 = 0.006

    lower = [-np.inf, a_bounds[0]]
    upper = [np.inf, a_bounds[1]+0.005]

    # Fit only the middle part of the data
    fit_slice = slice(2000, 8000)
    z_fit = z[fit_slice]
    signal_fit = signal[fit_slice]

    try:
        popt, pcov = curve_fit(z_formulae, z_fit, signal_fit, p0=[phi0, a0], bounds=(lower, upper), maxfev=20000)
        phi_fit, a_fit = popt
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        # fallback: return NaNs
        phi_fit, a_fit = np.nan, np.nan
        perr = [np.nan, np.nan]

    # Shift z by fitted a so that shape is centered at z=0
    z_shifted = z + a_fit
    fitted_curve = z_formulae(z_shifted, phi_fit, a_fit)

    fit_info = {
        'phi': phi_fit,
        'a': a_fit,
        'phi_err': perr[0] if not np.isnan(perr[0]) else np.nan,
        'a_err': perr[1] if not np.isnan(perr[1]) else np.nan,
    }

    return fitted_curve, fit_info



# -----------------------------
# Plotting
# -----------------------------

# def plot_results(z: np.ndarray, shift: float, original: np.ndarray, denoised: np.ndarray, fitted: np.ndarray, outpath: str = 'zscan_fit.png') -> None:
#     plt.figure(figsize=(12, 6))
#     # scatter original (transparent)
#     plt.scatter(z + shift, original, s=5, alpha=0.7, label='Original (normalized)')

#     # denoised as line
#     plt.plot(z + shift, denoised, color = 'red', label='Denoised (half-gaussian)')

#     # fitted as dashed line
#     plt.plot(z + 2*shift, fitted, color = 'C1', label='Fitted model')
#     plt.xlim(-0.04,0.04)
#     plt.xlabel('z (m)')
#     plt.ylabel('Normalized signal (arb. units)')
#     plt.title(f'Z-scan: original, denoised and fitted for ')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     # plt.savefig("/content/drive/MyDrive/fft results/zscan_fit.png", dpi=300)
#     plt.close()


# -----------------------------
# Main pipeline
# -----------------------------

def process_zscan_dataframe(df: pd.DataFrame,
                             sampling_rate: float = 1.0,
                             baseline_cutoff: float = 0.5,
                             sigma: float = 0.1,
                             z_start: float = -6000,
                             z_stop: float = 6000,
                             z_scale: float = 1e-5,
                             fit_a_bounds=(0.0, 0.01)) -> dict:
    """
    Full pipeline. Expects df['T'] to be the original raw signal (as described by user).

    Parameters:
      df: pandas DataFrame with column 'T'
      sampling_rate: sampling rate used to compute FFT frequency axis (units inverse of z sample spacing)
      baseline_cutoff: cutoff frequency (Hz or spatial frequency) used to compute baseline (for normalization)
      sigma: sigma for the half-Gaussian denoising (in frequency units)

    Returns a dictionary with results and metadata.
    """
    original = df['T'].to_numpy(dtype=float)
    N = len(original)

    # Step 1: Baseline normalization
    normalized, baseline_value = compute_baseline_and_normalize(original, sampling_rate, baseline_cutoff)

    # normalized becomes the "original" for further processing
    original_for_filter = normalized

    # Step 2: Denoise
    denoised_signal, meta = denoise_signal(original_for_filter, sampling_rate, sigma)

    # Step 3: Build z axis
    z = np.linspace(z_start, z_stop, N) * z_scale
    # plt.plot(z, normalized)
    # plt.show()
    # Step 4: Fit
    fitted_curve, fit_info = fit_zscan(z, denoised_signal, a_bounds=fit_a_bounds)

    # Step 5: Plot
    # plot_results(z ,fit_info['a'], original_for_filter, denoised_signal, fitted_curve, outpath='zscan_comparison.png')

    results = {
        'z': z,
        'original_normalized': original_for_filter,
        'denoised': denoised_signal,
        'fitted': fitted_curve,
        'fit_info': fit_info,
        'meta': meta,
        'baseline_value': baseline_value,
        'output_plot': 'zscan_comparison.png'
    }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="FFT filtering for Z-scan data")

    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Parent directory containing material subfolders"
    )

    parser.add_argument(
        "--materials",
        type=str,
        nargs="+",
        required=True,
        help="One or more material names (space-separated)"
    )

    args = parser.parse_args()

    base_path = args.base_path
    materials = args.materials

    for material_name in materials:
      sampling_rate = 12000 / 22
      baseline_cutoff = 0
      sigma = 1
      z_start = -6000.0
      z_stop = 6000.0
      z_scale = 1e-5
      fit_a_bounds = (0.0, 0.007)

      output_path = os.path.join(base_path, "fft_data", material_name)

      # Create directory if it does not exist
      os.makedirs(output_path, exist_ok=True)

      # --- Power range ---
      power_values = list(range(185, 255))
      #snr_values_db = []
      #snr_linear_values = []
      phi_values = []

      # --- PMW function ---
      def pmw(power):
          return -1467 + (12.6 * power) - 0.0242 * (power)**2

      # --- Loop over power values ---
      for power in power_values:
          filename = f"{power}_ca_oa.csv"
          print(f"\n⚙️ Processing power = {power} mW")

          try:
            # --- Load data ---
            df = load_material_data(base_path, material_name, filename)

            # --- Run Z-scan pipeline ---
            results = process_zscan_dataframe(
                df,
                sampling_rate=sampling_rate,
                baseline_cutoff=baseline_cutoff,
                sigma=sigma,
                z_start=z_start,
                z_stop=z_stop,
                z_scale=z_scale,
                fit_a_bounds=fit_a_bounds
            )

            def compute_fit_score(y_true, y_pred):
                """
                Compute normalized RMSE-based R score:
                R = 1 - RMSE(y_pred, y_true) / std(y_true)
                Closer to 1 means better fit.
                """
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                std_true = np.std(y_true)
                if std_true == 0:
                    return np.nan
                return 1 - rmse / std_true


            # --- extract arrays as before ---
            z = results['z']
            original = results['original_normalized']
            denoised = results['denoised']
            fitted = results['fitted']
            shift = results['fit_info']['a']
            print(shift, power, material_name)
            # --- Adjust z for each dataset ---
            z_original = z + shift
            z_filtered = z + shift
            z_fitted = z + 2 * shift

            # --- Save individual CSVs ---
            df_original = pd.DataFrame({'z': z_original, 'T': original})
            df_filtered = pd.DataFrame({'z': z_filtered, 'T': denoised})
            df_fitted   = pd.DataFrame({'z': z_fitted,   'T': fitted})
            base_name = f"{power}_{material_name}"

            original_path = os.path.join(output_path, f"{base_name}_original_normalized.csv")
            filtered_path = os.path.join(output_path, f"{base_name}_filtered.csv")
            fitted_path   = os.path.join(output_path, f"{base_name}_fitted.csv")

            df_original.to_csv(original_path, index=False)
            df_filtered.to_csv(filtered_path, index=False)
            df_fitted.to_csv(fitted_path, index=False)


            # --- Identify peak & valley from fitted curve ---
            z_peak = z_fitted[np.argmax(fitted)]
            z_valley = z_fitted[np.argmin(fitted)]

            z_low = min(z_peak, z_valley)
            z_high = max(z_peak, z_valley)

            # --- Expand range by 10% ---
            z_range = z_high - z_low
            z_low_ext = z_low - 0.1 * z_range
            z_high_ext = z_high + 0.1 * z_range

            # --- Select indices within expanded range ---
            mask = (z_original >= z_low_ext) & (z_original <= z_high_ext)

            # --- Compute fit scores (R values) ---
            filter_R = compute_fit_score(original[mask], denoised[mask])
            fitted_R = compute_fit_score(original[mask], fitted[mask])

            # --- Meta information ---
            #snr_db = results["meta"].get("snr_db", np.nan)
            #snr_linear = results["meta"].get("snr_linear", np.nan)
            phi_val = results["fit_info"].get("phi", np.nan)

            meta_data = {
                # 'snr_db': [snr_db],
                # 'snr_linear': [snr_linear],
                'phi': [phi_val],
                'filter_R': [filter_R],
                'fitted_R': [fitted_R]
            }

            df_meta = pd.DataFrame(meta_data)
            df_meta.to_csv(f"{output_path}/{base_name}_meta.csv", index=False)

            print(f"✅ Saved CSVs for {base_name}")
            print(f"   → Original, Filtered, Fitted, and Meta files in {output_path}")
            print(f"   → z-range used for R: {z_low_ext:.5f} to {z_high_ext:.5f}")
            print(f"   → filter_R={filter_R:.4f}, fitted_R={fitted_R:.4f}")


            # --- Build output plot path ---
            plot_path = os.path.join(output_path, f"{base_name}_plot.pdf")

            # --- Plot ---
            plt.figure(figsize=(8, 5))

            # Original (scatter)
            plt.scatter(
                df_original['z'],
                df_original['T'],
                s=5,
                alpha=0.7,
                label='Original'
            )

            # Filtered (line)
            plt.plot(
                df_filtered['z'],
                df_filtered['T'],
                color='red',
                linewidth=1.4,
                label='Filtered'
            )

            # Fitted (dashed line)
            plt.plot(
                df_fitted['z'],
                df_fitted['T'],
                color='C1',
                linestyle='--',
                linewidth=1.2,
                label='Fitted'
            )

            plt.title(f"{material_name} | pmw = {pmw(int(power)):.2f} mW | Power = {power}")
            plt.xlabel("z (m)")
            plt.ylabel("T (a.u.)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()

            # --- Save and close ---
            plt.savefig(plot_path)
            plt.close()
            print(f"💾 Saved plot → {output_path}")

          except Exception as e:
            print(f"❌ Failed for {filename}: {e}")
            #snr_values_db.append(np.nan)
            #snr_linear_values.append(np.nan)
