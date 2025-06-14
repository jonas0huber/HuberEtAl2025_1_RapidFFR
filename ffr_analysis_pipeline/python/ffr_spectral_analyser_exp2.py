#!/usr/bin/env python
# coding: utf-8


#  9-Minute Rapid FFR ERP Analysis Script
#
# This script analyses long-duration (9-minute) rapid ERP data by chunking it into
# time segments and computing ERP responses for each chunk:
# - Signal-to-Noise Ratio (SNR) calculations
# - Envelope Following Response (EFR) analysis
# - Temporal Fine Structure (TFS) analysis
# - Bootstrap permutation testing for statistical significance
#
# This script follows the same structure as the unified iterative FFR ERP analysis
# but processes temporal chunks instead of increasing trial counts.
#
# Author: Jonas Huber

# Import libraries
import numpy as np
import pandas as pd
import concurrent.futures
import math
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ExperimentType(Enum):
    """Enum for different experiment types with their specific parameters."""
    NINE_MINUTE_RAPID = "9MinRapid"


@dataclass
class ExperimentConfig:
    """Configuration class for 9-minute rapid experiment parameters."""
    experiment_type: ExperimentType
    input_directory: str
    output_directory: str
    trial_length: int
    n_partitions: int = 6  # Number of time chunks to divide 9-minute data into
    sampling_frequency: int = 16384
    n_noise_iterations: int = 10000
    n_harmonics_of_interest: int = 7
    participants: List[int] = None
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = list(range(1, 22))  # Default: participants 1-21
        
        # Calculate frequencies of interest (harmonics of 128 Hz)
        self.frequencies_of_interest = 128 * np.arange(1, self.n_harmonics_of_interest + 1)
        
        # Set padding for frequency spectrum
        self.padding_freq_spectrum = self.trial_length


class BootstrapMethods:
    """Class containing bootstrap methods for noise and signal generation."""
    
    @staticmethod
    def bootstrap_noise(data: np.ndarray, seed: int) -> np.ndarray:
        """
        Generate bootstrapped noise by randomly sampling segments from the data.
        
        Args:
            data: Input ERP data array (n_trials x trial_length)
            seed: Random seed for reproducibility
            
        Returns:
            Bootstrapped noise data with same dimensions as input
        """
        trial_length = data.shape[1]
        n_trials = data.shape[0]
        
        # Reshape data into a single vector for random sampling
        data_vector = data.reshape(1, -1)
        max_start_index = data_vector.shape[1] - trial_length - 1
        
        # Initialize output array
        bootstrapped_noise = np.zeros((n_trials, trial_length))
        
        # Set random seed and generate random starting indices
        np.random.seed(seed)
        random_indices = np.random.randint(0, max_start_index, size=n_trials)
        
        # Extract random segments
        for i, start_idx in enumerate(random_indices):
            bootstrapped_noise[i, 0] = data_vector[0, start_idx]
            bootstrapped_noise[i, 1:-1] = data_vector[0, start_idx + 1:start_idx + trial_length - 1]
        
        return bootstrapped_noise
    

class SignalProcessor:
    """Class for signal processing operations."""
    
    @staticmethod
    def extract_spectral_values(spectrum_data: np.ndarray, frequency_vector: np.ndarray,
                              target_frequency: float, bandwidth_hz: float = 1) -> Tuple[float, int]:
        """
        Extract maximum spectral value within a frequency band around target frequency.
        
        Args:
            spectrum_data: Spectrum amplitude data
            frequency_vector: Corresponding frequency values
            target_frequency: Center frequency of interest
            bandwidth_hz: Bandwidth (±Hz) around target frequency
            
        Returns:
            Tuple of (max_value, bin_index)
        """
        freq_low = target_frequency - bandwidth_hz
        freq_high = target_frequency + bandwidth_hz
        
        # Create frequency mask
        freq_mask = (frequency_vector >= freq_low) & (frequency_vector <= freq_high)
        
        # Extract spectrum values in frequency range
        spectrum_subset = spectrum_data[freq_mask]
        
        if len(spectrum_subset) == 0:
            return 0.0, 0
        
        # Find maximum value and its bin index
        max_value = np.max(spectrum_subset)
        bin_index = np.argmax(spectrum_data == max_value)
        
        return max_value, bin_index
    
    @staticmethod
    def compute_fft_spectrum(signal: np.ndarray, sampling_freq: int,
                           padding_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT spectrum with RMS normalization.
        
        Args:
            signal: Input time-domain signal
            sampling_freq: Sampling frequency in Hz
            padding_length: Zero-padding length for FFT
            
        Returns:
            Tuple of (spectrum_amplitude, frequency_vector)
        """
        n_samples = len(signal)
        
        # Compute FFT with zero-padding and normalize to RMS
        fft_result = np.fft.fft(signal, n=padding_length)
        spectrum_amplitude = 2 * np.abs(fft_result / n_samples) / np.sqrt(2)
        
        # Create frequency vector
        frequency_vector = np.arange(len(spectrum_amplitude)) * sampling_freq / len(spectrum_amplitude)
        
        return spectrum_amplitude, frequency_vector


class ChunkedERPAnalyzer:
    """Main class for 9-minute chunked ERP analysis."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.bootstrap = BootstrapMethods()
        self.processor = SignalProcessor()
        
        # Initialize result arrays
        self._initialize_arrays()
    
    def _initialize_arrays(self):
        """Initialize all numpy arrays for storing results."""
        n_partitions = self.config.n_partitions
        n_noise_iter = self.config.n_noise_iterations
        trial_len = self.config.trial_length
        padding_len = self.config.padding_freq_spectrum
        n_harmonics = self.config.n_harmonics_of_interest
        
        # ERP arrays
        self.erp_positive = np.zeros((n_partitions, trial_len))
        self.erp_negative = np.zeros((n_partitions, trial_len))
        
        # Signal arrays
        self.efr_signal = np.zeros((n_partitions, trial_len))
        self.tfs_signal = np.zeros((n_partitions, trial_len))
        
        # Spectrum arrays
        self.efr_spectrum = np.zeros((n_partitions, 2, padding_len))
        self.tfs_spectrum = np.zeros((n_partitions, 2, padding_len))
        
        # Noise arrays
        self.noise_erp_pos = np.zeros((n_partitions, n_noise_iter, trial_len))
        self.noise_erp_neg = np.zeros((n_partitions, n_noise_iter, trial_len))
        self.noise_efr_iterations = np.zeros((n_partitions, n_noise_iter, trial_len))
        self.noise_tfs_iterations = np.zeros((n_partitions, n_noise_iter, trial_len))
        self.noise_efr_spectrum_iterations = np.zeros((n_partitions, n_noise_iter, padding_len))
        self.noise_tfs_spectrum_iterations = np.zeros((n_partitions, n_noise_iter, padding_len))
        
        # Summary noise arrays
        self.noise_efr_spectrum = np.zeros((n_partitions, 3, padding_len))
        self.noise_tfs_spectrum = np.zeros((n_partitions, 3, padding_len))
        
        # Results arrays
        self.p_values = np.zeros((n_partitions, n_harmonics))
        self.snr_full_spectrum = np.zeros((1, n_partitions))
        self.snr_harmonic_spectrum = np.zeros((1, n_partitions))
        
        # Export arrays
        self.snr_harmonics_all = np.zeros((n_partitions, 3 + n_harmonics * 2))
        self.signal_and_noise = np.zeros((n_partitions, 1 + n_harmonics * 2))
        self.snr_harmonics_efr = np.zeros((n_partitions, 1 + n_harmonics * 3))
        self.snr_harmonics_tfs = np.zeros((n_partitions, 1 + n_harmonics * 3))
        
        # Working arrays for harmonic analysis
        self.signal_harmonics_efr = np.zeros((1, n_harmonics))
        self.signal_harmonics_tfs = np.zeros((1, n_harmonics))
        self.bin_signal_efr = np.zeros((1, n_harmonics), dtype=int)
        self.bin_signal_tfs = np.zeros((1, n_harmonics), dtype=int)
        self.noise_harmonics_efr = np.zeros((1, n_harmonics))
        self.noise_harmonics_tfs = np.zeros((1, n_harmonics))
        self.snr_harmonics_efr_vals = np.zeros((1, n_harmonics))
        self.snr_harmonics_tfs_vals = np.zeros((1, n_harmonics))
        self.spectrum_signal_harmonics_rms = np.zeros((1, n_harmonics))
        self.spectrum_noise_harmonics_rms = np.zeros((1, n_harmonics))
    
    def _load_data(self, participant_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load 9-minute ERP data for a specific participant.
        
        Args:
            participant_id: Participant identifier
            
        Returns:
            Tuple of (positive_polarity_data, negative_polarity_data)
        """
        pos_filename = f"TimLongRapid_Part{participant_id}posPol.csv"
        neg_filename = f"TimLongRapid_Part{participant_id}negPol.csv"
        
        
        pos_path = Path(self.config.input_directory) / pos_filename
        neg_path = Path(self.config.input_directory) / neg_filename
        
        positive_data = pd.read_csv(pos_path, header=None).to_numpy()
        negative_data = pd.read_csv(neg_path, header=None).to_numpy()
        
        return positive_data, negative_data
    
    def _compute_erp_signals(self, pos_data: np.ndarray, neg_data: np.ndarray, partition: int):
        """
        Compute ERP signals (EFR and TFS) for given partition.
        
        Args:
            pos_data: Positive polarity ERP data chunk
            neg_data: Negative polarity ERP data chunk
            partition: Current partition index
        """
        # Compute average ERPs
        self.erp_positive[partition, :] = np.mean(pos_data, axis=0)
        self.erp_negative[partition, :] = np.mean(neg_data, axis=0)
        
        # Compute EFR (Envelope Following Response) - average of both polarities
        self.efr_signal[partition, :] = (self.erp_positive[partition, :] +
                                        self.erp_negative[partition, :]) / 2
        
        # Compute TFS (Temporal Fine Structure) - difference between polarities
        self.tfs_signal[partition, :] = (self.erp_positive[partition, :] -
                                        self.erp_negative[partition, :]) * 2
        
        # Compute spectra
        efr_spectrum, efr_freqs = self.processor.compute_fft_spectrum(
            self.efr_signal[partition, :],
            self.config.sampling_frequency,
            self.config.padding_freq_spectrum
        )
        
        tfs_spectrum, tfs_freqs = self.processor.compute_fft_spectrum(
            self.tfs_signal[partition, :],
            self.config.sampling_frequency,
            self.config.padding_freq_spectrum
        )
        
        self.efr_spectrum[partition, 0, :] = efr_spectrum
        self.efr_spectrum[partition, 1, :] = efr_freqs
        self.tfs_spectrum[partition, 0, :] = tfs_spectrum
        self.tfs_spectrum[partition, 1, :] = tfs_freqs
    
    def _compute_noise_estimates(self, pos_data: np.ndarray, neg_data: np.ndarray, partition: int):
        """
        Compute noise estimates using bootstrap permutation.
        
        Args:
            pos_data: Positive polarity ERP data chunk
            neg_data: Negative polarity ERP data chunk
            partition: Current partition index
        """
        for noise_iter in range(self.config.n_noise_iterations):
            # Generate bootstrap noise
            noise_pos = self.bootstrap.bootstrap_noise(pos_data, noise_iter)
            noise_neg = self.bootstrap.bootstrap_noise(neg_data, noise_iter)
            
            # Compute noise ERPs
            self.noise_erp_pos[partition, noise_iter, :] = np.mean(noise_pos, axis=0)
            self.noise_erp_neg[partition, noise_iter, :] = np.mean(noise_neg, axis=0)
            
            # Compute noise EFR and TFS
            self.noise_efr_iterations[partition, noise_iter, :] = (
                self.noise_erp_pos[partition, noise_iter, :] +
                self.noise_erp_neg[partition, noise_iter, :]
            ) / 2
            
            self.noise_tfs_iterations[partition, noise_iter, :] = (
                self.noise_erp_pos[partition, noise_iter, :] -
                self.noise_erp_neg[partition, noise_iter, :]
            ) * 2
            
            # Compute noise spectra
            efr_noise_spectrum, _ = self.processor.compute_fft_spectrum(
                self.noise_efr_iterations[partition, noise_iter, :],
                self.config.sampling_frequency,
                self.config.padding_freq_spectrum
            )
            
            tfs_noise_spectrum, _ = self.processor.compute_fft_spectrum(
                self.noise_tfs_iterations[partition, noise_iter, :],
                self.config.sampling_frequency,
                self.config.padding_freq_spectrum
            )
            
            self.noise_efr_spectrum_iterations[partition, noise_iter, :] = efr_noise_spectrum
            self.noise_tfs_spectrum_iterations[partition, noise_iter, :] = tfs_noise_spectrum
        
        # Compute summary statistics of noise
        self.noise_efr_spectrum[partition, 0, :] = np.median(
            self.noise_efr_spectrum_iterations[partition, :, :], axis=0
        )
        self.noise_efr_spectrum[partition, 1, :] = self.efr_spectrum[partition, 1, :]
        self.noise_efr_spectrum[partition, 2, :] = np.std(
            self.noise_efr_spectrum_iterations[partition, :, :], axis=0
        )
        
        self.noise_tfs_spectrum[partition, 0, :] = np.median(
            self.noise_tfs_spectrum_iterations[partition, :, :], axis=0
        )
        self.noise_tfs_spectrum[partition, 1, :] = self.tfs_spectrum[partition, 1, :]
        self.noise_tfs_spectrum[partition, 2, :] = np.std(
            self.noise_tfs_spectrum_iterations[partition, :, :], axis=0
        )
    
    def _compute_harmonic_analysis(self, partition: int):
        """
        Compute harmonic analysis including SNR and p-values.
        
        Args:
            partition: Current partition index
        """
        # Extract values for each harmonic
        for harmonic_idx, freq in enumerate(self.config.frequencies_of_interest):
            # Signal values
            self.signal_harmonics_efr[0, harmonic_idx], self.bin_signal_efr[0, harmonic_idx] = (
                self.processor.extract_spectral_values(
                    self.efr_spectrum[partition, 0, :],
                    self.efr_spectrum[partition, 1, :],
                    freq,
                    bandwidth_hz=1
                )
            )
            
            self.signal_harmonics_tfs[0, harmonic_idx], self.bin_signal_tfs[0, harmonic_idx] = (
                self.processor.extract_spectral_values(
                    self.tfs_spectrum[partition, 0, :],
                    self.tfs_spectrum[partition, 1, :],
                    freq,
                    bandwidth_hz=1
                )
            )
            
            # Noise values
            self.noise_harmonics_efr[0, harmonic_idx], _ = (
                self.processor.extract_spectral_values(
                    self.noise_efr_spectrum[partition, 0, :],
                    self.noise_efr_spectrum[partition, 1, :],
                    freq,
                    bandwidth_hz=1
                )
            )
            
            self.noise_harmonics_tfs[0, harmonic_idx], _ = (
                self.processor.extract_spectral_values(
                    self.noise_tfs_spectrum[partition, 0, :],
                    self.noise_tfs_spectrum[partition, 1, :],
                    freq,
                    bandwidth_hz=1
                )
            )
            
            # SNR calculations (in dB)
            if self.noise_harmonics_efr[0, harmonic_idx] > 0:
                self.snr_harmonics_efr_vals[0, harmonic_idx] = 20 * np.log10(
                    self.signal_harmonics_efr[0, harmonic_idx] / self.noise_harmonics_efr[0, harmonic_idx]
                )
            
            if self.noise_harmonics_tfs[0, harmonic_idx] > 0:
                self.snr_harmonics_tfs_vals[0, harmonic_idx] = 20 * np.log10(
                    self.signal_harmonics_tfs[0, harmonic_idx] / self.noise_harmonics_tfs[0, harmonic_idx]
                )
        
        # Compute p-values using permutation testing
        for harmonic_idx in range(3):  # First 3 harmonics use EFR
            signal_val = self.signal_harmonics_efr[0, harmonic_idx]
            noise_distribution = self.noise_efr_spectrum_iterations[
                partition, :, self.bin_signal_efr[0, harmonic_idx]
            ]
            self.p_values[partition, harmonic_idx] = (
                np.sum(signal_val < noise_distribution) / self.config.n_noise_iterations
            )
        
        for harmonic_idx in range(3, self.config.n_harmonics_of_interest):  # Remaining use TFS
            signal_val = self.signal_harmonics_tfs[0, harmonic_idx]
            noise_distribution = self.noise_tfs_spectrum_iterations[
                partition, :, self.bin_signal_tfs[0, harmonic_idx]
            ]
            self.p_values[partition, harmonic_idx] = (
                np.sum(signal_val < noise_distribution) / self.config.n_noise_iterations
            )
        
        # Compute full-spectrum SNR (RMS-based)
        noise_waveform = np.median(self.noise_efr_iterations[partition, :, :], axis=0)
        noise_rms = np.sqrt(np.mean(noise_waveform**2))
        signal_rms = np.sqrt(np.mean(self.efr_signal[partition, :]**2))
        
        if noise_rms > 0:
            self.snr_full_spectrum[0, partition] = 20 * np.log10(signal_rms / noise_rms)
        
        # Compute harmonic-specific SNR
        self.spectrum_signal_harmonics_rms[0, :3] = self.signal_harmonics_efr[0, :3]
        self.spectrum_signal_harmonics_rms[0, 3:] = self.signal_harmonics_tfs[0, 3:]
        signal_harmonic_rms = np.sqrt(np.sum(self.spectrum_signal_harmonics_rms**2))
        
        self.spectrum_noise_harmonics_rms[0, :3] = self.noise_harmonics_efr[0, :3]
        self.spectrum_noise_harmonics_rms[0, 3:] = self.noise_harmonics_tfs[0, 3:]
        noise_harmonic_rms = np.sqrt(np.sum(self.spectrum_noise_harmonics_rms**2))
        
        if noise_harmonic_rms > 0:
            self.snr_harmonic_spectrum[0, partition] = 20 * np.log10(
                signal_harmonic_rms / noise_harmonic_rms
            )
        
        # Prepare export arrays
        self.snr_harmonics_all[partition, 0] = self.config.n_harmonics_of_interest
        self.snr_harmonics_all[partition, 1:] = np.concatenate([
            self.snr_harmonics_efr_vals[0, :3],
            self.snr_harmonics_tfs_vals[0, 3:],
            [self.snr_full_spectrum[0, partition], self.snr_harmonic_spectrum[0, partition]],
            self.p_values[partition, :]
        ])
        
        self.signal_and_noise[partition, 0] = self.config.n_harmonics_of_interest
        self.signal_and_noise[partition, 1:] = np.concatenate([
            self.signal_harmonics_efr[0, :3],
            self.signal_harmonics_tfs[0, 3:],
            self.noise_harmonics_efr[0, :3],
            self.noise_harmonics_tfs[0, 3:]
        ])
        
        self.snr_harmonics_efr[partition, 0] = self.config.n_harmonics_of_interest
        self.snr_harmonics_efr[partition, 1:] = np.concatenate([
            self.snr_harmonics_efr_vals[0, :],
            self.signal_harmonics_efr[0, :],
            self.noise_harmonics_efr[0, :]
        ])
        
        self.snr_harmonics_tfs[partition, 0] = self.config.n_harmonics_of_interest
        self.snr_harmonics_tfs[partition, 1:] = np.concatenate([
            self.snr_harmonics_tfs_vals[0, :],
            self.signal_harmonics_tfs[0, :],
            self.noise_harmonics_tfs[0, :]
        ])
    
    def analyze_participant(self, participant_id: int):
        """
        Analyze 9-minute data for a single participant by chunking into partitions.
        
        Args:
            participant_id: Participant identifier
        """
        print(f"Processing Participant {participant_id}")
        
        # Load data
        pos_data, neg_data = self._load_data(participant_id)
        
        # Calculate partition sizes
        n_trials_partition_pos = round(len(pos_data) / self.config.n_partitions)
        n_trials_partition_neg = round(len(neg_data) / self.config.n_partitions)
        
        # Process each time partition
        for partition in range(self.config.n_partitions):
            print(f"  Processing partition {partition + 1}/{self.config.n_partitions}")
            
            # Extract data for current partition
            start_pos = n_trials_partition_pos * partition
            end_pos = n_trials_partition_pos * (partition + 1)
            start_neg = n_trials_partition_neg * partition
            end_neg = n_trials_partition_neg * (partition + 1)
            
            pos_partition = pos_data[start_pos:end_pos, :]
            neg_partition = neg_data[start_neg:end_neg, :]
            
            # Compute ERP signals and spectra
            self._compute_erp_signals(pos_partition, neg_partition, partition)
            
            # Compute noise estimates
            self._compute_noise_estimates(pos_partition, neg_partition, partition)
            
            # Perform harmonic analysis
            self._compute_harmonic_analysis(partition)
        
        # Save results
        self._save_results(participant_id)
    
    def _save_results(self, participant_id: int):
        """
        Save analysis results to CSV files.
        
        Args:
            participant_id: Participant identifier
        """
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"9MinRapidERPData_Part{participant_id}"
        
        # Save main results
        np.savetxt(
            output_dir / f"{prefix}Values.csv",
            self.snr_harmonics_all,
            delimiter=","
        )
        
        np.savetxt(
            output_dir / f"{prefix}SignalNoiseValues.csv",
            self.signal_and_noise,
            delimiter=","
        )
        
        np.savetxt(
            output_dir / f"{prefix}ValuesEFR.csv",
            self.snr_harmonics_efr,
            delimiter=","
        )
        
        np.savetxt(
            output_dir / f"{prefix}ValuesTFS.csv",
            self.snr_harmonics_tfs,
            delimiter=","
        )


def run_9min_analysis(input_dir: str,
                     output_dir: str,
                     participants: Optional[List[int]] = None,
                     n_workers: Optional[int] = None):
    """
    Run the complete 9-minute rapid ERP analysis.
    
    Args:
        input_dir: Input directory containing CSV files
        output_dir: Output directory for results
        participants: List of participant IDs to analyze (default: 1-21)
        n_workers: Number of parallel workers (default: number of CPU cores)
    """
    # Set trial length for 1 cycle at 128 Hz
    sampling_freq = 16384
    f0_stimulus = 128
    cycle_length = sampling_freq // f0_stimulus
    trial_length = cycle_length  # 1 cycle = 128 samples
    
    # Create configuration
    config = ExperimentConfig(
        experiment_type=ExperimentType.NINE_MINUTE_RAPID,
        input_directory=input_dir,
        output_directory=output_dir,
        trial_length=trial_length,
        participants=participants or list(range(1, 22))
    )
    
    # Create analyzer
    analyzer = ChunkedERPAnalyzer(config)
    
    # Run analysis in parallel
    if n_workers is None:
        n_workers = min(len(config.participants), 8)  # Limit to 8 workers
    
    print(f"Starting 9-minute rapid ERP analysis with {n_workers} workers")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Participants: {config.participants}")
    print(f"Trial length: {trial_length} samples")
    print(f"Number of partitions: {config.n_partitions}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        executor.map(analyzer.analyze_participant, config.participants)
    
    print("9-minute rapid ERP analysis complete")


def main():
    """
    Main function demonstrating usage of the 9-minute rapid ERP analysis script.
    """
    # Example usage for 9-minute rapid ERP analysis
    run_9min_analysis(
        input_dir='./input/',
        output_dir='./output/',
        participants=list(range(1, 22)),  # Participants 1-21
        n_workers=3,
    )


if __name__ == '__main__':
    main()

# In[ ]:




