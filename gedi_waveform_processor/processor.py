import numpy as np
import pandas as pd
import h5py
import ast
import matplotlib.pyplot as plt
from scipy.signal import resample, resample_poly
from .utils import decode_waveform, resample_waveform, normalize_waveform, process_waveform

class GediWaveformProcessor:
    """
    Class for processing GEDI L1B waveforms for use in ML and DL pipelines.
    Designed to work with .geojson output from NASA's GEDI_Subsetter tool.
    """
    def __init__(self, gdf):
        """
        Initialize the processor with a GeoDataFrame containing GEDI L1B shots.

        Parameters:
            gdf (GeoDataFrame): Input GeoDataFrame with GEDI shots, metadata, and rxwaveform data.
            The filtered_gdf attribute is assigned when .filter_shots() is run.
        """
        self.gdf = gdf.copy()
        self.filtered_gdf = None # stores filtered gdf
        self.processed_gdf = None # stores filtered gdf with processed waveform column

    def filter_shots(self, max_elevation_diff=300):
        """
        Filter GEDI shots using quality flags and user determined elevation difference tolerance. 

        Parameters:
            max_elevation_diff (float): Maximum allowed difference between GEDI elevation_lastbin and DEM (meters). 

        Returns:
            GeoDataFrame: Filtered GeoDataFrame. Populates the filtered_gdf attribute.
        """
        self.filtered_gdf = self.gdf[
            (self.gdf['stale_return_flag'] == 0) &
            (self.gdf['geolocation_degrade'] == 0) &
            ((self.gdf['geolocation_elevation_lastbin'] - self.gdf['geolocation_digital_elevation_model']) <= max_elevation_diff)
        ]
        return self.filtered_gdf

    def _get_active_df(self):
        """
        Helper function which returns the GeoDataFrame with the highest level of processing.

        Parameters:


        Returns: 
            the most processed GeoDataFrame available.
        """
        if self.processed_gdf is not None:
            return self.processed_gdf
        elif self.filtered_gdf is not None:
            return self.filtered_gdf
        elif self.gdf is not None:
            return self.gdf
        else:
            raise ValueError("No GeoDataFrame is available, please initialize GediWaveformProcessor")

    def _get_waveform_and_label(self, row):
        """
        Helper function which creates labels for waveform plots.

        Parameters:
            row (pd.Series): A row from the active GeoDataFrame which represents a GEDI shot. 

        Returns:
             waveform (list) and label ('Processed' or 'Raw')
        """
        if 'rxwaveform_pro' in row.index and isinstance(row['rxwaveform_pro'], (list, np.ndarray)) and row['rxwaveform_pro'] is not None:
            return row['rxwaveform_pro'], 'Processed'
        else:
            return decode_waveform(row['rxwaveform']), 'Raw'

    def plot_waveform(self, shot_number_x):
        """
        Plot a single GEDI waveform for a given shot_number_x.

        If a processed waveform (`rxwaveform_pro`) is available, it is used for plotting;
        otherwise, the raw waveform (`rxwaveform`) is decoded and plotted.

        Parameters:
            shot_number_x (int): Unique GEDI shot number identifier.

        Raises:
            ValueError: If the shot number is not found or the waveform is missing/empty.
        """
        df = self._get_active_df()

        row = df[df['shot_number_x'] == shot_number_x]
        if row.empty:
            raise ValueError(f"Shot {shot_number_x} not found.")

        row = row.iloc[0]
        wf, label = self._get_waveform_and_label(row)

        if wf is None or len(wf) == 0:
            raise ValueError(f"Waveform for shot {shot_number_x} is missing or empty.")

        plt.plot(wf)
        plt.title(f"{label} Waveform for Shot {shot_number_x}")
        plt.xlabel("Bin")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def plot_random_waveforms(self, n=5):
        """
        Plot a random sample of GEDI waveforms from the most processed available GeoDataFrame.

        Plots either all processed or all raw waveforms, depending on the first valid waveform
        encountered. Mixed types are not displayed together.

        Parameters:
            n (int): Number of waveforms to plot (default is 5).

        Raises:
            ValueError: If no data is available to sample from.
        """
        df = self._get_active_df()

        if len(df) == 0:
            raise ValueError("No data available to plot.")

        samples = df.sample(min(n, len(df)))

        label = None  # We'll update this as we go

        for _, row in samples.iterrows():
            wf, this_label = self._get_waveform_and_label(row)

            # Only plot waveforms of the same type
            if label is None:
                label = this_label
            if this_label != label:
                continue  # Skip rows with the other type

            if wf is not None and len(wf) > 0:
                plt.plot(wf, alpha=0.7)
            else:
                print(f"⚠️ Skipping empty or invalid waveform for shot {row.get('shot_number_x', 'unknown')}")

        plt.title(f"{min(n, len(samples))} Random {label} GEDI Waveforms")
        plt.xlabel("Bin")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()


    def process_all_waveforms(self, target_length=None, resample_method=None, norm_method=None):
        """
        Process all waveforms by decoding, optionally resampling and normalizing them.

        Uses `filtered_gdf` if available, otherwise falls back to `gdf`.
        Creates the column `rxwaveform_pro` to store processed waveforms and saved gdf to `self.processed_gdf`.

        Parameters:
            target_length (int, optional): Length to resample waveforms to.
            resample_method (str, optional): Resampling method ('fft', 'poly', 'pad'). -- poly not looking great at the moment
            norm_method (str, optional): Normalization method ('zscore', 'minmax', or None).

        Returns:
            GeoDataFrame: Copy of the input with processed waveforms in `rxwaveform_pro`.

        Prints:
            Number of waveforms skipped due to decoding or empty input.
        """
        df = self.filtered_gdf if self.filtered_gdf is not None else self.gdf
        processed = []
        skipped = 0

        for i, row in df.iterrows():
            wf_raw = row['rxwaveform']
            wf = decode_waveform(wf_raw)

            if wf is None or len(wf) == 0:
                skipped += 1
                processed.append(None)
                continue

            if target_length and resample_method is not None:
                wf = resample_waveform(wf, target_length, resample_method)

            if norm_method is not None:
                wf = normalize_waveform(wf, norm_method)

            processed.append(wf)

        df = df.copy()
        df['rxwaveform_pro'] = processed
        df = df[df['rxwaveform_pro'].notnull()]  # Optionally drop failed ones
        self.processed_gdf = df

        print(f"Done. Skipped {skipped} invalid or malformed waveforms.")
        return df

    def export_ml_ready_dataset(self, out_path, format='npz'):
        """
        Export processed waveforms and metadata as an ML-ready dataset.

        Assumes `self.processed_gdf` has already been created using `process_all_waveforms()`.

        Parameters:
            out_path (str): Output file path (.npz, .pt, or .h5).
            format (str): 'npz' for TensorFlow, 'pt' for PyTorch, or 'h5' for HDF5.

        Saves:
            - waveforms: array or torch.Tensor of shape (n_samples, waveform_length)
            - metadata: array or torch.Tensor of shape (n_samples, 4) [lat, lon, elev0, elev_last]
            - shot_index: array of shape (n_samples,) as strings (shot_number_x)
        """
        if self.processed_gdf is None:
            raise RuntimeError("No processed_gdf found. Please run `process_all_waveforms()` first.")

        df = self.processed_gdf

        # Convert waveforms to uniform float32 array
        waveforms = np.array(df["rxwaveform_pro"].tolist(), dtype=np.float32)

        # Fast vectorized metadata extraction using np.stack
        metadata = np.stack([
            df.geometry.y.values,  # Latitude
            df.geometry.x.values,  # Longitude
            df['geolocation_elevation_bin0'].values,
            df['geolocation_elevation_lastbin'].values
        ], axis=1).astype(np.float64)

        # Preserve shot_number_x as string array (safe for long IDs)
        shot_index = np.array(df['shot_number_x'], dtype = 'U21') # let's try it this way

        # Save based on format
        if format == 'npz':
            np.savez(out_path, waveforms=waveforms, metadata=metadata, shot_index=shot_index)
            print(f"Saved TensorFlow-ready dataset to {out_path}")

        elif format == 'pt':
            data = {
                'waveforms': torch.tensor(waveforms, dtype=torch.float32),
                'metadata': torch.tensor(metadata, dtype=torch.float32),
                'shot_index': shot_index.astype('U21')  # Keep as list of strings for PyTorch compatibility
            }
            torch.save(data, out_path)
            print(f"Saved PyTorch-ready dataset to {out_path}")

        elif format == 'h5':
            with h5py.File(out_path, 'w') as f:
                f.create_dataset('waveforms', data=waveforms, compression='gzip')
                f.create_dataset('metadata', data=metadata, compression='gzip')
                f.create_dataset('shot_index', data=shot_index, dtype=dt, compression='gzip')
                dt = h5py.string_dtype(encoding='utf-8')
            print(f"Saved HDF5 dataset to {out_path}")

        else:
            raise ValueError("Format must be one of: 'npz', 'pt', or 'h5'")