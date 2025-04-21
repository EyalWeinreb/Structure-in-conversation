This folder contains Python code (`IUClust_csv.py`) for clustering Intonation Units (IUs) based on their acoustic features (pitch and intensity). It employs a pipeline involving an Autoencoder (AE) for dimensionality reduction and feature extraction, followed by a Gaussian Mixture Model (GMM) for clustering in the latent space.

It also contains clustering results for 

## Overview

The script processes IUs by:

1.  **Reading Data:** Loads IU data (pitch vectors, intensity vectors, lengths, quality flags) from an HDF5 file.
2.  **Preprocessing:**
    *   Filters out IUs marked as 'bad'.
    *   Splits IUs into subsets ('acts') based on their duration using deciles, or alternatively taking just a single user-specified duration range.
    *   Preprocesses pitch vectors within each act: interpolation, intensity-weighted rolling mean smoothing, and resampling to a fixed length (determined by the median IU length in the act and `--feats-per-sec`).
3.  **Autoencoder Training:** For each 'act' (and optionally for 'hunks' of leftover IUs):
    *   Trains an autoencoder on the preprocessed feature vectors.
    *   Optionally uses a Sliced-Wasserstein (SW) loss component during training to enforce a desired distribution (hyperspherical shell) in the latent space, aiming for more structured representations.
    *   Extracts the latent representations (encoded vectors) from the autoencoder for each IU in the current act/hunk.
4.  **GMM Clustering:**
    *   Applies a Bayesian Gaussian Mixture Model (GMM) to cluster the latent representations obtained from the autoencoder.
    *   Assigns each IU within the act/hunk to a cluster.
5.  **Handling Leftovers (Optional):**
    *   If enabled (`--leftovers`), IUs belonging to very small clusters (below `--lo-ratio` threshold) after the initial clustering of an 'act' are considered 'leftovers'.
    *   These leftovers are grouped into 'hunks' and reprocessed through the AE+GMM pipeline iteratively to attempt further clustering.
6.  **Output:**
    *   Writes the final clustering results (including original IU info) to a CSV file.
    *   Saves run parameters and metadata (including final training losses) to a YAML file.

## Input

The script requires an HDF5 file (`--input-file`, default: `Input.h5`) containing:

*   A dataset named `ius` (readable as a Pandas DataFrame) which includes at minimum the following columns:
    *   `viu_t_len`: Length of the IU vector in seconds.
    *   `good`: Boolean flag indicating if the IU is suitable for processing.
*   A dataset named `t_db_mst` (readable as a NumPy array) containing the time, intensity (dB), and pitch (median normalized semitone) vectors for each IU. Shape: `(number of IUs, number of points in the longest IU vector, 3)`.
An example input file with 500 entries is given as `example_input_format.h5`

## Output

The script generates output files in a timestamped subdirectory within the specified output directory (`--output-directory`, default: `IUClust_csv.out/<timestamp>/`):

1.  **CSV File:** `IUClust_csv.csv` - Contains all data present in the input `ius` dataset along with added columns:
    *   `act`: The length-based subset the IU belongs to (e.g., `decile_01`, `len_850ms-1230ms`).
    *   `hunk`: The specific processing group (same as `act` unless leftovers are processed, e.g., `decile_01_leftovers_1`).
    *   `cluster`: The final assigned cluster ID (e.g., `c1_10001`).
    *   `cluster_n`: Number of IUs in the assigned cluster.
    *   `cluster_population`: Total number of IUs in the cluster's originating act.
    *   `cluster_ratio`: `cluster_n / cluster_population`.
2.  **YAML File:** `metadata.yml` - Contains metadata about the run, including:
    *   Command-line arguments used.
    *   Timestamp.
    *   Final training losses for the autoencoder(s) for each hunk.

## Dependencies

*   code was run using python 3.8.10
*   package requirements are detailed in `requirements.txt`

## Usage

Run the script from the command line:
python IUClust_csv.py [OPTIONS]

**Common Options:**

*   `--input-file <path>`: Path to the input HDF5 file (default: `Input.h5`).
*   `--output-directory <path>`: Directory to save output files (default: `IUClust_csv.out/<timestamp>`).
*   `--lengths <LENGTH_STR>`: How to split IUs by length. Use `deciles` (default) or a range like `0.5-1.0`.

For a full list of options and their descriptions run:
python IUClust_csv.py --help
