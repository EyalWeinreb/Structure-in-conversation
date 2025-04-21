"""
Autoencoder + GMM Pipeline for Intonation Unit Clustering

This program clusters intonation unit (IU)
It extracts latent representations of IU features using autoencoders
It then cluster these latent representations using GMM

IUs are first split into subsets ('acts') based on their lengths (either by deciles or a specified range)
These 'acts' are then processed separately, with an independent autoencoder being trained and used for each 'act'
and each 'act' being clustered separately

For each 'act', IUs that have been clustered into very small clusters are optionally considered 'leftovers', 
and are then reprocessed separately in sub-subsets ('hunks'), with an independent autoencoder being trained and used for each 'hunk'
and each 'hunk' being clustered separately

The pipeline employs a sliced-Wasserstein approach to enforce a desired distribution in the latent space,
which helps create more meaningful and structured representations

The IU feature vectors used for clustering are based on the IUs' pitch and intensity vectors
The pitch vectors first undergo interpolation and an intensity-weighted rolling mean
They are then resampled to a fixed length for all IUs per 'act'

Input - 
- HDF5 file with IU pitch and intensity vectors, lengths, and 'good' IU flags

Outputs - 
- CSV file with clustering results
- YAML metadata file with run information and parameters
"""
import logging
import argparse
import dataclasses
import itertools
import math
import datetime
import sys
import random
import pickle
import os
import warnings
from tables import NaturalNameWarning
from typing import Generator
from pathlib import Path

import h5py
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split

# set up a logger
logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclasses.dataclass
class Length:
    """ A user-defined length setting """
    as_str: str             # The string passed on the command line
    mode:   str             # 'range' or 'deciles'
    val_1:  float = 0.0     # min IU length (in seconds) for 'range' mode
    val_2:  float = 0.0     # max IU length (in seconds) for 'range' mode

class IUClust:
    def run(self):
        # suppress some warnings from tensorflow and pandas 
        suppress_some_warnings()

        # get path
        self.path = Path(__file__).resolve()

        self.parse_command_line()

        # use consistent 'random' state for reproducibility
        if self.args.rand_file:
            # self.make_and_save_random_state(self.args.rand_file)
            self.load_random_state(self.args.rand_file)
        else:
            self.rand_seed = None

        self.read_data()

        for _ in self.create_lengths():
            for self.curr_act in self.create_acts():
                for self.curr_hunk in self.create_hunks():
                    self.autoencoder()
                    self.clusterer()
        self.write_csv()

    def parse_command_line(self):
        """ Parse and check input from user
        
        sets - self.args*, self.timestamp, self.out_fn, self.metadata
        creates - output directory
        """
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_parent = self.path.parent / f"{self.path.stem}.out"

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        group = parser.add_argument_group(title="control and data parameters")
        group.add_argument("--input-file",       type=Path, default=self.path.parent/"Input.h5",  help="Input h5 file")
        group.add_argument("--output-directory", type=Path, default=output_parent/self.timestamp, help="Output directory")
        group.add_argument(
            "--lengths", 
            metavar="LENGTH_STR", 
            type=self.parse_length_arg, 
            default=[self.parse_length_arg("deciles")], 
            help="Either 'deciles' or a range of IU lengths to process (for example '0.85-1.23')"
        )            
        group.add_argument("--feats-per-sec", type=int,   default=100,  help="Resample each act's IUs to MEDIAN_LENGTH * FEATS_PER_SEC points")
        group.add_argument("--winsize",       type=int,   default=10,   help="Intensity weighted rolling mean window size")
        group.add_argument("--rand-file",     type=Path,  default=None, help="Load random state from this file")
        group.add_argument("--leftovers",     action="store_true",      help="Try to cluster leftover IUs")
        group.add_argument("--lo-ratio",      type=float, default=0.01, help="If leftovers, threshold cluster-size (as a ratio of the 'act' population) under which to attempt reclustering as leftovers")
        group.add_argument("--lo-max-runs",   type=int,   default=None, help="If leftovers, maximum number of leftover clustering attempts")
        group.add_argument("--lo-min-ius",    type=int,   default=600,  help="If leftovers, minimum number of leftover IUs to attempt reclustering")

        group = parser.add_argument_group(title="autoencoder hyperparameters")
        group.add_argument("--intermediate-layer-sizes", type=int, nargs="+", default=[100, 50], help="Intermediate layer sizes")
        group.add_argument("--latent-dim",               type=int,            default=8,         help="latent layer size")
        group.add_argument("--dropout-rate",             type=float,          default=0.2,       help="Dropout rate")
        group.add_argument("--epochs",                   type=int,            default=1000,      help="Training epochs")
        group.add_argument("--batch-size",               type=int,            default=500,       help="Training batch size")
        group.add_argument("--train-percent",            type=int,            default=80,        help="Percentage of dataset to use for training")
        group.add_argument("--activation",               type=str,            default="selu",    help="Activation function for non-output layers")

        group = parser.add_argument_group(title="sliced-wasserstein parameters")
        group.add_argument("--sliced-wasserstein", choices=["on", "off"], default="on", help="Use sliced-wasserstein (SW) to enforce a desired distribution in the latent space")
        group.add_argument("--sw-weight",          type=float,            default=100,  help="If SW, SW loss weight")
        group.add_argument("--sw-n-projections",   type=int,              default=50,   help="If SW, number of SW slices to use")

        group = parser.add_argument_group(title="clustering hyperparameters (per hunk)")
        group.add_argument("--gmm-n-components", type=int,                       default=300,   help="GMM max number of components")
        group.add_argument("--gmm-max-iter",     type=int,                       default=1000,  help="GMM max iterations")
        group.add_argument("--gmm-max-attempts", type=int,                       default=3,     help="GMM max attempts at convergence")
        group.add_argument("--on-no-converge",   choices=["skip", "use", "die"], default="use", help="What to do with the clustering results if it fails to converge")

        self.args = parser.parse_args()

        self.args.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Outputs will be created in %s", self.args.output_directory)
        self.out_fn = self.args.output_directory / f"{self.path.stem}.csv"

        self.metadata = {
            "args": self.args,
            "timestamp": self.timestamp,
        }

    def parse_length_arg(self, as_str):
        """ Parser for the --lengths argument
        """
        if as_str == "deciles":
            return Length(as_str, "deciles")
        else:
            try:
                (str_min, str_max) = as_str.split("-")
                (val_min, val_max) = (float(str_min), float(str_max))
                if val_min >= val_max:
                    raise argparse.ArgumentTypeError("Second value in range must be bigger than the first")
                return Length(as_str, "range", val_min, val_max)
            except ValueError:
                raise argparse.ArgumentTypeError("Must be either 'deciles' or a range (0.58-1.23)")

    def read_data(self):
        """ Read data, filter out bad IUs
        
        sets   - self.all_ius, self.all_vius, self.good_tmap, self.good_ius
        resets - self.curr_act_num, self.curr_hunk_num
        """
        fn = self.args.input_file
        
        self.all_ius = pd.read_hdf(fn, "ius")
        logger.info("Read %s row(s) from %s/%s", len(self.all_ius), fn, "ius")

        vius_name = "t_db_mst"
        with h5py.File(fn, "r") as fo:
            self.all_vius = np.array(fo[vius_name])
        logger.info("Read %s from %s/%s", self.all_vius.shape, fn, vius_name)
       
        self.good_tmap = self.all_ius.good
        self.good_ius  = self.all_ius.loc[self.good_tmap]

        num_good_ius = np.count_nonzero(self.good_tmap)
        logger.info("IUs marked 'good': %s", x_of_y(num_good_ius, len(self.all_ius)))
        if num_good_ius == 0:
            self.die("No IUs to work on")

        # Basically cosmetic; initialize some columns we're going to set later
        self.all_ius["subset"] = -1
        self.all_ius["act"] = ""
        self.all_ius["cluster"] = ""
        self.all_ius["cluster_n"] = 0
        self.all_ius["cluster_population"] = 0
        self.curr_act_num  = 0
        self.curr_hunk_num = 0

    def create_lengths(self) -> Generator:
        """ Selects the length parameters for the separation of IUs into acts
        
        sets     - self.curr_length_mode, self.curr_length_mnem, self.curr_length_min, self.curr_length_max, self.curr_length_subset
        modifies - self.all_ius[subset]
        """
        lspec = self.args.lengths

        if lspec.mode == "range":
            self.curr_length_mode = "range"
            self.curr_length_min = lspec.val_1
            self.curr_length_max = lspec.val_2
            self.curr_length_mnem = f"len_{self.curr_length_min * 1000:.0f}ms-{self.curr_length_max * 1000:.0f}ms"
            yield
        elif lspec.mode == "deciles":
            self.curr_length_mode = "subset"
            n_subsets = 10
            iu_subset = pd.qcut(self.good_ius.viu_t_len, n_subsets, labels=False)
            self.all_ius.loc[self.good_tmap, "subset"] = iu_subset
            for subset in range(n_subsets):
                self.curr_length_subset = subset
                self.curr_length_mnem = f"decile_{subset + 1:02d}"
                yield
        else:
            self.die(f"Unknown lengths parameter --lengths {lspec.as_str}?")

    def create_acts(self) -> Generator:
        """ Selects next act of IUs, based on self.curr_length_*
        Creates the feature vectors by interpolation, rolling mean, and resampling of the IU's data

        sets     - self.curr_act_tmap, self.curr_act_ius, self.curr_act_vius, self.curr_act_width, self.curr_act_inputs, self.curr_act_input_dim, self.curr_act
        modifies - self.curr_act_num, self.all_ius[act]
        """
        self.curr_act_num += 1

        # get this act's IUs and their data
        if   self.curr_length_mode == "subset":
            tmap = (self.all_ius.subset == self.curr_length_subset)                                                   # filtering of non-good IU already done in create_lengths
        elif self.curr_length_mode == "range":
            tmap = (self.good_tmap & self.all_ius.viu_t_len.between(self.curr_length_min, self.curr_length_max))    # filters non-good IUs
        else:
            raise NotImplementedError(f'curr_length_mode == "{self.curr_length_mode}"')

        tmap &= (self.all_ius.cluster == "")                  # don't process IUs that have already been clustered (should not happen)
        act_n = np.count_nonzero(tmap)

        if act_n < 1:
            logger.error("ERROR: No IUs for %s", self.curr_length_mnem)
            return

        self.curr_act_tmap = tmap
        self.curr_act_ius  = self.all_ius.loc[self.curr_act_tmap]
        self.curr_act_vius = self.all_vius[   self.curr_act_tmap]

        act = self.curr_length_mnem
        self.all_ius.loc[self.curr_act_tmap, "act"] = act

        # get the desired vector length for the current act
        curr_act_ius_median = self.curr_act_ius.viu_t_len.median()
        self.curr_act_width = int(curr_act_ius_median * self.args.feats_per_sec)

        curr_act_ius_min = self.curr_act_ius.viu_t_len.min()
        curr_act_ius_max = self.curr_act_ius.viu_t_len.max()
        logger.info(
            "%s: Working on %s IUs, median length %.02fs (%.03f-%.03fs), %s feature points",
            self.curr_length_mnem,
            act_n,
            curr_act_ius_median,
            curr_act_ius_min,
            curr_act_ius_max,
            self.curr_act_width,
        )
        
        # get the feature vectors for the current act
        vius_t    = self.curr_act_vius[:, :, 0]    # t_db_mst
        vius_mst  = self.curr_act_vius[:, :, 2]
        vius_db01 = norm_zero_one(self.curr_act_vius[:, :, 1])

        # interpolate and rolling mean the feature vectors
        vius_iwmst = []
        for viu_t, viu_mst, viu_db01 in zip(vius_t, vius_mst, vius_db01):                                    # go over all IUs in the act
            tmap = np.isfinite(viu_mst)                                                                      # get indexes where pitch is defined
            viu_imst = np.interp(viu_t, viu_t[tmap], viu_mst[tmap])                                          # interpolate missing pitch 
            vius_iwmst.append(weighted_rolling_mean(viu_imst, weights=viu_db01, winsize=self.args.winsize))  # intensity weighted rolling mean
        vius_iwmst = np.array(vius_iwmst)
        vius_t = vius_t[:, -vius_iwmst.shape[1]:]   # crop the time vectors to the same size as that of the feature vectors which has been reduced by the rolling mean

        # resample the feature vectors to the desired length while filling in invalid values
        vectors = []
        for viu_t, viu_feat in zip(vius_t, vius_iwmst):
            tmap = np.isfinite(viu_feat)
            if np.count_nonzero(tmap) < 2:
                self.die("IU feature vector has too few valid points")

            viu_vt = viu_t[tmap]
            viu_feat = viu_feat[tmap]
            vector_t = np.linspace(viu_vt[0], viu_vt[-1], self.curr_act_width)
            vectors.append(np.interp(vector_t, viu_vt, viu_feat))

        self.curr_act_inputs  = np.array(vectors)
        self.curr_act_input_dim = self.curr_act_inputs.shape[-1]

        yield act

    def create_hunks(self):
        """ Create the hunks for the current act -
        The main hunk, and then optionally hunks of leftovers

        sets     - self.curr_hunk_tmap, self.curr_hunk_inputs, self.curr_hunk
        modifies - self.curr_hunk_num, self.all_ius[hunk]
        """
        self.curr_hunk_num += 1
        
        hunk = self.curr_act 
        self.curr_hunk_tmap   = self.curr_act_tmap
        self.curr_hunk_inputs = self.curr_act_inputs

        self.all_ius.loc[self.curr_hunk_tmap, "hunk"] = hunk
        yield hunk


        if not self.args.leftovers:
            return

        # handle leftovers
        if self.args.lo_max_runs:
            n_leftovers = range(1, self.args.lo_max_runs + 1)
        else:
            n_leftovers = itertools.count(start=1)

        for n_leftover in n_leftovers:
            # get leftover IUs (IUs in tiny clusters)
            leftover_tmap = self.curr_act_tmap & (self.all_ius["cluster_ratio"] < self.args.lo_ratio) 

            # conditions to stop clustering leftovers
            curr_hunk_ratios = self.all_ius.loc[self.curr_hunk_tmap, 'cluster_ratio']
            if np.count_nonzero(leftover_tmap) < self.args.lo_min_ius:
                logger.info("No leftovers after %s", self.curr_hunk)
                return
            elif np.count_nonzero(curr_hunk_ratios >= self.args.lo_ratio) == 0:
                logger.info("No non-tiny clusters found in %s", self.curr_hunk)
                return


            self.curr_hunk_num += 1

            # set the leftovers to be the current hunk
            hunk = f"{self.curr_act}_leftovers_{n_leftover}"
            self.curr_hunk_tmap   = leftover_tmap
            self.curr_hunk_inputs = self.curr_act_inputs[self.curr_hunk_tmap[self.curr_act_tmap]]

            self.all_ius.loc[self.curr_hunk_tmap, "hunk"] = hunk
            yield hunk      # self.curr_hunk

    def autoencoder(self):
        """ Build and train the autoencoder for the current hunk
        
        sets - self.curr_history, self.curr_hunk_encoded, self.k_theta, self.k_z, self.k_SW_weight
        modifies - self.metadata
        """
        ## Construct the autoencoder network
        input_vectors         = {}
        output_vectors        = {}
        output_layers         = []
        output_layer_names    = []
        output_loss_functions = {}

        curr_hunk_n = np.count_nonzero(self.curr_hunk_tmap)

        # input and output vectors are the same for autoencoders
        input_vectors[  "input"] = self.curr_hunk_inputs
        output_vectors["output"] = self.curr_hunk_inputs

        # input layer
        layer = Input(shape=(self.curr_act_input_dim,), name="input")
        input_layer = layer

        # intermediate layers of the encoding
        for n, layer_size in enumerate(self.args.intermediate_layer_sizes, 1):
            if self.args.dropout_rate:
                layer = Dropout(self.args.dropout_rate)(layer)
            layer = Dense(layer_size, activation=self.args.activation, name=f"encoder_{n}",)(layer)

        # latent layer
        layer = Dense(self.args.latent_dim, activation=self.args.activation, name="latent",)(layer)
        latent_layer = layer

        # intermediate layers of the decoding
        for n, layer_size in enumerate(reversed(self.args.intermediate_layer_sizes), 1):
            layer = Dense(layer_size, activation=self.args.activation, name=f"decoder_{n}",)(layer)

        # output layer
        output_layer = Dense(self.curr_act_input_dim, name="output",)(layer)
        output_layers.append(output_layer)
        output_layer_names.append("output")
        output_loss_functions["output"] = self.mse_loss

        # sliced-wasserstein (SW) - used to enforce a desired distribution in the latent space
        if self.args.sliced_wasserstein == "on":
            self.k_theta     = K.variable(self.SW_generate_theta(), name="theta")  # initialize SW theta - random directions in the latent space to slice by
            self.k_z         = K.variable(self.SW_generate_z(),     name="z")      # initialize SW z - random points from the desired distribution. loss is calculated from the distance to these points
            self.k_SW_weight = K.variable(self.args.sw_weight,      name="weight") # set SW weight

            # add SW loss to the autoencoder
            output_layers.append(latent_layer)
            output_layer_names.append("latent")
            output_loss_functions["latent"] = self.SW_loss
            output_vectors["latent"] = np.zeros((curr_hunk_n, self.args.latent_dim))  # keras needs to have a desired output defined, even though we don't use it

        # compile the autoencoder
        e_model  = Model(inputs=input_layer, outputs=latent_layer,  name="Encoder")
        ae_model = Model(inputs=input_layer, outputs=output_layers, name="AE")
        ae_model.compile(optimizer="adam", loss=output_loss_functions,)

        ## train the autoencoder
        logger.info(f"{self.curr_hunk}: train {self.args.epochs} epochs - starting")

        # split train and test data
        train_idxs, test_idxs = train_test_split(
            np.arange(curr_hunk_n),
            train_size=self.args.train_percent / 100,
            random_state=self.rand_seed,
        )
        train_inputs  = {key: value[train_idxs] for key, value in input_vectors.items()}
        test_inputs   = {key: value[test_idxs]  for key, value in input_vectors.items()}
        train_outputs = {key: value[train_idxs] for key, value in output_vectors.items()}
        test_outputs  = {key: value[test_idxs]  for key, value in output_vectors.items()}

        # train
        history = ae_model.fit(
            train_inputs,
            train_outputs,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            shuffle=True,
            validation_data=(test_inputs, test_outputs),
            callbacks=[self.SW_make_reslice_callback()],
            verbose=False,
        )

        # get final losses
        self.curr_history = pd.DataFrame(history.history)
        final_round = self.curr_history.iloc[-1]
        losses = {
            col: final_round[col]
            for col in self.curr_history.columns
            if col.endswith("loss")
        }
        self.metadata[self.curr_hunk] = {"losses": losses}
        logger.info("Losses: %s",", ".join(f"{key}: {value:.03f}" for key, value in sorted(losses.items())),)
        if np.isnan(losses["loss"]):
            self.die("NaN loss, no point in proceeding")

        ## get the latent layer vectors of all the hunk's IUs
        self.curr_hunk_encoded = e_model.predict(self.curr_hunk_inputs)

        # get latent vector distances info
        dists = np.sqrt(np.sum(self.curr_hunk_encoded ** 2, axis=1))                                    # euclidean distances of the latent vectors from the origin
        logger.info(
            "%s ||latent vectors||: %.02f±%.02f (mean±SD), in shell: %s, in 1 SD: %s",
            self.curr_act,
            np.mean(dists),
            np.std(dists),
            x_of_y(np.count_nonzero((dists >= 0.9) & (dists <= 1.0)), len(dists)),                 # latent vectors that are in the 0.9-1.0 shell
            x_of_y(np.count_nonzero(np.abs(dists - np.mean(dists)) <= np.std(dists)), len(dists)), # latent vectors that are within 1 SD from the mean distance
        )

    def clusterer(self):
        """ Look for clusters in the latent distribution

        modifies - self.all_ius
        """
        # run GMM on the latent vectors of the current hunk
        logger.info(f"{self.curr_hunk}: GMM - starting")
        attempt_random_seed = self.rand_seed
        for attempt in range(self.args.gmm_max_attempts):
            gmm = BayesianGaussianMixture(
                n_components = self.args.gmm_n_components,
                max_iter     = self.args.gmm_max_iter,
                random_state = attempt_random_seed,
            )
            iu_labels = gmm.fit_predict(self.curr_hunk_encoded)

            if gmm.converged_:
                break
            else:
                logger.info("GMM attempt %d of %d failed to converge", attempt + 1, self.args.gmm_max_attempts,)
                attempt_random_seed = attempt_random_seed + 1

        if gmm.converged_:
            logger.info(
                "GMM converged to %d components after %d of max %d iterations",
                np.count_nonzero(~np.isclose(gmm.weights_, 0)),
                gmm.n_iter_,
                self.args.gmm_max_iter,
            )
        elif self.args.on_no_converge == "die":
            self.die("ERROR: GMM failed to converge. Exiting")
        elif self.args.on_no_converge == "skip":
            logger.error("ERROR: GMM failed to converge. Ignoring its labels and skipping to the next act")
            return
        elif self.args.on_no_converge == "use":
            logger.error("ERROR: GMM failed to converge. Using its labels anyway")
        else:
            raise NotImplementedError(f"--on-no-converge={self.args.on_no_converge}")

        # unique cluster labels (across acts)
        clust_num_buffer = 10**math.ceil(math.log10(self.args.gmm_n_components))       # make sure the numerical component of clusters' names doesn't overlap between hunks
        iu_cluster_names = (clust_num_buffer * self.curr_hunk_num) + iu_labels

        cluster_names, cluster_counts = np.unique(iu_cluster_names, return_counts=True)
        cluster_ns = dict(zip(cluster_names, cluster_counts))
        iu_cluster_ns = np.array([cluster_ns[cluster_name] for cluster_name in iu_cluster_names])
        act_n = len(self.curr_act_inputs)
        iu_cluster_ratios = iu_cluster_ns / act_n       # number of IUs in the cluster as a ratio of the number of IUs in the act 

        # add prefix to cluster names
        cluster_prefix = f"c{self.curr_act_num}"
        iu_cluster_names = [f"{cluster_prefix}_{iuc}" for iuc in iu_cluster_names]

        self.all_ius.loc[self.curr_hunk_tmap, "cluster"]            = iu_cluster_names
        self.all_ius.loc[self.curr_hunk_tmap, "cluster_n"]          = iu_cluster_ns         # number of IUs in this cluster
        self.all_ius.loc[self.curr_hunk_tmap, "cluster_population"] = act_n                 # number of IUs in this cluster's act
        self.all_ius.loc[self.curr_hunk_tmap, "cluster_ratio"]      = iu_cluster_ratios     # cluster_n / cluster_population

    def write_csv(self):
        """ Write outputs
        """
        ## write clustering results
        df = self.all_ius

        # remove 'subset' column
        df.drop(columns=["subset"], inplace=True)     

        # move 'act' and 'hunk' columns to the right
        cols_to_move = ['act', 'hunk']
        new_column_order = [col for col in df.columns if col not in cols_to_move] + cols_to_move
        df = df[new_column_order]

        # write to csv without index
        logger.info("Writing %s row(s) to %s", len(df), self.out_fn)
        df.to_csv(self.out_fn, index=False)


        ## write parameters
        fn = self.args.output_directory / "metadata.yml"
        metadata = yamlify(self.metadata)
        with open(fn, "w") as fo:
            fo.write(yaml.dump(metadata, allow_unicode=True))
    
    def make_and_save_random_state(self, random_state_fn):
        """make and save random state for reproducible results"""
        self.rand_seed = np.random.randint(0, 1e8)
        with open(random_state_fn, "wb") as fo:
            state = {
                "np":  np.random.get_state(),
                "python": random.getstate(),
                "seed": self.rand_seed,
            }
            pickle.dump(state, fo)
        tf.random.set_seed(self.rand_seed)

    def load_random_state(self, random_state_fn):
        """load random state for reproducible results"""
        with open(Path(random_state_fn), "rb") as fo:
            state = pickle.load(fo)
            np.random.set_state(state["np"])
            random.setstate(state["python"])
            self.rand_seed = state["seed"]          # used for tensorflow and sklearn (used as a parameter in the GMM)
            tf.random.set_seed(self.rand_seed)
            
    def mse_loss(self, inputs, autoencoded):
        """ Mean squared error loss function
        """
        return K.mean(K.square(inputs - autoencoded), axis=-1)

    # sliced-wasserstein (SW) methods

    def SW_generate_theta(self):
        """ generate SW theta
        return 'sw_n_projections' random directions in the latent space
        the directions are used to slice by when calculating the SW distance
        """
        return generate_points_on_unit_sphere(self.args.sw_n_projections, self.args.latent_dim)

    def SW_generate_z(self):
        """ generate SW z
        return 'self.args.batch_size' random points in a thin hypersphere shell in the latent space
        the distance of the autoencoded points from these points is used to calculate the SW loss
        """
        n = self.args.batch_size
        d = self.args.latent_dim
        r = np.power(np.random.uniform(low=0.9, high=1.0, size=n), 1 / d)
        return r[:, None] * generate_points_on_unit_sphere(n, d)

    def SW_loss(self, y_ones, y_pred):
        """ SW loss function
        Kolouri et al., 2018: Sliced-Wasserstein Autoencoder (arXiv:1804.01947v3)
        """
        n = K.shape(y_pred)[0]
        projected_y_pred = K.dot(y_pred,   K.transpose(self.k_theta))  # project the autoencoded points onto the slicing directions
        projected_z      = K.dot(self.k_z, K.transpose(self.k_theta))  # project the desired-distribution random points onto the slicing directions

        sorted_y_pred = tf.nn.top_k(tf.transpose(projected_y_pred), k=n, name="sorted_y_pred").values  # sort
        sorted_z      = tf.nn.top_k(tf.transpose(projected_z),      k=n, name="sorted_z"     ).values  # sort

        dist_unweighted = K.mean(K.square(sorted_y_pred - sorted_z))  # calculate the distance between the autoencoded points and the desired-distribution points
        return dist_unweighted * self.k_SW_weight                     # scale the distance by the weight

    def SW_make_reslice_callback(self):
        """ Keras callback to create for each training batch a new set of random SW slicing directions, 
        and a new set of random desired-distribution points to measure distance to
        
        sets - self.k_theta, self.k_z
        """
        class ResliceCallback(tf.keras.callbacks.Callback):
            def on_train_batch_begin(zelf, batch, logs=None):
                if self.args.sliced_wasserstein == "on":
                    K.set_value(self.k_theta, self.SW_generate_theta())
                    K.set_value(self.k_z,     self.SW_generate_z())

        return ResliceCallback()

def suppress_some_warnings():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"                                          # for tensorflow
    warnings.simplefilter(  action="ignore", category=pd.errors.PerformanceWarning)   # 'your performance may suffer as PyTables will pickle'
    warnings.filterwarnings(action='ignore', category=NaturalNameWarning)             # 'object name is not a valid Python identifier ... you will not be able to use natural naming to access this object'

def x_of_y(x, y):
    """Returns a nicely-formatted 'x percent of y' string
    """
    if y > 0:
        return f"{x:,} ({x * 100 / y:.02f}%) of {y:,}"
    return f"{x:,} of {y:,}"

def norm_zero_one(a):
    """ Normalize each row in 'a' to the [0, 1] range
    """
    row_min = np.nanmin(a, axis=1)
    row_max = np.nanmax(a, axis=1)
    row_range = row_max - row_min
    norm = (a - np.expand_dims(row_min, -1)) / np.expand_dims(row_range, -1)
    norm[row_range == 0, :] = 0
    return norm

def weighted_rolling_mean(a, weights, winsize):
    """ Weighting is within a window only,
    meaning that a point is weighted by its weight relative to the weight of the other points within a window,
    windows with an overall lower weight than other windows are not weighted less
    """
    assert a.shape == weights.shape

    a_times_w = a * weights

    windowed_a_times_w = as_strided_window(a_times_w, winsize)   # create rolling window representation
    windowed_weights   = as_strided_window(  weights, winsize)   # create rolling window representation

    numers = windowed_a_times_w.sum(axis=1)
    denoms = windowed_weights.sum(axis=1)

    denoms[denoms == 0] = np.nan
    means  = numers / denoms

    # for IUs with very few windows that have no nans, average also over windows that have some nans
    if sum(np.isfinite(denoms)) < 2:
        denom_nans = np.isnan(windowed_weights).all(axis=1)
        denoms = np.nansum(windowed_weights, axis=1)
        denoms[denom_nans] = np.nan

        numer_nans = np.isnan(windowed_a_times_w).all(axis=1)
        numers = np.nansum(windowed_a_times_w, axis=1)
        numers[numer_nans] = np.nan
        
        denoms[denoms == 0] = np.nan
        means = numers / denoms
        
    return means

def as_strided_window(a, winsize):
    """ Creates a read-only rolling window representation of an array without copying its data
    """
    a_stride, = a.strides
    return np.lib.stride_tricks.as_strided(
        a,
        shape=((len(a) - winsize + 1), winsize),
        strides=(a_stride, a_stride),
        writeable=False,
    )

def generate_points_on_unit_sphere(n, d):
    """ Return n random directions (points on the unit sphere) in d-dimensional space
    Method 19 in http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    u = np.random.normal(size=(n, d))
    norm = np.sqrt(np.sum(np.square(u), axis=1))
    return u / norm[:, None]

def yamlify(o):
    """ Return an easily-yamlable version of `o`.
    """
    if o is None:
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.float64):
        return float(o)
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, (int, str, float)):
        return o
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, argparse.Namespace):
        return yamlify(vars(o))
    if isinstance(o, (list, tuple, set)):
        return [yamlify(e) for e in o]
    if isinstance(o, dict):
        return {yamlify(k): yamlify(v) for k, v in o.items()}
    if callable(getattr(o, "group", None)):
        return yamlify(o.group(0))
    if dataclasses.is_dataclass(o):
        return yamlify(dataclasses.asdict(o))
    raise NotImplementedError(f"yamlify({type(o)})")

def die(message):
    logger.error(message)
    sys.exit(-1)

if __name__ == "__main__":
    IUClust().run()
