import yaml
import hydra
import warnings

import pandas as pd

import src.DataSplit as splitter
import src.FeatureSelector as selector
import src.SklearnMainModelling as smm

from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

warnings.filterwarnings("ignore")

# Load global variables
class Globals:
    def load_globals(self, cfg):
        self.RS = cfg.RS
        self.PATH_TO_RAW_DATA = cfg.PATH_TO_RAW_DATA
        self.PATH_TO_SPLIT_DATA = cfg.PATH_TO_SPLIT_DATA
        self.TARGET = cfg.TARGET
        self.TEST_PROP = cfg.TEST_PROP
        self.SHUFFLE = cfg.SHUFFLE
        self.RESAMPLER = cfg.RESAMPLER
        self.RESAMPLER_SAMP = cfg.RESAMP_SAMP

        self.FEATURES = cfg.FEATURES
        self.FEATURES_LEN = cfg.FEATURES_LEN

        print("Globals loaded...")






# Execute whole script
@hydra.main(version_base=None, 
    config_path="./config", 
    config_name="config")

def main(cfg: DictConfig):

    # Load globals
    globals = Globals()
    globals.load_globals(cfg)

    # Split and resample data
    split_data = splitter.DataSplitter(
            data_path = globals.PATH_TO_RAW_DATA,
            save_path = globals.PATH_TO_SPLIT_DATA,
            target_var = globals.TARGET,
            shuffler = globals.SHUFFLE,
            resampler = globals.RESAMPLER,
            test_prop = globals.TEST_PROP,
            random_state = globals.RS,
            resamp_n_samples = globals.RESAMPLER_SAMP
        )
    split_data()

    # Select features with Lasso...
    LassoSelectFeatures = selector.LassoSelector(
        globals.PATH_TO_SPLIT_DATA,
        globals.RS)
    LassoSelectFeatures()

    # Select features with XGB...
    XgbSelectFeatures = selector.XgbSelector(
        globals.PATH_TO_SPLIT_DATA,
        globals.RS)
    XgbSelectFeatures()

    # Train / validate on sklearn models
    sklearner = smm.Preprocessor(
        globals.PATH_TO_SPLIT_DATA,
        globals.FEATURES,
        globals.FEATURES_LEN,
        globals.RS
        )
    sklearner.preprocess(False)
    sklearner.run(15)

    print('Full script ready...')

if __name__ == "__main__":
    main()