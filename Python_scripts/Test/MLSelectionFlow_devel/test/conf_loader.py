import pandas as pd
import yaml

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore



# +-----------------------------------------------------------------------------
# DATA LOADING

@dataclass
class Config:
    cfg: 

configuration = cs = ConfigStore.instance()
cs.store(name="config", node=Config)

class SearchFlow:
    def __init__(self):
        pass

     
    def load_globals(self, cfg):
        
        # GLOBALS
        self.X = pd.read_pickle(cfg.PATH_TO_X)
        self.Y = pd.read_pickle(cfg.PATH_TO_Y)
        self.RS = cfg.RANDOM_STATE
        self.OPTUNA_TRIAL_NUMBER = cfg.OPTUNA_TRIAL_NUMBER
        self.TEST_SIZE = cfg.TEST_SIZE
        self.STRATIFY = cfg.STRATIFY
        self.CLASS_WEIGHT = dict(cfg.CLASS_WEIGHT)
        self.N_PREDICTORS = cfg.N_PREDICTORS
        self.BALANCED = cfg.BALANCED
        self.PATH_TO_OUTPUT = cfg.PATH_TO_OUTPUT
        print("Globals loaded...")
        print(type(self.CLASS_WEIGHT))
        print(self.CLASS_WEIGHT)


@hydra.main(version_base=None, 
    config_path="../MLSelectionFlow_devel/data/config", 
    config_name="config") 
def main(cfg: Config):
    data = SearchFlow()
    data.load_globals(cfg)

if __name__ == "__main__":
    main()
