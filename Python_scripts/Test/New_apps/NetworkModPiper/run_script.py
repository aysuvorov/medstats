# utf8

import hydra
import warnings

import pandas as pd


import src.DataSplit as splitter
import src.FeatureSelector as selector
import src.SelectedFeaturePreprocessor as sfp
import src.SynoliticEncoder as genc
import src.ConvNetwork as CNN

from omegaconf import DictConfig

warnings.filterwarnings("ignore")

# Execute whole script
@hydra.main(version_base=None, 
    config_path="./config", 
    config_name="config")
def main(cfg: DictConfig):

    # Split and resample data
    SplitData = splitter.DataSplitter(
            data_path = cfg.data.PATH_TO_RAW_DATA,
            save_path = cfg.data.PATH_TO_SPLIT_DATA,
            target_var = cfg.data.TARGET,
            shuffler = cfg.training.SHUFFLE,
            resampler = cfg.training.RESAMPLER,
            test_prop = cfg.training.TEST_PROP,
            random_state = cfg.RS,
            resamp_n_samples = cfg.training.RESAMP_SAMP
        )
    SplitData()

    # Select features with Lasso...
    LassoSelectFeatures = selector.LassoSelector(
        cfg.data.PATH_TO_SPLIT_DATA,
        cfg.data.PATH_TO_SELECTED_FEATURES,
        cfg.RS)
    LassoSelectFeatures()

    # Select features with XGB...
    XgbSelectFeatures = selector.XgbSelector(
        cfg.data.PATH_TO_SPLIT_DATA,
        cfg.data.PATH_TO_SELECTED_FEATURES,
        cfg.RS)
    XgbSelectFeatures()

    # Preprocess selected features
    PreprocessFeatures = sfp.Preprocessor(
        cfg.data.PATH_TO_SPLIT_DATA,
        cfg.data.PATH_TO_SELECTED_FEATURES,
        cfg.data.PATH_TO_PREPROCESSED_DATA,
        cfg.features.FEATURE_SELECTOR_MODEL,
        cfg.features.FEATURES_LEN,
        cfg.features.USE_PCA,
        cfg.RS
        )
    PreprocessFeatures()

    # Create graphs
    AdjMatrixEncoder = genc.AdjMatrixComputer(
        cfg.data.PATH_TO_PREPROCESSED_DATA,
        cfg.data.PATH_TO_ADJUST_MATRICES,
        cfg.RS
        )
    AdjMatrixEncoder()

    # Train / Validate CNN
    # CNNTrainValidate = CNN.CNNTrainer(
    #     cfg.data.PATH_TO_ADJUST_MATRICES,
    #     cfg.data.PATH_TO_CNN_MODELS,
    #     cfg.neural_network.LEARNING_RATE,
    #     cfg.neural_network.EPOCHS,
    #     cfg.neural_network.BATCH_SIZE,     
    #     cfg.RS
    #     )
    # CNNTrainValidate.fit()

    cnn = CNN.CNNTrainer()
    cnn.fit(
        cfg.data.PATH_TO_ADJUST_MATRICES,
        cfg.data.PATH_TO_CNN_MODELS,
        cfg.neural_network.LEARNING_RATE,
        cfg.neural_network.EPOCHS,
        cfg.neural_network.BATCH_SIZE,     
        cfg.RS
    )

    print('Full script ready...')

if __name__ == "__main__":
    main()