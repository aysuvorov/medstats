import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# +-----------------------------------------------------------------------------

# Load data

class DataSplitter:
    def __init__(self,
        data_path,
        save_path,
        target_var,
        shuffler,
        resampler,
        test_prop,
        random_state,
        resamp_n_samples = None
        ):

        self.data_path = data_path
        self.save_path = save_path
        self.target_var = target_var
        if shuffler == 0:
            self.shuffler = False
        else:
            self.shuffler = True
        self.test_prop = test_prop
        self.random_state = random_state
        if resampler == 0:
            self.resampler = False
        else:
            self.resampler = True
        self.resamp_n_samples = resamp_n_samples
        


    def __call__(self):
        self.df = pd.read_pickle(self.data_path)

        self.X = self.df.drop(self.target_var, axis='columns')
        self.Y = self.df[self.target_var]

        self.X_train, self.X_test, \
            self.y_train, self.y_test = train_test_split(
                self.X, 
                self.Y, 
                test_size=self.test_prop, 
                random_state=self.random_state,
                shuffle=self.shuffler
                )

        # To resample data
        if self.resampler:

            _target_0_df = self.X_train.loc[self.y_train[self.y_train == 0].index, :]
            _target_1_df = self.X_train.loc[self.y_train[self.y_train == 1].index, :]

            # Create resampled dataset with 500 patients and 1:1 target classes 
            _X_train_res = pd.concat([
                resample(_target_0_df, n_samples = self.resamp_n_samples, \
                    random_state = self.random_state),
                resample(_target_1_df, n_samples = self.resamp_n_samples, \
                    random_state = self.random_state)
            ])

            _y_train_res = np.hstack([np.zeros(self.resamp_n_samples), \
                np.ones(self.resamp_n_samples)])

            _X_train_res['target'] = _y_train_res

            _X_train_res = _X_train_res.sample(frac = 1, random_state = self.random_state)

            self.X_train = _X_train_res.drop('target', axis='columns')
            self.y_train = _X_train_res['target']

            return_phrase = '\nData splitted, resampled and saved\n'

        else:

            return_phrase = '\nData splitted and saved\n'
            
        # Printing information about classes in target variables for train/test:
        print('\nTarget parameters: \n')
        for target, name in zip([self.y_train, self.y_test], ['TRAIN', 'VALIDATION']):
            print(f'- {name} size is {len(target)} with {int(target.astype(float).sum())} targets which is {int(target.astype(float).sum() / len(target)*100)}%')

        # Printing information about train dataset:
        print(f'\nThere are {len(self.X_train.columns)} columns in the sets')

        self.X_train.to_pickle(self.save_path + "/X_train.pkl")
        self.y_train.to_pickle(self.save_path + "/y_train.pkl")
        self.X_test.to_pickle(self.save_path + "/X_test.pkl")
        self.y_test.to_pickle(self.save_path + "/y_test.pkl")

        print(return_phrase)








