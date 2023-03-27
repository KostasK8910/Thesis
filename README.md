# Thesis
ECG Analysis via Deep Learning

## Project's goal

- Check the ability of a Deep Learning Model to Classify Electrocardiograms (ECGs) into Cardiovascular Diseases (CVDs).


## Approach

1. Train the model on a subset of the data (20,000/345,779 ECGs ~ 5.8%)
2. Perform Hyperparameter Tuning on the full dataset (345,779 ECGs) for the Parameters: 
    1. kernel size
    2. dropout rate
    3. optimizer
3. Train the model on a different dataset in order to check its ability to generalize
4. Preprocess the new dataset and retrain the model
5. Train the model on the initial dataset before and after the selection of the optimal Parameters
6. Test the final model on:
    1. initial data
    2. raw data from second dataset
    3. preprocessed data from second dataset through:
        1. cropping
        2. downsampling
        

## Model Architecture and datasets

The model architecture that was selected for this project was designed by A.H. Ribeiro et al., for their research, titled
[Automatic diagnosis of the 12-lead ECG using a deep neural network.](https://www.nature.com/articles/s41467-020-15432-4) Nature Communications, 11, 1760, (2020).

Data used for training the models:

1. Initial [dataset](https://zenodo.org/record/4916206) which is part (~15%) of the dataset used for the research of A.H. Ribeiro et al.
2. Secondary [dataset](https://physionet.org/content/ptb-xl/1.0.3/)
