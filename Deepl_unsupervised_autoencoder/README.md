# Unsupervised Autoencoder by Nathan Dze Kadisoh 

## data preperation for UNSW datset feaure analysis from Alexej and data preparation for credit card by Nathan 
I got the help of my partner for the data analysis and feature analysis part (data_preperation.py) i used in this my code 

## data_preperation.py (Code thanks to Alexej for feature analysis)
implements feature analysis which generally picks out the most import features of the choosen data set the method call to this file involves a parameter called "feature_analysis" which is either true or false and nothing else. true implies the feature analysis will be made on the choosen dataset and false implies no feature analysis will be made 

## data_normalisation.py
this file is involved in normalising the dataset using 3 different methods. it takes a key parameter called "normalisation" which when called can be set to either 1, 2 or 3.
### parameter is 1 normalisation used will be standard scaler and returns the datset for training using this normalisation
### parameter is 2 normalisation used will be min max scaler and returns the datset for training using this normalisation
### parameter is 3 normalisation used will be a simple one and returns the datset for training using this normalisation

## credit_card_data_preparation.py
this file prepares the credit card dataset for training it also adds a time series column called "time_in_years" of data type timestamp. This helps to simulate time dependencies in the dataset.
The file takes 2 main parameters called sequence_length and normalization 
### sequence_length parameter used by function create_sequences to create a sequence of length sequence_length( in code 12)
### normalization parameter used by function normalises the dataset either according to standard scaler (either normalisation is 1) or according to min max scaler(either normalisation is 2) 

### the method extract_time_interval_df in the file used to extract a set of data points to perform anomalie detection in that set. it uses parameters 
#### start_time indicates the start time from which the data extraction starts. eg '2013-09-01 00:00:01'
#### end_time indicates the end time from which the data extraction stops. eg '2013-09-01 00:51:19'
#### sequence_length the length of sequence for creating sequence data

## unsupervised_Autoencoder.ipynb
this notebook contains the implementation of 4 types of autoencoders each used for a particular data set

### Sparse autoencoder 
includes parameters like num_features, encoding_dim and sparsity_factor to vary the strength and constraction of the autoencoder according to the dataset used. produces best results when normalisation is 2 (MinMaxScaler) and feature analysis set to false 
### convolutional autoencoder
includes parameters like initial_learning_rate, decay_steps, decay_rate,lam which all help to regularize the behaviour of the model. the best results using the unsw dataset is obtained when feature analysis is set to false and normalisation is 2(MinMaxScaler)
### sparse convolutional autoencoder
used in the credit card dataset and uses simialr parameter like the sparse autoencoder used. in this case rather than simple dense layers used rather there is the use of convolutional layers
### contractive convolutional autoencoder
this autoencoder used in this case too is similar to the contractive autoencoder with it's parameters. it uses in this case convolutiional layers. 
convolutional layers are used because of the datset structure which changes and the perfomance is overall better.
