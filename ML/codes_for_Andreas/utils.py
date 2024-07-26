import numpy as np
import pandas as pd
import os
import os
os.environ["KERAS_BACKEND"] = "jax"
# Ensure no Keras-related imports happen before setting the environment variable
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise, Activation, LayerNormalization
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# Now check the backend
backend = K.backend()
print(f"Active backend: {backend}")

def create_3layer_NN(hidden_neurons_layer_1, hidden_neurons_layer_2, hidden_neurons_layer_3, 
                     input_dimension, dropout_rate, gaussian_noise,
                     kernel_constraint, learning_rate, loss_function):

    """
    Creates a 3-layer neural network model.

    :param hidden_neurons_layer_1: int, number of neurons in the first hidden layer.
    :param hidden_neurons_layer_2: int, number of neurons in the second hidden layer.
    :param hidden_neurons_layer_3: int, number of neurons in the third hidden layer.
    :param input_dimension: int, number of input features.
    :param dropout_rate: float, dropout rate for regularization.
    :param gaussian_noise: float, standard deviation of Gaussian noise to be added.
    :param kernel_constraint: keras constraint, constraint function applied to the kernel weights.
    :param learning_rate: float, learning rate for the optimizer.
    :param loss_function: function, loss function for model training.

    :return: keras.Sequential, compiled Keras model.
    """
    
    model = Sequential()
    #hidden layer 1
    model.add(Dense(hidden_neurons_layer_1, input_dim=input_dimension))
    model.add(GaussianNoise(gaussian_noise))
    model.add(Activation('gelu'))# gelu or, 'softplus'
    model.add(Dropout(dropout_rate))
    model.add(LayerNormalization(center=True , scale=True))

    # hidden layer 2
    model.add(Dense(hidden_neurons_layer_2))
    model.add(GaussianNoise(gaussian_noise))
    model.add(Activation('gelu'))
    model.add(Dropout(dropout_rate))
    model.add(LayerNormalization(center=True , scale=True))

    # hidden layer 3
    model.add(Dense(hidden_neurons_layer_3))
    model.add(GaussianNoise(gaussian_noise))
    model.add(Activation('gelu'))
    model.add(Dropout(dropout_rate))
    model.add(LayerNormalization(center=True , scale=True))

    #output layer
    model.add(Dense(1, kernel_constraint = kernel_constraint))

    opt = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function()) 
    
    return model
    
def Add_FFT(input_df, feature_list, feature_lim, max_order):
    """
    Adds Fourier features for given input variables and given maximum FFT orders to the input dataframe.

    :param input_df: pd.DataFrame, input dataframe.
    :param feature_list: list, list of feature names to apply FFT.
    :param feature_lim: list, list of feature limits corresponding to feature_list.
    :param max_order: int, maximum order of the Fourier series.

    :return: pd.DataFrame, dataframe with added FFT features.
    """

    for feature_ind in range(len(feature_list)):
        for order in range(1, max_order+1):
            feature_sine_name = 'sin_{}_{}'.format(order, feature_list[feature_ind])
            sine_values = np.sin(order* input_df[feature_list[feature_ind]].values/feature_lim[feature_ind] *2*np.pi)
            input_df[feature_sine_name]= sine_values

            feature_cosine_name = 'cos_{}_{}'.format(order, feature_list[feature_ind])
            cosine_values = np.cos(order* input_df[feature_list[feature_ind]].values/feature_lim[feature_ind] *2*np.pi)
            input_df[feature_cosine_name]= cosine_values

        input_df = input_df.drop([feature_list[feature_ind]], axis=1)

    return input_df

def Add_FFT_position(input_df, feature_list, feature_lim, max_order, mlat):
    """
    Adds Fourier features for given input variables and given maximum FFT orders to the input dataframe, 
    and damps the harmonics close to the Poles.

    :param input_df: pd.DataFrame, input dataframe.
    :param feature_list: list, list of feature names to apply FFT.
    :param feature_lim: list, list of feature limits corresponding to feature_list.
    :param max_order: int, maximum order of the Fourier series.
    :param mlat: np.ndarray, array used to damp harmonics close to the Poles.

    :return: pd.DataFrame, dataframe with added FFT features.
    """

    polar_angle = np.abs(np.deg2rad(mlat))
    smoothing_factor = np.exp(-50 * np.square(polar_angle - np.pi/2))

    for feature_ind in range(len(feature_list)):
        for order in range(1, max_order+1):
            feature_sine_name = 'sin_{}_{}'.format(order, feature_list[feature_ind])
            sine_values = np.sin(order* input_df[feature_list[feature_ind]].values/feature_lim[feature_ind] *2*np.pi)
            input_df[feature_sine_name]= sine_values*(1-smoothing_factor) + 1e-6

            feature_cosine_name = 'cos_{}_{}'.format(order, feature_list[feature_ind])
            cosine_values = np.cos(order* input_df[feature_list[feature_ind]].values/feature_lim[feature_ind] *2*np.pi)
            input_df[feature_cosine_name]= cosine_values*(1-smoothing_factor) + 1e-6

        input_df = input_df.drop([feature_list[feature_ind]], axis=1)
    
    return input_df

def preprocess_CHAMP_Te(df_filename:str,
                         input_columns:list):

    """
    Processes CHAMP Te data, interpolates geomagnetic and solar indices, and adds Fourier features.

    :param df_filename: str, filename of the input dataframe.
    :param input_columns: list, list of input column names.

    :return: tuple, (input dataframe, output dataframe, datenum values)
    """

    df_Te = pd.read_csv(df_filename,index_col=0).drop_duplicates().reset_index(drop=True).astype('float')

    print(df_Te.head(5))
    
    df_Te.rename(columns={'hour_UT': 'UT'}, inplace=True)
    
    # interpolating omni_data:
    omni_df = pd.read_csv('./../0_prepare_omni_data/OMNI_df.csv', index_col=0)
    
    for col in ['smoothed SYM-H', 'd(SYM-H)/dt', 'HpS', 'P10.7', 'Em_s']:
        df_Te[col] = np.interp(df_Te['datenum'].values, omni_df['datenum'].values, omni_df[col].values)
        
    df_Te = df_Te.dropna().reset_index(drop=True)
    
    # see that data are within constraints:
    mask = (df_Te['Te'].values >= 400)&(df_Te['Te'].values <= 4500)&(df_Te['Ne'].values >= 2*1e4)
    df_Te = df_Te[mask].dropna().reset_index(drop=True)
    
    # selecting input and output parameters:
    input_df = df_Te.loc[:, input_columns].copy()
    output_df = df_Te.loc[:, 'Te'].copy()    
    
    input_df = Add_FFT_position(input_df, 
                       ['QDLon'], [360], 3, input_df['QDLat'].values.astype('float'))
    
    input_df = Add_FFT(input_df, 
                       ['QDLat'],
                       [360],
                       4)
    
    input_df = Add_FFT(input_df, 
                       ['DOY'],
                       [365],
                       3)
    
    input_df = Add_FFT(input_df, ['UT'], [24], 4)
        
    return input_df, output_df, df_Te['datenum'].values


def data_splitting(datetime_rdt):
    """
    Splits data into training, validation, and test sets based on specific dates.
    The general idea is to split the data into blocks in time (I used 1 week duration and defined this splitting before), and then
    use those dates to split the data into 3 sets.

    :param datetime_rdt: np.ndarray, array of datetime values.

    :return: tuple, (train indices, valid indices, test indices)
    """

    train_segments = np.load('./../1_prepare_CHAMP_data/train_segments.npy')
    train_indices = []
    for ind in range(len(train_segments)):

        indices = np.where((datetime_rdt>=train_segments[ind,0])&(datetime_rdt<train_segments[ind,1]))[0].tolist()
        train_indices = [*train_indices, *indices]

    valid_segments = np.load('./../1_prepare_CHAMP_data/valid_segments.npy')
    valid_indices = []
    for ind in range(len(valid_segments)):

        indices = np.where((datetime_rdt>=valid_segments[ind,0])&(datetime_rdt<valid_segments[ind,1]))[0].tolist()
        valid_indices = [*valid_indices, *indices]

    test_segments = np.load('./../1_prepare_CHAMP_data/test_segments.npy')
    test_indices = []
    for ind in range(len(test_segments)):

        indices = np.where((datetime_rdt>=test_segments[ind,0])&(datetime_rdt<test_segments[ind,1]))[0].tolist()
        test_indices = [*test_indices, *indices]

    return train_indices, valid_indices, test_indices

def scale_data(scaling_method, input_data, output_data, train_indices, valid_indices, test_indices):
    """
    Scales the input and output data using the provided scaling method.

    :param scaling_method: callable, method to scale the data.
    :param input_data: np.ndarray, input data to be scaled.
    :param output_data: np.ndarray, output data to be scaled.
    :param train_indices: list, indices for training data.
    :param valid_indices: list, indices for validation data.
    :param test_indices: list, indices for test data.

    :return: tuple, (scaler for input, scaler for output, scaled training data, 
                     scaled training output, scaled validation data, scaled validation output, 
                     scaled test data, scaled test output)
    """
    
    # fitting the scalers on the train data
    scaler_input = scaling_method().fit(input_data[train_indices])
    scaler_output = scaling_method().fit(output_data[train_indices].reshape(-1, 1))
    # transform ALL data
    input_scaled = scaler_input.transform(input_data)
    output_scaled = scaler_output.transform(output_data.reshape(-1, 1))
    # selecting the subsets
    X_train = input_scaled[train_indices,:]
    Y_train = output_scaled[train_indices]
    X_valid = input_scaled[valid_indices,:]
    Y_valid = output_scaled[valid_indices]
    X_test = input_scaled[test_indices,:]
    Y_test = output_scaled[test_indices]
    
    return scaler_input, scaler_output, X_train, Y_train, X_valid, Y_valid, X_test, Y_test