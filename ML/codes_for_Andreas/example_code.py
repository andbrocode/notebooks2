import os
import joblib
import os
os.environ["KERAS_BACKEND"] = "jax"
# Ensure no Keras-related imports happen before setting the environment variable
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from utils import *

# Now check the backend
backend = K.backend()
print(f"Active backend: {backend}")

# Load data:
input_df, output_df, datetime_rdt = preprocess_CHAMP_Te('./../1_prepare_CHAMP_data/CHAMP_data_filtered.csv', 
                                                        ['Alt', 'QDLat', 'QDLon', 'UT', 'DOY', 'smoothed SYM-H', 'd(SYM-H)/dt', 'HpS', 'P10.7', 'Em_s'])#'Alt', 'QDLat', 'QDLon', 'UT', 'DOY', 'SYM-H', 'HpS', 'P10.7', 'Em_s'
print(input_df.keys())

print(len(input_df), len(output_df), len(datetime_rdt))

# split the data into train - valid. - test parts:
train_indices, valid_indices, test_indices = data_splitting(datetime_rdt)
    
# scale the data and save scalers:
scaler_input, scaler_output, X_train, Y_train, X_valid, Y_valid, X_test, Y_test = scale_data(MinMaxScaler, input_df.to_numpy(), output_df.to_numpy(), train_indices, valid_indices, test_indices)

joblib.dump(scaler_input, './models/scaler_input_Te_3.save')
joblib.dump(scaler_output, './models/scaler_output_Te_3.save')

keras.backend.set_floatx('float32')  # Set JAX float type for Keras (string format)

#------ Define the model (see utils for the complete function):
model = create_3layer_NN(hidden_neurons_layer_1 = 256,
                          hidden_neurons_layer_2 = 64,
                          hidden_neurons_layer_3 = 256,
                          input_dimension        = X_train.shape[1], 
                          dropout_rate           = 0.03,
                          gaussian_noise         = 0.03,
                          kernel_constraint      = keras.constraints.MinMaxNorm(min_value=-0.05, max_value=1.0, rate=1.0), 
                          learning_rate          = 0.0002856,
                          loss_function          = keras.losses.MeanAbsoluteError
                          ) # feel free to change these parameters! This is just for this application.


# we will save models into the following directory:
try:
    os.makedirs('./models/models_by_epoch/')
except:
    print('exists')

#------ DEFINING CALLBACKS:
# This one stops the training when the validation metrics dont improve over 75 epochs:
early_stopping = EarlyStopping(monitor='val_loss', patience=75)

# This one saves the best model:
model_checkpoint_callback = ModelCheckpoint(
    filepath='./models/model_Te_CHAMP_5.keras',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# This one saves the models at every epoch:
filepath = "./models/models_by_epoch/model_Te_CHAMP_{epoch:02d}_5.keras"
model_checkpoint_callback_everyepoch = ModelCheckpoint(
    filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False)

# This one saves the training and validation loss values at every epoch to a csv file:
csv_logger = CSVLogger(
  './models/model_Te_CHAMP_5.csv', 
  separator=',', 
  append = False
)

# This one reduces the learning rate in case the learning doesnt improve for 15 epochs:
RL = ReduceLROnPlateau(
  monitor='val_loss', factor=0.5, patience=15,
  mode='auto', min_delta=0.0002, cooldown=0, min_lr=0,
)

#------ This is the model traning:
model_history = model.fit(X_train, Y_train,
                     validation_data=(X_valid, Y_valid),
                     batch_size=64, epochs=200, verbose=1, shuffle=True, 
                     callbacks=[early_stopping, model_checkpoint_callback, model_checkpoint_callback_everyepoch, RL, csv_logger])