from GRU import GRUModel
import pickle
import os
from setup import *

if __name__ == '__main__':
    # Loading the saved data.
    with open(os.path.join(train_directory, X_train_file_name), 'rb') as file:
        X_train = pickle.load(file)

    with open(os.path.join(train_directory, Y_train_file_name), 'rb') as file:
        Y_train = pickle.load(file)

    with open(os.path.join(train_directory, X_val_file_name), 'rb') as file:
        X_val = pickle.load(file)

    with open(os.path.join(train_directory, Y_val_file_name), 'rb') as file:
        Y_val = pickle.load(file)
        
    # Model creation.
    print("Creating the GRU model.")
    model = GRUModel(input_shape = (X_train.shape[1],1))

    # Training the model.
    print("Training the model.")
    model.train(training_data=(X_train, Y_train), epochs=10, batch_size=32, validation_data=(X_val, Y_val))

    # Evaluating the model on the validation set.
    print("Evaluating the model on the validation set.")
    model.eval(data=(X_val, Y_val))

    # Saving the model.
    print("Saving the trained model to the following path: ", model_file_path)
    model.save(path = model_file_path)




