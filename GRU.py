import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GRUModel:
    """
    Class for defining the GRU model.
    """
    
    def __init__(self, input_shape: tuple) -> None:
        """
        Initializes an object of the GRU Model class.

        Args:
            input_shape (tuple): A tuple denoting the input shape.
        """
        self.model = Sequential() # Creates a sequential class object.
        self.model.add(GRU(64, input_shape=input_shape, return_sequences=False)) # Adds a GRU layer.
        self.model.add(Dense(1, activation='sigmoid')) # Adds a dense layer for binary classification.
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    def train(self, training_data: tuple, epochs: int, batch_size: int, validation_data: tuple) -> None:
        """Method to train the model.

        Args:
            training_data (tuple): A tuple of (train_x, train_y).
            epochs (int): Number of epochs.
            batch_size (int): Gives the batch size.
            validation_data (tuple): A tuple of (val_x, val_y).
        """
        train_x, train_y = training_data # Getting the train_x and train_y.
        val_x, val_y = validation_data # Getting the val_x and val_y.
       
        # Converting the data to numpy arrrays.       
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)

        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y)

        # Training the model.
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

    def eval(self, data: tuple) -> None:
        """Method to evaluate the model on some data.

        Args:
            data (tuple): A tuple of (X_data, Y_data).
        """
        X_data, Y_data = data # Getting the X_data and Y_data.
        y_pred_probs = self.model.predict(X_data)  # Gets the prediction as probabilitites.
        y_pred = (y_pred_probs > 0.5).astype(int) # Converts the probabilities to class labels.

        # Measuring the model performance in terms of Precion, Recall and F1_Score.
        report = classification_report(Y_data, y_pred, digits=4)
        print("The classfication report for the model is given below.")
        print(report)

        # Calculating the confusion matrix.
        cm = confusion_matrix(Y_data, y_pred)

        # Visualizing the confusion matrix using a heatmap
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()

    def save(self, path: str) -> None:
        """Saves the trained model to the given path.

        Args:
            path (str): Path to save the model to.
        """
        # Saves the model.
        self.model.save(path)


    

