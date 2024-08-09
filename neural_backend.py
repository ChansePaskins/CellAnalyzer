import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(main_folder_path, img_height, img_width):
    images = []
    cell_counts = []

    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        images_path = os.path.join(folder_path, 'images')
        masks_path = os.path.join(folder_path, 'masks')

        img_files = os.listdir(images_path)
        img_files.sort()

        for img_name in img_files:
            img_file_path = os.path.join(images_path, img_name)

            img = load_img(img_file_path, color_mode='grayscale')
            img = img_to_array(img)
            img = cv2.resize(img, (img_width, img_height))
            img = img / 255.0
            images.append(img)

            # Count the number of mask files
            cell_count = len(os.listdir(masks_path))
            cell_counts.append(cell_count)

    return np.array(images), np.array(cell_counts)


def regression_model(img_height, img_width, img_channels):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)  # Single output for the cell count
    ])
    return model


# Example usage
if __name__ == '__main__':
    image_path = r"C:\Users\chans\Documents\GitHub\CellAnalyzer\keras_labels\kaggle"
    IMG_HEIGHT = 1024
    IMG_WIDTH = 1024
    IMG_CHANNELS = 1  # Grayscale images

    images, cell_counts = load_data(image_path, IMG_HEIGHT, IMG_WIDTH)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, cell_counts, test_size=0.2, random_state=42)

    # Create the regression model
    model = regression_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Define callbacks for saving the best model and early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('cell_count_model.keras', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', save_freq='epoch'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Load the best saved model
    model = tf.keras.models.load_model('cell_count_model.keras')

    # Evaluate the model on validation data
    loss, mae = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss}')
    print(f'Validation MAE: {mae}')

    # Predict on a few images
    predictions = model.predict(X_val[:5])

    for i in range(5):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(X_val[i].squeeze(), cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title(f'Predicted Count: {predictions[i][0]:.2f}')
        plt.axis('off')

        plt.show()

    # Save the model in the .keras format
    model.save('cell_count_model_final.keras')
