import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os

# Parameters
img_width, img_height = 32, 32
batch_size = 64
epochs = 20
input_shape = (32, 32, 3)
train_directory = './train'
test_directory = './test'

# Load the data
data = pd.read_csv('train_labels.csv')

# Assuming the correct columns are 'id' and 'label'
filenames = data['id'].values
labels = data['label'].values

# Split the data into training and validation sets
train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, test_size=0.1, random_state=1975)

# train_filenames = train_filenames[:10]
# val_filenames = val_filenames[:10]
# train_labels = train_labels[:10]
# val_labels = val_labels[:10]

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(file_path, label):
    img = Image.open(file_path.numpy().decode('utf-8'))
    img = img.resize((64,64))
    img = img.crop((16,16,48,48))
    img = np.array(img) / 255.0
    return img, label

def load_and_preprocess_image_map(file_path, label):
    img, label = tf.py_function(load_and_preprocess_image, [file_path, label], [tf.float32, tf.int64])
    img.set_shape((img_width, img_height, 3))
    label.set_shape([])
    return img, label

# Function to create dataset
def create_dataset(filenames, labels, directory, batch_size, is_training=True):
    file_paths = [os.path.join(directory, fname + '.tif') for fname in filenames]
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_and_preprocess_image_map, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Create training and validation datasets
train_ds = create_dataset(train_filenames, train_labels, train_directory, batch_size, is_training=True)
val_ds = create_dataset(val_filenames, val_labels, train_directory, batch_size, is_training=False)

# Load pre-trained MobileNet model + higher level layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Combine the base model with custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['AUC'])

# Train the model
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# Load test data (only filenames)
test_filenames = [f.split('.')[0] for f in os.listdir(test_directory) if f.endswith('.tif')]
# test_filenames = test_filenames[:10]

# Create test dataset
test_labels = np.zeros(len(test_filenames), dtype=np.int64)
test_ds = create_dataset(test_filenames, test_labels, test_directory, batch_size)

# Generate predictions for the test set
predictions = model.predict(test_ds)

# Binarize the predictions
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Create a DataFrame with the IDs and predicted labels
submission = pd.DataFrame({'id': test_filenames, 'label': predicted_labels})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
