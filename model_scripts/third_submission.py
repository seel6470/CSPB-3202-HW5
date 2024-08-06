import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.debugging.set_log_device_placement(True)

# output image size after pre-processing
img_width, img_height = 32, 32
batch_size = 64
epochs = 1
input_shape = (32, 32, 3)
train_directory = './train'
test_directory = './test'

data = pd.read_csv('train_labels.csv', dtype=str)
# data = data.head(100) # for testing

data['id'] = data['id'] + '.tif'  # Add file extension

filenames = data['id'].values
labels = data['label'].values

# Split the data into training and validation sets
train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, test_size=0.2, random_state=75)

# Create training and validation datasets
train_df = pd.DataFrame({'id': train_filenames, 'label': train_labels})
val_df = pd.DataFrame({'id': val_filenames, 'label': val_labels})

# Define a preprocessing function to crop the center 32x32 pixels
def center_crop(image):
    center = (image.shape[0] // 2, image.shape[1] // 2)
    half_crop_size = img_width // 2
    cropped_image = image[center[0] - half_crop_size:center[0] + half_crop_size, center[1] - half_crop_size:center[1] + half_crop_size]
    return cropped_image

# create image data generators, normalizing RGB values
train_generator = ImageDataGenerator(rescale=1/255, preprocessing_function = center_crop)
validation_generator = ImageDataGenerator(rescale=1/255, preprocessing_function = center_crop)

train_dataset = train_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=train_directory,
    x_col='id',
    y_col='label',
    batch_size=batch_size,
    seed=1,
    shuffle=True,
    class_mode='categorical',
    target_size=(32, 32)
)

val_dataset = validation_generator.flow_from_dataframe(
    dataframe=val_df,
    directory=train_directory,
    x_col='id',
    y_col='label',
    batch_size=batch_size,
    seed=1,
    shuffle=False,
    class_mode='categorical',
    target_size=(32, 32)
)

# Load pre-trained EfficientNetB0 model + higher level layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

cnn = Sequential([
    base_model,

    # 16 filters capture low level features (e.g. edges)
    Conv2D(64, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2, padding='same'),
    Dropout(0.5),
    BatchNormalization(),
    
    # 128 filters capture high-level abstract features
    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2, padding='same'),
    Dropout(0.5),
    BatchNormalization(),

    GlobalAveragePooling2D(),
    Flatten(),
    
    # final fully connected layers
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(2, activation='softmax')
])

opt = tf.keras.optimizers.Adam(0.001)
cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch < 80:
        return lr * 0.5
    else:
        return lr * 0.1

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train the model
model = cnn.fit(
    x = train_dataset, 
    steps_per_epoch = len(train_dataset),
    epochs = 100,
    validation_data = val_dataset, 
    validation_steps = len(val_dataset), 
    verbose = 1,
    callbacks = [lr_callback]
)

history = model.history

epoch_range = range(1, len(history['loss'])+1)

plt.figure(figsize=[14,4])
plt.subplot(1,3,1)
plt.plot(epoch_range, history['loss'], label='Training')
plt.plot(epoch_range, history['val_loss'], label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
plt.legend()
plt.subplot(1,3,2)
plt.plot(epoch_range, history['accuracy'], label='Training')
plt.plot(epoch_range, history['val_accuracy'], label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
plt.legend()
plt.subplot(1,3,3)
plt.plot(epoch_range, history['auc'], label='Training')
plt.plot(epoch_range, history['val_auc'], label='Validation')
plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.title('AUC')
plt.legend()
plt.tight_layout()
plt.savefig('third_graphs.png')

# Create test dataset
test_filenames = [f for f in os.listdir(test_directory)]
# test_filenames = test_filenames[:50] # for testing

test_labels = np.zeros(len(test_filenames), dtype=np.int64)
test_df = pd.DataFrame({'id': test_filenames, 'label': test_labels})
print(test_df.head())

test_dataset = validation_generator.flow_from_dataframe(
    dataframe=test_df,
    directory=test_directory,
    x_col='id',
    y_col='label',
    batch_size=batch_size,
    seed=1,
    shuffle=False,
    class_mode=None,
    target_size=(32, 32)
)

predictions = cnn.predict(test_dataset)
# predictions are presented as a list of tuples with probabilities for each category
# e.g. [0.5671,0.4329]
# the actual category will be equal to the index of the maximum element in the tuplepredicted_labels = np.argmax(predictions, axis=1)
predicted_labels = np.argmax(predictions, axis=1)
# Create a DataFrame with the IDs and predicted labels
submission = pd.DataFrame({'id': test_filenames, 'label': predicted_labels})

# Save the DataFrame to a CSV file
submission.to_csv('third_submission.csv', index=False)
