import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

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

data['id'] = data['id'] + '.tif'  # Add file extension
(data['label'].value_counts() / len(data)).to_frame().sort_index().T

filenames = data['id'].values
labels = data['label'].values

# Split the data into training and validation sets
train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, test_size=0.2, random_state=42)

# Create data augmentation generators
train_datagen = ImageDataGenerator(rescale=1/255)

validation_datagen = ImageDataGenerator(rescale=1/255)

# Create training and validation datasets
train_df = pd.DataFrame({'id': train_filenames, 'label': train_labels})
val_df = pd.DataFrame({'id': val_filenames, 'label': val_labels})

print("Train DataFrame head:")
print(train_df.head())

print("Validation DataFrame head:")
print(val_df.head())

train_ds = train_datagen.flow_from_dataframe(
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

val_ds = validation_datagen.flow_from_dataframe(
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

# Check the contents of the datasets
print("Number of batches in train_ds:", len(train_ds))
print("Number of batches in val_ds:", len(val_ds))

# Load pre-trained EfficientNetB0 model + higher level layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

cnn = Sequential([
    Conv2D(16, (3,3), activation = 'relu', padding = 'same', input_shape=(32,32,3)),
    Conv2D(16, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),
    
    Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
    Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    BatchNormalization(),

    Flatten(),
    
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(2, activation='softmax')
])

opt = tf.keras.optimizers.Adam(0.001)
cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])

# Train the model
h1 = cnn.fit(
    x = train_ds, 
    steps_per_epoch = len(train_ds),
    epochs = 40,
    validation_data = val_ds, 
    validation_steps = len(val_ds), 
    verbose = 1
)
history = h1.history
h2 = cnn.fit(
    x = train_ds, 
    steps_per_epoch = len(train_ds), 
    epochs = 30,
    validation_data = val_ds, 
    validation_steps = len(val_ds), 
    verbose = 1
)
h3 = cnn.fit(
    x = train_ds, 
    steps_per_epoch = len(train_ds), 
    epochs = 20,
    validation_data = val_ds, 
    validation_steps = len(val_ds), 
    verbose = 1
)

for k in history.keys():
    history[k] += h3.history[k]

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
plt.savefig('graphs.png')

cnn.save('HCDv01.h5')
pickle.dump(history, open(f'HCDv01.pkl', 'wb'))

# Create test dataset
test_filenames = [f for f in os.listdir(test_directory)]
#test_filenames = test_filenames[:100]

test_labels = np.zeros(len(test_filenames), dtype=np.int64)
test_df = pd.DataFrame({'id': test_filenames, 'label': test_labels})
print(test_df.head())

test_ds = validation_datagen.flow_from_dataframe(
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

# Generate predictions for the test set

predictions = cnn.predict(test_ds)
# Binarize the predictions
predicted_labels = np.argmax(predictions, axis=1)
# Create a DataFrame with the IDs and predicted labels
submission = pd.DataFrame({'id': test_filenames, 'label': predicted_labels})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
