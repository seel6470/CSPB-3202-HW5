import pandas as pd
import matplotlib as plt
import os
from PIL import Image
import matplotlib.pyplot as plt

data = pd.read_csv('train_labels.csv', dtype=str)

data['id'] = data['id'] + '.tif'  # Add file extension

# Select a random subset of 256 images
subset = data.sample(n=256, random_state=1975)

train_directory = './train'

# Create lists to store image widths and heights
widths = []
heights = []

# Iterate over the subset to get image dimensions
for image_file in subset['id']:
    image_path = os.path.join(train_directory, image_file)
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
    else:
        print(f"Image file {image_file} does not exist.")

# Plot Histogram for Widths
plt.figure(figsize=(12, 6))
plt.hist(widths, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Width (pixels)')
plt.ylabel('Number of Images')
plt.title('Histogram of Image Widths for 256 Random Samples')
plt.tight_layout()
plt.savefig('image_widths_histogram.png')

# Plot Histogram for Heights
plt.figure(figsize=(12, 6))
plt.hist(heights, bins=20, color='salmon', edgecolor='black')
plt.xlabel('Height (pixels)')
plt.ylabel('Number of Images')
plt.title('Histogram of Image Heights for 256 Random Samples')
plt.tight_layout()
plt.savefig('image_heights_histogram.png')