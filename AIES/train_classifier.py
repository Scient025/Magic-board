import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image  # Modified import
import os

def binarize(img):
    img = image.img_to_array(img, dtype='uint8')
    binarized = np.expand_dims(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2), -1)
    inverted_binary_img = ~binarized
    return inverted_binary_img

data_dir = 'archive/data/extracted_images'

image_dir = 'AIES/equation_images/'

if not os.path.exists(data_dir):
    print(f"Directory does not exist: {data_dir}")
else:
    print(f"Using data directory: {data_dir}")

batch_size = 32
img_height = 45
img_width = 45

train_datagen = ImageDataGenerator(preprocessing_function=binarize)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    seed=123
)

class_names = [k for k, v in train_generator.class_indices.items()]
num_classes = len(class_names)

# Save class names to a text file for consistency
with open("class_names.txt", "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

model = tf.keras.Sequential([
    tf.keras.layers.Input((45, 45, 1)),  # Change this to 1 for grayscale
    layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),  # Increased filters
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),  # Increased filters
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_generator, epochs=5)

model.save('AIES/eqn-detect-model.keras')  # Or use .keras if you prefer

# Contour detection and processing function (no changes)
def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def detect_contours(img_path):
    input_image = cv2.imread(img_path, 0)
    input_image_cpy = input_image.copy()
    binarized = cv2.adaptiveThreshold(input_image_cpy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted_binary_img = ~binarized
    contours_list, hierarchy = cv2.findContours(inverted_binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    l = []
    for c in contours_list:
        x, y, w, h = cv2.boundingRect(c)
        l.append([x, y, w, h])
    
    lcopy = l.copy()
    keep = []
    while len(lcopy) != 0:
        curr_x, curr_y, curr_w, curr_h = lcopy.pop(0)
        if curr_w * curr_h < 20:
            continue
        throw = []
        for i, (x, y, w, h) in enumerate(lcopy):
            curr_interval = [curr_x, curr_x + curr_w]
            next_interval = [x, x + w]
            if getOverlap(curr_interval, next_interval) > 1:
                new_interval_x = [min(curr_x, x), max(curr_x + curr_w, x + w)]
                new_interval_y = [min(curr_y, y), max(curr_y + curr_h, y + h)]
                curr_x, curr_y, curr_w, curr_h = new_interval_x[0], new_interval_y[0], new_interval_x[1] - new_interval_x[0], new_interval_y[1] - new_interval_y[0]
                throw.append(i)
        for ind in sorted(throw, reverse=True):
            lcopy.pop(ind)
        keep.append([curr_x, curr_y, curr_w, curr_h])
    return keep

IMAGE = "2numbers_018.png"
img_path = image_dir + IMAGE
input_image = cv2.imread(img_path, 0)
input_image_cpy = input_image.copy()
keep = detect_contours(img_path)

for (x, y, w, h) in keep:
    cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 1)
plt.imshow(input_image_cpy, cmap='gray')
plt.show()

def resize_pad(img, size, padColor=255):
    h, w = img.shape[:2]
    sh, sw = size
    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
    aspect = w / h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

new_model = tf.keras.models.load_model('AIES/eqn-detect-model.keras')

eqn_list = []
IMAGE = "2numbers_048.png"
img_path = image_dir + IMAGE
input_image = cv2.imread(img_path, 0)
inverted_binary_img = binarize(input_image)

for (x, y, w, h) in sorted(keep, key=lambda x: x[0]):
    plt.imshow(inverted_binary_img[y:y + h, x:x + w])
    plt.show()
    img = resize_pad(inverted_binary_img[y:y + h, x:x + w], (45, 45), 0)  # Resize and pad
    plt.imshow(img)
    plt.show()

    # Get the predicted probabilities from the model
    predictions = new_model.predict(tf.expand_dims(tf.expand_dims(img, 0), -1))
    
    # Get the class index with the highest probability
    predicted_index = np.argmax(predictions)

    # Map the index to the actual class name
    if predicted_index < len(class_names):
        pred_class = class_names[predicted_index]
    else:
        pred_class = "unknown"

    # Handle the special case for multiplication
    if pred_class == "times":
        pred_class = "*"

    eqn_list.append(pred_class)

# Join the list to form the final equation string
eqn = "".join(eqn_list)
print("Predicted Equation:", eqn)  # This should print the final predicted equation
