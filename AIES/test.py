import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

def binarize(img):
    img = image.img_to_array(img, dtype='uint8')
    binarized = np.expand_dims(cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2), -1)
    inverted_binary_img = ~binarized
    return inverted_binary_img

def detect_contours(img):
    inverted_binary_img = binarize(img)
    contours_list, _ = cv2.findContours(inverted_binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for c in contours_list:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 20 and w < 100 and h < 150:  # Filter out small and large bounding boxes
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

def resize_pad(img, size, padColor=255):
    h, w = img.shape[:2]
    sh, sw = size

    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
    aspect = w / h

    if aspect > 1:  # Horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # Vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # Square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

# Load your model
model_path = r'C:\Users\Anil Cerejo\OneDrive\Desktop\AIES\Magic-board\eqn-detect-model.keras'
model = tf.keras.models.load_model(model_path)

# Class names from your training generator
data_dir = r'C:\Users\Anil Cerejo\OneDrive\Desktop\AIES\data1'
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

# Process a specific image
IMAGE = "equation.png"  # Change this to your target image file
image_path = os.path.join(r"C:\Users\Anil Cerejo\OneDrive\Desktop\AIES", IMAGE)
# Read and process the image
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if input_image is None:
    print(f"Error: Unable to load image at path {image_path}")
else:
    inverted_binary_img = binarize(input_image)
    plt.imshow(input_image, cmap='gray')
    plt.title('Original Image')
    plt.show()
    
    plt.imshow(inverted_binary_img, cmap='gray')
    plt.title('Inverted Binary Image')
    plt.show()

    # Detect contours and get bounding boxes
keep = detect_contours(input_image)

eqn_list = []

for (x, y, w, h) in sorted(keep, key=lambda x: x[0]):
        plt.imshow(inverted_binary_img[y:y+h, x:x+w], cmap='gray')
        plt.title('Detected Region')
        plt.show()

        # Resize and pad the cropped image for model input
        img = resize_pad(inverted_binary_img[y:y+h, x:x+w], (45, 45), 0)

        plt.imshow(img, cmap='gray')
        plt.title('Resized and Padded Image')
        plt.show()

        # Expand dimensions for model prediction
        img_expanded = np.expand_dims(img, axis=0)  # Add batch dimension
        img_expanded = np.expand_dims(img_expanded, axis=-1)  # Add channel dimension

        # Predict class
        pred_class = class_names[np.argmax(model.predict(img_expanded))]
        if pred_class == "times":
            pred_class = "*"  # Replace "times" with "*"
        
        eqn_list.append(pred_class)

eqn = "".join(eqn_list)
print(f"Detected Equation: {eqn}")