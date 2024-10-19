import cv2

image_path = 'archive/extracted_images/2/2numbers_040.png'
print(f"Trying to read image from: {image_path}")
input_image = cv2.imread(image_path)