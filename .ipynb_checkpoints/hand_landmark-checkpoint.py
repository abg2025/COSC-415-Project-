import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.image as mpimg

# Define paths
input_dir = 'new_train'
output_dir = 'hand_train'

# Initialize MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Function to process and annotate images
def process_and_annotate_image(image_path, save_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image:", image_path)
        return

    # Convert BGR to RGB as MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ensure the image is in uint8 format
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)


    # Prepare the image for MediaPipe
    try:
        # Create a packet with image data and format for MediaPipe
        # Process the image using the hand landmark detector
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)
    except Exception as e:
        print(f"Failed to process image {image_path}: {str(e)}")
        return

    # Draw the hand landmarks on the image
    annotated_image = draw_landmarks_on_image(image_rgb.copy(), detection_result)

    # Convert RGB back to BGR for saving
    #annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Save the annotated image
    #cv2.imwrite(save_path, annotated_image)
    #print("Saved annotated image to:", save_path)




# Function to draw landmarks on image
def draw_landmarks_on_image(image, results):
    # Check if any hand landmarks were detected
    if results.hand_landmarks:
        # Draw landmarks for each hand
        for hand_landmarks in results.hand_landmarks:
            # hand_landmarks is already the correct format: List[NormalizedLandmark]
            # Draw landmarks on the image using the correct connections for hands
            mp.solutions.drawing_utils.draw_landmarks(
                image, 
                hand_landmarks,  # This should be a list of NormalizedLandmark
                mp.solutions.hands.HAND_CONNECTIONS)
    return image



# Walk through input directory and process each image
for subdir, dirs, files in os.walk(input_dir):
    for file in files:
        full_input_path = os.path.join(subdir, file)
        relative_path = os.path.relpath(full_input_path, input_dir)
        full_output_path = os.path.join(output_dir, relative_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        # Process and annotate image
        process_and_annotate_image(full_input_path, full_output_path)

