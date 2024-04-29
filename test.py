import os
import cv2
import mediapipe as mp
import numpy as np

# Define paths
input_dir = 'new2_test'
output_dir = 'hand_not_removed_test'

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to count the number of images in a directory
def get_image_count(directory):
    """ Count the number of files in a directory. """
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

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

    # Create a black image of the same dimensions as the original image
    height, width, _ = image.shape
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Process the image using the hand landmark detector
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)

        # Check if any hand landmarks were detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check if the output directory has fewer than 450 images before saving
            if get_image_count(os.path.dirname(save_path)) < 450:
                cv2.imwrite(save_path, black_image)
                print("Saved annotated image to:", save_path)
            else:
                print("Image not saved, directory has reached the limit of 450 images.")
        else:
        # Check if the label is 'nothing' which indicates no hand expected
            if 'nothing' in image_path:
                if get_image_count(os.path.dirname(save_path)) < 450:
                    cv2.imwrite(save_path, black_image)
                    print("Saved blank image for 'nothing' label to:", save_path)
                else:
                    print("Image not saved, directory has reached the limit of 450 images.")

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
