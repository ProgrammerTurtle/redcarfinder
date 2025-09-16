# detector.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model once when the module is imported.
# This is more efficient than loading it every time we call a function.
try:
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

def _is_red(image_crop):
    """Internal function to analyze the color of a specific image crop."""
    # Convert the image from BGR to HSV color space for better color analysis.
    hsv_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Define the HSV color ranges for red. Red wraps around the 0/180 mark,
    # so we need two separate ranges.
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks that capture only the pixels within the red ranges.
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Calculate the percentage of red pixels in the crop.
    red_pixel_count = cv2.countNonZero(red_mask)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]
    if total_pixels == 0:
        return False
    
    red_percentage = (red_pixel_count / total_pixels) * 100

    # If over 20% of the pixels in the crop are red, we classify it as a red car.
    # This threshold can be adjusted for sensitivity.
    return red_percentage > 20

def find_red_cars(image_path):
    """
    Main function to find red cars in an image file.
    
    Args:
        image_path (str): The path to the image file.

    Returns:
        A tuple: (original_image_with_boxes, found_red_cars_info)
        - The first element is the image with red cars highlighted.
        - The second is a list of dictionaries, each containing info about a found car.
    """
    if model is None:
        print("YOLO model not loaded. Cannot perform detection.")
        return None, []

    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Failed to load image at {image_path}")
            return None, []
    except Exception as e:
        print(f"Error reading image file {image_path}: {e}")
        return None, []

    # Let YOLOv8 detect all objects in the frame.
    results = model(frame, verbose=False)[0] # verbose=False cleans up the output
    class_names = results.names
    found_red_cars_info = []

    # Loop through all detected objects.
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        # We are only interested in objects classified as 'car' or 'truck'.
        if class_name in ['car', 'truck']:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            # Crop the image to get only the detected car.
            car_crop = frame[y1:y2, x1:x2]

            # Analyze the crop to see if it's red.
            if _is_red(car_crop):
                # If it's red, draw a green box and label on the original image.
                label = "Red Car"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Store the coordinates of this red car.
                found_red_cars_info.append({'class': class_name, 'box': (x1, y1, x2, y2)})
                
    return frame, found_red_cars_info