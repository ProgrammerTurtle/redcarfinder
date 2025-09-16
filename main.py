# main.py
import scanner
import detector
import cv2
import os
import time

# --- CONFIGURATION ---
# Define the geographic area to scan.
# Example: A bounding box covering downtown San Francisco.
# You can get bounding boxes from sites like https://boundingbox.klokantech.com/
BBOX_SAN_FRANCISCO = (-122.42, 37.77, -122.40, 37.79) # (min_lon, min_lat, max_lon, max_lat)
TARGET_BBOX = BBOX_SAN_FRANCISCO

# --- DIRECTORIES ---
DOWNLOADS_FOLDER = "downloads"
RESULTS_FOLDER = "red_car_results"

# Create results folder if it doesn't exist.
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
    
def run_bot():
    """Main function to run the scanning and detection process."""
    print("--- Starting Red Car Detection Bot ---")
    
    # 1. Use the scanner to get a list of image sequences in our target area.
    sequences = scanner.get_image_sequences(TARGET_BBOX)
    
    if not sequences:
        print("No image sequences found or API error. Exiting.")
        return

    total_sequences = len(sequences)
    red_cars_found_count = 0

    # 2. Loop through each sequence.
    for i, seq in enumerate(sequences):
        print(f"\n--- Processing Sequence {i+1}/{total_sequences} (ID: {seq['id']}) ---")
        
        # 3. Download the representative image for this sequence.
        image_path = scanner.download_image(seq, DOWNLOADS_FOLDER)
        
        if image_path:
            # 4. Use the detector to find red cars in the downloaded image.
            image_with_boxes, found_cars_info = detector.find_red_cars(image_path)
            
            # 5. If red cars are found, save the result.
            if found_cars_info:
                red_cars_found_count += len(found_cars_info)
                print(f"SUCCESS: Found {len(found_cars_info)} red car(s) in {image_path}!")
                
                # Save the image with bounding boxes to the results folder.
                result_filename = os.path.basename(image_path)
                result_path = os.path.join(RESULTS_FOLDER, result_filename)
                cv2.imwrite(result_path, image_with_boxes)
                print(f"Result image saved to {result_path}")
            else:
                print(f"No red cars found in {image_path}.")
        
        # Be a good internet citizen: pause briefly between downloads.
        time.sleep(1) 

    print("\n--- Scan Complete ---")
    print(f"Processed {total_sequences} image sequences.")
    print(f"Found a total of {red_cars_found_count} red cars.")
    print(f"Check the '{RESULTS_FOLDER}' directory for saved images.")

if __name__ == "__main__":
    run_bot()