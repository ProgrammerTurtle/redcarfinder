# scanner.py
import requests
import os

# The base URL for the KartaView API v2.
API_BASE_URL = "https://api.kartaview.com/v2/search/sequence"

def get_image_sequences(bounding_box, start_date="2023-01-01"):
    """
    Queries the KartaView API to find image sequences within a geographic area.
    
    Args:
        bounding_box (tuple): (min_lon, min_lat, max_lon, max_lat)
        start_date (str): The earliest date for photos (YYYY-MM-DD).
        
    Returns:
        A list of sequence data dictionaries, or None if the request fails.
    """
    min_lon, min_lat, max_lon, max_lat = bounding_box
    
    # API parameters: define the search area, date, and other filters.
    params = {
        'bbox': f"{min_lon},{min_lat},{max_lon},{max_lat}",
        'startDate': start_date,
        'limit': 500,  # Max limit per request
        'processed': 'true' # Only get sequences that have been processed
    }
    
    try:
        print(f"Querying KartaView API for bounding box: {bounding_box}...")
        response = requests.get(API_BASE_URL, params=params, timeout=20)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        
        data = response.json()
        print(f"Found {len(data['data'])} image sequences in the area.")
        return data['data']
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to KartaView API: {e}")
        return None

def download_image(sequence, download_folder="downloads"):
    """
    Downloads the first high-res image from a sequence.
    
    Args:
        sequence (dict): A sequence data dictionary from the API response.
        download_folder (str): Folder to save the image in.
        
    Returns:
        The file path of the downloaded image, or None if download fails.
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Get the URL for the highest resolution image (l = large).
    image_url = sequence['photos'][0]['url_l']
    sequence_id = sequence['id']
    lat = sequence['lat']
    lon = sequence['lon']

    # Create a unique filename that includes location data.
    filename = f"{sequence_id}_{lat:.5f}_{lon:.5f}.jpg"
    filepath = os.path.join(download_folder, filename)

    try:
        # Check if file already exists to avoid re-downloading.
        if os.path.exists(filepath):
            print(f"Image {filename} already exists. Skipping.")
            return filepath
            
        print(f"Downloading {image_url}...")
        img_data = requests.get(image_url, timeout=20).content
        
        with open(filepath, 'wb') as handler:
            handler.write(img_data)
        
        print(f"Saved image to {filepath}")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {image_url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None