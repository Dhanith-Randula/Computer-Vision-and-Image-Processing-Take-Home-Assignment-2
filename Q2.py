#----------------------------------------------------------
# EC7212 - Computer Vision and Image Processing
# Name: Randula R.D.
# RegNo: EG/2020/4149
# Take Home Assignment 2
#----------------------------------------------------------

# Load necessary libraries for image processing
import cv2
import numpy as np

# Function to display intermediate segmentation results
def show_segmentation(mask):
    cv2.imshow('Segmentation Process', mask)
    cv2.waitKey(1)  # Display window without blocking execution

# Function to perform region growing segmentation
def region_growing(image, seed_points, threshold_range):
    # Initialize binary mask for storing segmented regions (0 = background, 255 = foreground)
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Initialize processing queue with provided seed points
    queue = []
    for seed_point in seed_points:
        queue.append(seed_point)
    
    iteration = 0  # Initialize iteration counter
    
    # Main loop for region growing
    while queue:
        iteration += 1  # Track number of iterations

        # Retrieve and remove the first point from the queue
        current_point = queue.pop(0)

        # Get the intensity value of the current pixel
        current_value = image[current_point[1], current_point[0]]

        # Mark the current pixel in the mask as part of the segmented region
        mask[current_point[1], current_point[0]] = 255

        # Display the current segmentation state periodically
        if iteration % 10 == 0:  # Display every 10th iteration
            show_segmentation(mask)

        # Examine all 8-connected neighboring pixels
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Skip the center pixel (already processed)
                if i == 0 and j == 0:
                    continue

                # Compute coordinates of the neighboring pixel
                x = current_point[0] + i
                y = current_point[1] + j

                # Check if neighbor is within image bounds
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # Get the intensity value of the neighboring pixel
                    neighbor_value = image[y, x]

                    # If the intensity difference is within the threshold
                    if np.abs(neighbor_value - current_value) <= threshold_range:
                        # If the neighbor has not been visited yet
                        if mask[y, x] == 0:
                            # Add the neighbor to the queue for further processing
                            queue.append((x, y))
                            # Mark the neighbor as visited
                            mask[y, x] = 255

    # Return the final binary mask of the segmented region
    return mask

# Read input image in grayscale format
image = cv2.imread('C:/Users/busin/Desktop/MY/SEM 7/Com vision/Take Home 2/Computer-Vision-and-Image-Processing-Take-Home-Assignment-2/Input/tree.jpg', cv2.IMREAD_GRAYSCALE)

# Validate image loading
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Display image properties for debugging purposes
print(f"Image dimensions: {image.shape} (Height x Width)")
print(f"Image size: {image.shape[0]} rows x {image.shape[1]} columns")

# Set initial seed points for segmentation (updated to fit within current image dimensions)
seed_points = [(150, 100), (250, 150), (200, 300)]  # Modified seed locations to ensure they're valid

# Validate each seed point to ensure it's within image bounds
valid_seed_points = []
for seed in seed_points:
    x, y = seed
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        valid_seed_points.append(seed)
        print(f"Seed point ({x}, {y}) is valid")
    else:
        print(f"Warning: Seed point ({x}, {y}) is outside image boundaries")

# Update the seed points list to include only valid coordinates
seed_points = valid_seed_points

# Exit if no valid seed points remain
if len(seed_points) == 0:
    print("Error: No valid seed points found!")
    exit()

# Define intensity similarity threshold for region growing
threshold_range = 10

# Perform segmentation using region growing algorithm
segmented_image = region_growing(image, seed_points, threshold_range)

# Close any previously opened OpenCV windows
cv2.destroyAllWindows()

# Display the original grayscale image and final segmentation result
cv2.imshow('GrayScale Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
