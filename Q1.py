#----------------------------------------------------------
# EC7212 - Computer Vision and Image Processing
# Name: Randula R.D.
# RegNo: EG/2020/4149
# Take Home Assignment 2
#----------------------------------------------------------


# Import required libraries
import numpy as np
import cv2

def generateImage(width, height):
    # Create a blank grayscale image with all pixels initialized to 0
    image = np.zeros((height, width), dtype=np.uint8)

    # Set the entire image to white (255)
    image[:, :] = 255

    # Add a gray square on the left side
    square_size = width // 2
    square_x = 0 
    square_y = (height - square_size) // 2
    image[square_y:square_y+square_size, square_x:square_x+square_size] = 128  # Gray color

    
    # Draw a black rectangle on the right side
    rect_x = 2*width//3
    rect_y = height//4
    rect_width = width//4
    rect_height = height//2
    
    cv2.rectangle(image, (rect_x, rect_y), 
                 (rect_x + rect_width, rect_y + rect_height), 50, -1)
    return image

def addGaussianNoise(image):
    # Define the mean and standard deviation for Gaussian noise
    mean = 0
    stddev = 50
    # Convert the image to float for accurate noise addition
    image_float = image.astype(np.float32)
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, size=image.shape).astype(np.float32)
    # Add noise to the image
    img_noised = image_float + noise
    # Clip the result to the valid range [0, 255] and convert back to uint8
    noisy_image = np.clip(img_noised, 0, 255).astype(np.uint8)
    return noisy_image

# Generate a synthetic image
generatedImage = generateImage(300,300)
cv2.imshow("Image with 3 Pixel Values", generatedImage)

# Add Gaussian noise to the generated image
noisyImage = addGaussianNoise(generatedImage)
cv2.imshow("Noise added Image", noisyImage)

# Apply Otsu's thresholding to the noisy image
_, otsuThreshold = cv2.threshold(noisyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the binarized result after Otsu's method
cv2.imshow("Otsu's Thresholding", otsuThreshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

