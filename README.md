# 🧠 Image Segmentation & Thresholding

This repository contains the solution for **Computer Vision and Image Processing Take Home Assignment 2** .

---

## 🔖 Assignment Details

- **Course:** EC7212 – Computer Vision and Image Processing 
- **Name:** Randula R.D.  
- **Reg No:** EG/2020/4149  
- **Semester:** 07  
- **Assignment:** Take Home Assignment 2  

---

## 📌 Contents

###  Question 1 – Otsu’s Thresholding
- A synthetic image with two objects (square and rectangle) and three pixel intensities was generated using NumPy and OpenCV.
- Gaussian noise (mean = 0, stddev = 50) was added to the image.
- A **manual implementation of Otsu’s thresholding** algorithm was developed and applied.
- Results were compared with OpenCV’s built-in Otsu thresholding method.
- Outputs: Generated image with 3 pixel values, Noise added image, Otsu's threshold applied image.

###  Question 2 – Region Growing Segmentation
- Grayscale image loaded using OpenCV.
- Seed points manually selected.
- Custom region growing algorithm implemented to segment regions similar to the seed pixel intensities.
- Real-time visualization included to show segmentation progress.
- Final output is a binary mask showing the segmented region.

---

You can install the dependencies via:

```bash
pip install opencv-python numpy
