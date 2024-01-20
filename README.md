# Lane Line Detection

Lane Line Detection is a Python project using OpenCV to detect and visualize lane lines in images and videos. The project also includes functionality to analyze road surface conditions.

## Features

- **Lane Detection:** Utilizes Canny edge detection and Hough line transform to identify and draw lane lines on images and videos.
- **Road Surface Analysis:** Evaluates road surface conditions based on pixel intensity in a thresholded image.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/WaliMuhammadAhmad/LaneLineDetection.git
   cd LaneLineDetection
   ```
2. Install Dependencies:

```bash
pip install opencv-python numpy matplotlib
```
3. Run Lane Detection on an Image:

```bash
python main.py
```
4. Run Lane Detection on a Video:

```bash
python main.py --video
```
Press 'q' to exit the video preview.

5. Road Surface Analysis
The project includes an additional feature to analyze road surface conditions based on pixel intensity in a thresholded image.

## Examples
Image Processing:

```bash
Copy code
python main.py
```
Video Processing:

```bash
Copy code
python main.py --video
```

## Contributions
Feel free to contribute, report issues, or suggest improvements!
