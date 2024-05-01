# Lane Line Detection

Lane Line Detection is a Python project for detecting and visualizing lane lines in images and videos using OpenCV and computer vision techniques.

## Overview

The project implements various image processing and computer vision algorithms to detect lane lines on roads in images and videos. It includes functionalities such as edge detection, region of interest selection, Hough line detection, averaging detected lines, and lane width calculation.

## Features

- Edge detection using the Canny edge detector
- Region of interest selection to focus on the lane area
- Hough line detection for finding lines in the image
- Averaging and extrapolating detected lines to generate lane lines
- Calculation of lane width based on detected lane lines

## How to get started


```bash
git clone https://github.com/WaliMuhammadAhmad/LaneLineDetection.git
```

Navigate to the project directory:

```bash
cd LaneLineDetection
```

### Install the required dependencies:

```bash
pip install numpy matplotlib opencv-python
```

### Running the Code

1. Run the main script:

```bash
python main.py
```
This script will process the test image included in the repository, detect lane lines, and display the result.

3. To process a video, replace `"test2.mp4"` with the path to your video file in the `cap = cv2.VideoCapture("test2.mp4")` line of the code and run the script.

4. Press the 'q' key to exit the video playback.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bug fixes.