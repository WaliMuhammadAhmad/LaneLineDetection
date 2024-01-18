
import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines, color=(255, 0, 0), thickness=10):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (800, 350), (600, 350)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line, right_line = [], []

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line], dtype=object), left_fit, right_fit

def calculate_lane_width(left_fit, right_fit, y_value):
    left_x = int((y_value - left_fit[1]) / left_fit[0])
    right_x = int((y_value - right_fit[1]) / right_fit[0])
    return abs(right_x - left_x)

def draw_lane_width(image, lane_width):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Lane Width: {lane_width} pixels', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

def analyze_road_surface(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    non_zero_pixels = cv2.countNonZero(thresholded)
    wet_threshold = 50000

    if non_zero_pixels > wet_threshold:
        print("Caution: Wet or icy road detected!")
        # Add your behavior adjustment logic here, such as slowing down or activating stability control.
    else:
        print("Road surface appears normal.")

# Image processing for a single image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Analyze road surface
analyze_road_surface(image)

# Continue with existing lane detection code
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines, left_fit, right_fit = average_slope_intercept(lane_image, lines)

# Draw detected lanes in different colors
left_color = (255, 0, 0)  # Red
right_color = (0, 0, 255)  # Blue
left_line_image = display_lines(lane_image, [averaged_lines[0]], color=left_color)
right_line_image = display_lines(lane_image, [averaged_lines[1]], color=right_color)
combo_image = cv2.addWeighted(lane_image, 0.8, left_line_image, 1, 0)
combo_image = cv2.addWeighted(combo_image, 1, right_line_image, 1, 0)

# Calculate lane width at a specific y-value (e.g., the bottom of the image)
lane_width = calculate_lane_width(left_fit[0], right_fit[0], lane_image.shape[0])
print(f"Lane width: {lane_width} pixels")

# Draw lane width on the image
draw_lane_width(combo_image, lane_width)

plt.imshow(combo_image)
plt.show()

cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines, left_fit, right_fit = average_slope_intercept(frame, lines)

    # Draw detected lanes in different colors
    left_line_image = display_lines(frame, [averaged_lines[0]], color=left_color)
    right_line_image = display_lines(frame, [averaged_lines[1]], color=right_color)
    combo_image = cv2.addWeighted(frame, 0.8, left_line_image, 1, 0)
    combo_image = cv2.addWeighted(combo_image, 1, right_line_image, 1, 0)

    # Calculate lane width at a specific y-value (e.g., the bottom of the image)
    lane_width = calculate_lane_width(left_fit[0], right_fit[0], frame.shape[0])
    print(f"Lane width: {lane_width} pixels")

    # Draw lane width on the image
    draw_lane_width(combo_image, lane_width)

    cv2.imshow("result", combo_image)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()