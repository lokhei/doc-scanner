import cv2
import numpy as np
import argparse
import os
from imutils.perspective import four_point_transform

WIDTH, HEIGHT = 1920, 1080
SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

def image_processing(image):
    """
    Convert image to grayscale and apply thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return threshold

def draw_max_contour(image, frame):
    """
    Find and draw the largest contour (document edges) on the image.
    """
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the largest contour with 4 points
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)
    return document_contour

def center_text(image, text):
    """
    Draw centered text on the image.
    """
    text_size = cv2.getTextSize(text, FONT, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), FONT, 2, (255, 0, 255), 5, cv2.LINE_AA)


def save_image(image, filename):
    """
    Save the processed image.
    """
    if not os.path.exists('output'):
        os.makedirs('output')
    cv2.imwrite(filename, image)

def scan(webCamFeed, pathImage):
    """
    Main scanning function.
    """
    cap = None
        

    while True:
        if webCamFeed:
            cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                break
        else:
            frame = cv2.imread(pathImage)
            if frame is None:
                print(f"Failed to load image from {pathImage}")
                break

        frame_copy = frame.copy()
        document_contour = draw_max_contour(frame_copy, frame)

        cv2.imshow("border", cv2.resize(frame, (int(SCALE * WIDTH), int(SCALE * HEIGHT))))

        # apply perspsective transform to straighten doc
        warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
        cv2.imshow("Warped", cv2.resize(warped, (int(SCALE * warped.shape[1]), int(SCALE * warped.shape[0]))))

        processed = image_processing(warped)
        # rm 10 pix from each side
        processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
        cv2.imshow("Processed", cv2.resize(processed, (int(SCALE * processed.shape[1]),
                                                    int(SCALE * processed.shape[0]))))

        # Key events
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == 27:  # ESC key
            break
        elif pressed_key == ord('s'):  # Save scan
            save_image(processed, "output/scanned.jpg")
            center_text(frame, "Scan Saved")
            cv2.imshow("Border", cv2.resize(frame, (int(SCALE * WIDTH), int(SCALE * HEIGHT))))
            cv2.waitKey(500)

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Scanner using OpenCV")
    parser.add_argument('--webcam', action='store_true', help="Use webcam feed")
    parser.add_argument('--image', type=str, default="test_image.jpg", help="Path to the image file")
    args = parser.parse_args()

    scan(args.webcam, args.image)
