import cv2
import numpy as np
from imutils.perspective import four_point_transform


pathImage = "test_image.jpg"
scale = 0.5 # scale for text overlap
font = cv2.FONT_HERSHEY_SIMPLEX

WIDTH, HEIGHT = 1920, 1080


def image_processing(image):
    # convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


def draw_max_contour(image, frame): 
    # find largest contour (i.e. edges of document) and draw
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # find max contour
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
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, 2, (255, 0, 255), 5, cv2.LINE_AA)

def scan(webCamFeed):
    while True:
        if webCamFeed:
            cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            _, frame = cap.read()
            

        else:
            frame = cv2.imread(pathImage)

        frame_copy = frame.copy()
        document_contour = draw_max_contour(frame_copy, frame)

        cv2.imshow("border", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))

        # apply perspsective transform to straighten doc
        warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
        cv2.imshow("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))

        processed = image_processing(warped)
        # rm 10 pix from each side
        processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
        cv2.imshow("Processed", cv2.resize(processed, (int(scale * processed.shape[1]),
                                                    int(scale * processed.shape[0]))))


        # key events
        pressed_key = cv2.waitKey(1) & 0xFF
        
        if pressed_key == 27: # esc
            break
        elif pressed_key == ord('s'): # save scan
            cv2.imwrite("output/scanned.jpg", processed)
            center_text(frame, "Scan Saved")
            cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))
            cv2.waitKey(500)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    webCamFeed = True
    scan(webCamFeed)
