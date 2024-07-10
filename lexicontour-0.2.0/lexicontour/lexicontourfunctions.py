from pdf2image import convert_from_path
from pytesseract import pytesseract
import cv2
import os
import numpy as np
import pytesseract

def get_images(file_path):
    images = convert_from_path(file_path)
    pdf_to_image = []
    for i, image in enumerate(images):
        image_array = np.array(image)
        pdf_to_image.append(image_array)

    return pdf_to_image 

def get_dict(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"I did not find the file at {file_path}")
        exit()
    with open(file_path, "r+") as iof:
        print("File Found!")
        images = get_images(file_path)
        image_processing(images[0])
        text_regions = get_contours()
        return text_regions

def image_processing(img):
    img_copy = img.copy()
    k_length = np.array(img).shape[0]//150

    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray = 255-gray_img

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_length, 1))

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    #detect vertical lines
    img_temp1 = cv2.erode(gray, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    #detect horizontal lines
    img_temp2 = cv2.erode(gray, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=3)

    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    no_line_img = cv2.subtract(gray, 255 - img_final_bin)
    no_line_img =  255 - no_line_img

    ret, thresh1 = cv2.threshold(no_line_img, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    thresh1 = 255-thresh1
    cv2.imwrite("current.png",thresh1)
    return 

def get_contours():
    large = cv2.imread("current.png")
    small = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    sm = large.copy()

    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    cont, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    def get_contour_precedence(contour, cols):
        tolerance_factor = 10
        origin = cv2.boundingRect(contour)
        return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

    contours = sorted(cont, key=lambda x:get_contour_precedence(x, large.shape[1]))

    mask = np.zeros(bw.shape, dtype=np.uint8)

    text_regions = {}

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(large, (x, y), (x+w-1, y+h-1), (0, 255, 0), 0)
            cropped = sm[y:y+h , x:x+w ]
            h,w = cropped.shape[:2]
            h = h + (h//2)
            w = w + (w//2)

            resize_img = cv2.resize(cropped, (w,h))
            
            text = pytesseract.image_to_string(resize_img, lang = 'eng', config = '--psm 6').strip()

            origin  = cv2.boundingRect(contours[idx])
            text_regions[origin] = text
 
    cv2.imwrite("contours.png",large)
    return text_regions