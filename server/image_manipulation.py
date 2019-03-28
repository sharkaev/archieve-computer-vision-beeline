import numpy as np
import argparse
import cv2
import time
from PIL import Image
import pytesseract
import os
from . import config
from collections import Counter
# from keras.models import load_model
# model = load_model('mnist_keras_cnn_model.h5')
# construct the argument parse and parse the arguments



def findBoxes(img):
    # find boxes
    box_template = cv2.imread('box.jpg', 0)
    w, h = box_template.shape[::-1]
    # Calculate Moments
    moments = cv2.moments(im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * \
            copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    cv2.imshow('boxes', img)
    print(result)


def skew_it(img):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.bitwise_not(img)
    # cv2.imshow('gray', gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(
        gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # # show the output image
    # print("[INFO] angle: {:.3f}".format(angle))

    return rotated


def get_digits_from_crop(img):
    cv2.imwrite('test.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img = Image.open('test.jpg')

    text = pytesseract.image_to_string(img, lang='rus')
    print(text)


def rotate(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
    return result


def rotate_bound(image, angle, center):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def get_text(img, psm=7):
    # 0    Orientation and script detection (OSD) only.
    # 1    Automatic page segmentation with OSD.
    # 2    Automatic page segmentation, but no OSD, or OCR.
    # 3    Fully automatic page segmentation, but no OSD. (Default)
    # 4    Assume a single column of text of variable sizes.
    # 5    Assume a single uniform block of vertically aligned text.
    # 6    Assume a single uniform block of text.
    # 7    Treat the image as a single text line.
    # 8    Treat the image as a single word.
    # 9    Treat the image as a single word in a circle.
    # 10    Treat the image as a single character.
    # 11    Sparse text. Find as much text as possible in no particular order.
    # 12    Sparse text with OSD.
    # 13    Raw line. Treat the image as a single text line,
    i = Image.fromarray(img)
    text = pytesseract.image_to_string(
        i, lang='Arial', config='--psm '+str(psm)+' --oem 2 ')

    return text

def get_max_white(img):
    max_black = 0
    for i in img:
        for j in i:
            if j > max_black:
                max_black = j
    return max_black

def split_image(img):
    (h, w) = img.shape[:2]
    box_width = w//9
    # print('box height', h)
    splitted_array = []
    current_step = 0
    for i in range(9):
        # print(i)
        splitted_array.append(img[0:h, current_step:current_step+box_width])
        current_step += box_width
    return splitted_array


def get_nearest_angle(image, template, rotation_point, start_angle, end_angle, step):
    
    angle = start_angle
    raiting = {}
    # cv2.namedWindow('nearest', cv2.WINDOW_NORMAL)
    while angle <= end_angle:

        rotated = rotate(image, angle, rotation_point)
        bitwise_and = cv2.bitwise_and(rotated, template)
        # cv2.imshow('nearest', bitwise_and)
        cv2.waitKey(1)
        raiting[f'{angle}'] = cv2.countNonZero(bitwise_and)
        angle = round(angle + step, 4)

    max_white = None
    needed_rotate_angle = 'None'
    
    for i in raiting.keys():

        if max_white == None or raiting[i] >= max_white:
            max_white = raiting[i]
            needed_rotate_angle = i
    
    # For Demonstration ONLY
    # print('Amount of max rotate angle', raiting[needed_rotate_angle])
    # rotated = rotate(image, float(needed_rotate_angle), rotation_point)
    # bitwise_and = cv2.bitwise_and(rotated, template)
    # cv2.imshow('nearest', bitwise_and)
    # cv2.waitKey(0)
    
    return float(needed_rotate_angle)


def get_rotate_angle(image, template):
    
    needed_rotate_angle = get_nearest_angle(
        image, template, config.MONOBRAND_TEMPLATE_CENTER, -4.00, 6.00, 0.20)
    needed_rotate_angle = get_nearest_angle(
        image, template, config.MONOBRAND_TEMPLATE_CENTER, needed_rotate_angle-2.00, needed_rotate_angle+2.00, 0.1)
    needed_rotate_angle = get_nearest_angle(
        image, template, config.MONOBRAND_TEMPLATE_CENTER, needed_rotate_angle - 0.10, needed_rotate_angle + 0.10, 0.01)
    needed_rotate_angle = get_nearest_angle(
        image, template, config.MONOBRAND_TEMPLATE_CENTER, needed_rotate_angle - 0.010, needed_rotate_angle + 0.010, 0.001)
    return float(needed_rotate_angle)

def resize_initial_img_to_template(img):
    (h, w) = img.shape[:2]
    
    if h > w:
        # if height of image is bigger than width - rotate 90 degree
        img = image_manipulation.rotate_bound(img, 90, (w//2, h//2))
    # Resize image for the ratio

    ratio = config.MONOBRAND_TEMPLATE_WIDTH // w
    needed_height = h * ratio
    img = cv2.resize(img, (config.MONOBRAND_TEMPLATE_WIDTH,
                           config.MONOBRAND_TEMPLATE_HEIGHT))
    return img

def get_x_translate(img, template, max_step=5, return_image=False):
    (th, tw) = img.shape[:2]
    translateX = -50
    translate_max_white = None
    needed_translate_x = 0
    
    while translateX <= max_step:
        M = np.float32([[1, 0, translateX], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (tw, th))
        dst = dst[config.MONOBRAND_PHONE_Y:config.MONOBRAND_PHONE_Y +
                                    config.MONOBRAND_PHONE_H, config.MONOBRAND_PHONE_X:config.MONOBRAND_PHONE_X+config.MONOBRAND_PHONE_W]
        
        bitwise_and = cv2.bitwise_and(dst, template)
        cv2.imshow('bitwise_and', bitwise_and)
        cv2.imshow('t', template)
        cv2.waitKey(75)
        current_white_step = cv2.countNonZero(bitwise_and)
        if translate_max_white is None or current_white_step > translate_max_white:
            needed_translate_x = translateX
        
        translateX += 1
    
    # print(needed_translate_x)
    M = np.float32([[1, 0, translateX], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (tw, th))
    dst = dst[config.MONOBRAND_PHONE_Y:config.MONOBRAND_PHONE_Y + config.MONOBRAND_PHONE_H, config.MONOBRAND_PHONE_X:config.MONOBRAND_PHONE_X+config.MONOBRAND_PHONE_W]
    cv2.imshow('dst', dst)
    bitwise_and = cv2.bitwise_and(dst, template)
    cv2.imshow('bitwise_and', bitwise_and)
    cv2.waitKey(0)
    print(needed_translate_x)
    if return_image:
        return bitwise_and
    return needed_translate_x
def get_splitted_text(img):
    splitted_text = ""
    numbers = split_image(img)
    i = 0
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    for number in numbers:
        # number = cv2.filter2D(number, -1, kernel)
        # add Gauss
        # number = addWeighted(number, 1.5, number, -0.5, 0, number)
        # number = cv2.blur(number,(2,2))
        max_white_in_number =  get_max_white(number)
        print('max white', max_white_in_number)
        # r, number = cv2.threshold(number, 18, 255, cv2.THRESH_TOZERO)
        im2, contours, hierarchy = cv2.findContours(
            number, cv2.RETR_TREE  , cv2.CHAIN_APPROX_SIMPLE  )
        
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(cnt) > 1:
            for contour in cnt[1:]:
                cv2.drawContours(number, [contour], 0,0,-1)
        x,y,w,h = cv2.boundingRect(cnt[0])
        # draw the book contour (in green)
        
        n = get_text(number, psm=10)
        splitted_text += str(n)
        # For Demonstration ONLY
        # cv2.rectangle(number,(x,y),(x+w,y+h),255,2)
        # cv2.putText(number,str(n), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255)
        # cv2.drawContours(number, contours, -1, 255, 3)
        # cv2.imshow(str(i), number)
        # print(i, n)
        i += 1
    return splitted_text