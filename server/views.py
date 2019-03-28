import json
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import numpy as np
import argparse
import cv2
import time
from PIL import Image
import pytesseract
from . import config, image_manipulation
import os
# from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt


@csrf_exempt
def monobrand(request):
    responseData = {
        'id': 4,
        'name': 'Test Response',
        'roles': ['Admin', 'User']
    }
    if request.method == 'POST' and request.FILES['img']:
        print('HAHAHAHAHAHAHAAHAHAHAHAH')
        file = request.FILES['img']
        print(file.name)           # Gives name
        file.content_type   # Gives Content type text/html etc
        file.size           # Gives file's size in byte
        file.read()         # Reads file
        fs = FileSystemStorage()
        filename = fs.save('./images/'+file.name, file)
        phones = detect_from_monobrand('./images/'+file.name)

    return JsonResponse(phones)


def detect_from_monobrand(path_to_image):
    img = cv2.imread(path_to_image, 0)
    (h, w) = img.shape[:2]
    if h > w:
        # if height of image is bigger than width - rotate 90 degree
        img = image_manipulation.rotate_bound(img, 90, (w//2, h//2))
    # Resize image for the ratio
    img = cv2.resize(img, (config.MONOBRAND_TEMPLATE_WIDTH,
                           config.MONOBRAND_TEMPLATE_HEIGHT))
    (h, w) = img.shape[:2]
    threshold = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 121, 32)
    # Get rotation angle
    needed_rotate_angle = image_manipulation.get_rotate_angle(
        threshold, config.MONOBRAND_TEMPLATE)
    # rotate initaial image and threshold
    img = image_manipulation.rotate(
        img, needed_rotate_angle, config.MONOBRAND_TEMPLATE_CENTER)
    threshold = image_manipulation.rotate(
        threshold, needed_rotate_angle, config.MONOBRAND_TEMPLATE_CENTER)

    # TRANSLATE X FIRST TIME
    needed_translate_x = image_manipulation.get_x_translate(
        threshold, config.MONOBRAND_PHONE_MASK)

    M = np.float32([[1, 0, int(needed_translate_x)], [0, 1, 0]])
    threshold = cv2.warpAffine(threshold, M, (w, h))
    img = cv2.warpAffine(img, M, ((w, h)))

    # ROTATE ONE MORE TIME
    final_needed_angle = image_manipulation.get_nearest_angle(
        threshold, config.MONOBRAND_TEMPLATE, (config.MONOBRAND_PHONE_X + config.MONOBRAND_PHONE_MASK_HALF_W, config.MONOBRAND_PHONE_Y + config.MONOBRAND_PHONE_MASK_HALF_H), -5, 5, 0.1)
    print('needed_rotate_angle', needed_rotate_angle)
    print('final_needed_angle', final_needed_angle)
    threshold = image_manipulation.rotate(threshold, final_needed_angle, (config.MONOBRAND_PHONE_X +
                                                                          config.MONOBRAND_PHONE_MASK_HALF_W, config.MONOBRAND_PHONE_Y + config.MONOBRAND_PHONE_MASK_HALF_H))
    img = image_manipulation.rotate(img, final_needed_angle, (config.MONOBRAND_PHONE_X +
                                                              config.MONOBRAND_PHONE_MASK_HALF_W, config.MONOBRAND_PHONE_Y + config.MONOBRAND_PHONE_MASK_HALF_H))
    # TRANSLATE X ONE MORE TIME
    threshold = image_manipulation.get_x_translate(
        threshold, config.MONOBRAND_PHONE_MASK, return_image=True)
    img = image_manipulation.get_x_translate(
        img, config.MONOBRAND_PHONE_MASK, return_image=True)

    max_black = 255
    min_black = 1
    for i in img:
        for j in i:
            if j < max_black:
                max_black = j
            elif j > min_black:
                min_black = j
    print('max_black', max_black)
    print('min_black', min_black)

    border = int(min_black) - int(max_black)
    print('border:', border)
    border //= 4
    print('border:', border)

    img = cv2.resize(img, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)

    phone_number_mask = cv2.resize(
        config.MONOBRAND_PHONE_MASK, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)

    final_image = cv2.bitwise_and(img, phone_number_mask)
    mask = cv2.bitwise_not(phone_number_mask)

    f = cv2.bitwise_xor(mask, final_image)
    f = cv2.bitwise_not(f)
    r, custom_thresh = cv2.threshold(f, border, 255, cv2.THRESH_BINARY)
    custom_text = image_manipulation.get_text(custom_thresh)
    f_text = image_manipulation.get_text(f).replace('/', '7')
    print("F without chars", custom_text)
    print("Splitted text:", f_text)
    cv2.imshow('phone_number_mask', phone_number_mask)
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('f', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return {'f_text': f_text, 'custom_text': custom_text}
