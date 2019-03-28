import cv2
import numpy as np
MONOBRAND_TEMPLATE_PATH = 'C:/projects/archieve/server/templates/monobrand_template.jpg'
MONOBRAND_PHONE_MASK_PATH = 'C:/projects/archieve/server/templates/monobrand_phone_mask.jpg'

MONOBRAND_TEMPLATE = cv2.imread(MONOBRAND_TEMPLATE_PATH, 0)
MONOBRAND_PHONE_MASK = cv2.imread(MONOBRAND_PHONE_MASK_PATH, 0)

(MONOBRAND_PHONE_MASK_H,
 MONOBRAND_PHONE_MASK_W) = MONOBRAND_PHONE_MASK.shape[:2]
MONOBRAND_PHONE_MASK_HALF_W = MONOBRAND_PHONE_MASK_W // 2
MONOBRAND_PHONE_MASK_HALF_H = MONOBRAND_PHONE_MASK_H // 2


(MONOBRAND_TEMPLATE_HEIGHT,
 MONOBRAND_TEMPLATE_WIDTH) = MONOBRAND_TEMPLATE.shape[:2]

MONOBRAND_TEMPLATE_CENTER = (
    MONOBRAND_TEMPLATE_WIDTH // 2, MONOBRAND_TEMPLATE_HEIGHT // 2)

# template contours
contours, _ = cv2.findContours(
    MONOBRAND_TEMPLATE, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

MONOBRAND_PHONE_X, MONOBRAND_PHONE_Y, MONOBRAND_PHONE_W, MONOBRAND_PHONE_H = cv2.boundingRect(
    biggest_contour)
MONOBRAND_PHONE_MASK = cv2.resize(
    MONOBRAND_PHONE_MASK, (MONOBRAND_PHONE_W, MONOBRAND_PHONE_H))
print('MONOBRAND_PHONE_X', MONOBRAND_PHONE_X)
print('MONOBRAND_PHONE_Y', MONOBRAND_PHONE_Y)
print('MONOBRAND_PHONE_H', MONOBRAND_PHONE_H)
print('MONOBRAND_PHONE_W', MONOBRAND_PHONE_W)
