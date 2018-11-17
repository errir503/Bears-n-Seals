import os
import cv2
from random import randint

# can chose to prep one label with a hotspot identifier
def crop_hotspot(out_dir, delta_bb, hs, minShift = 0, maxShift=0, label=True):
    if hs.classIndex > 1:
        print("Skipping, not a seal")
        return

    img = hs.rgb.image
    classIndex = hs.classIndex
    id = hs.id
    rgb_bb_b = hs.rgb_bb_b
    rgb_bb_t = hs.rgb_bb_t
    rgb_bb_l = hs.rgb_bb_l
    rgb_bb_r = hs.rgb_bb_r
    if classIndex > 1:
        return

    imgh = img.shape[0]
    imgw = img.shape[1]

    # center points of bounding box
    center_y = (rgb_bb_b - rgb_bb_t) / 2
    center_x = (rgb_bb_r - rgb_bb_l) / 2

    tcrop = max(rgb_bb_t, 0)
    bcrop = max(rgb_bb_b, 0)
    bcrop = min(bcrop, imgh)
    lcrop = max(rgb_bb_l, 0)
    rcrop = max(rgb_bb_r, 0)
    rcrop = min(rcrop, imgw)

    if tcrop == 0:
        center_y += rgb_bb_t
    if lcrop == 0:
        center_x += rgb_bb_l

    dx, dy = random_shift(tcrop, bcrop, lcrop, rcrop, imgw, imgh, minShift, maxShift)

    lcrop += dx
    rcrop += dx
    bcrop += dy
    tcrop += dy
    center_y -= dy
    center_x -= dx

    tcrop = max(tcrop, 0)
    bcrop = max(bcrop, 0)
    bcrop = min(bcrop, imgh)
    lcrop = max(lcrop, 0)
    rcrop = max(rcrop, 0)
    rcrop = min(rcrop, imgw)

    crop_img = img[tcrop:bcrop, lcrop: rcrop]

    croph = crop_img.shape[0]
    cropw = crop_img.shape[1]

    cv2.circle(crop_img, (center_x, center_y), 5, (0, 255, 0), 2)
    cv2.rectangle(crop_img, (center_x-delta_bb, center_y-delta_bb), (center_x+delta_bb, center_y+delta_bb), (0, 255, 0), 2) #draw rect

    file_name = out_dir +  "/crop_" + id + "_" + str(classIndex)
    cv2.imwrite(file_name + ".jpg", crop_img)

    if classIndex == 0 or classIndex == 1:
        classIndex == 0

    if label:
        with open(file_name + ".txt", 'a') as file:
            file.write(str(classIndex) + " " + str((center_x + 0.0) / cropw) + " " +
                       str((center_y + 0.0) / croph) + " " +
                       str((delta_bb + 0.0) / cropw) + " " +
                       str((delta_bb + 0.0) / croph) + "\n")

        with open('training_list.txt', 'a') as file:
            file.write(os.getcwd() + "/" + file_name + ".jpg" + "\n")


def random_shift(topCrop, bottomCrop, leftCrop, rightCrop, w, h, minShift, maxShift):
    if maxShift == 0:
        return 0, 0

    dx = 0
    dy = 0

    # make dx
    if randint(0, 1) == 1:
        if leftCrop != 0 and randint(0, 1) == 1:
            dx -=  randint(minShift, maxShift)

        elif rightCrop != 0 and randint(0, 1) == 1:
            dx += randint(minShift, maxShift)

        # left crop outside image bounds
        if not leftCrop + dx > 0:
            dx = 0
        if not rightCrop + dx < w:
            dx = 0

    # make dy
    if randint(0, 1) == 1:
        if topCrop != 0 and randint(0, 1) == 1:
            dy -= randint(minShift, maxShift)

        elif bottomCrop != 0 and randint(0, 1) == 1:
            dy += randint(minShift, maxShift)

        # left crop outside iomage bounds
        if topCrop + dy < 0:
            dy = 0
        if bottomCrop + dx > h:
            dy = 0

    return dx, dy
