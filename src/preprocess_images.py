import os
import sys
import cv2
import numpy as np
import csv
import random
import NOAA

config = {
    "output_dir": "results/"
}

def main():
    rows = list()

    f = open(sys.argv[1], 'r')
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)
    f.close()

    global res_path
    res_path = sys.argv[2]

    hsm = create_hotspot_map(rows)
    del rows

    crop_hotspots(hsm)


def crop_hotspots(hsm):
    # Check if output directory exists, if not create
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    i = 0
    for id in hsm.hotspots:
        hs = hsm.hotspots[id]
        print("Cropping hotspot:" + str(id) + " -" + str(round((i+0.0)/len(hsm.hotspots), 2))) + "% complete"
        i += 1

        foundEmptyHS = False

        if hs.classIndex == 4:
            # if anomaly or NA skip for now
            continue

        if (not hs.rgb.load_image()):
            print("Failed to load images for hotspot" + hs.id)
            continue
        imgh = hs.rgb.image.shape[0]
        imgw = hs.rgb.image.shape[1]

        # center points of bounding box
        center_y = (hs.rgb_bb_b - hs.rgb_bb_t) / 2
        center_x = (hs.rgb_bb_r - hs.rgb_bb_l) / 2

        topCrop = max(hs.rgb_bb_t, 0)
        bottomCrop = max(hs.rgb_bb_b, 0)
        bottomCrop = min(bottomCrop, imgh)
        leftCrop = max(hs.rgb_bb_l, 0)
        rightCrop = max(hs.rgb_bb_r, 0)
        rightCrop = min(rightCrop, imgw)

        if topCrop == 0:
            center_y += hs.rgb_bb_t
        if leftCrop == 0:
            center_x += hs.rgb_bb_l

        crop_img = hs.rgb.image[topCrop:bottomCrop, leftCrop: rightCrop]

        croph = crop_img.shape[0]
        cropw = crop_img.shape[1]

        # cv2.circle(crop_img, (center_x, center_y), 5, (0, 255, 0), 2)

        img_name = config["output_dir"]+"crop_" + id
        cv2.imwrite(img_name + ".jpg", crop_img)

        with open(img_name + ".txt", 'a') as file:
            file.write(str(hs.classIndex) + " " + str((center_x + 0.0) / cropw) + " " +
                       str((center_y + 0.0) / croph) + " " +
                       str(40.0 / cropw) + " " +
                       str(40.0 / croph) + "\n")

        with open('training_list.txt', 'a') as file:
            file.write(os.getcwd() + "/" + img_name + ".jpg" + "\n")

        hs.rgb.free()


def create_hotspot_map(rows):
    # Column index for each attribute in the given data
    HOTSPOT_ID_COL_IDX = 0
    TIMESTAMP = 1
    IMG_THERMAL8_COL_IDX = 2
    IMG_THERMAL16_COL_IDX = 3
    IMG_RGB_COL_IDX = 4
    XPOS_IDX = 5
    YPOS_IDX = 6
    LEFT_IDX = 7
    TOP_IDX = 8
    RIGHT_IDX = 9
    BOT_IDX = 10
    HOTSPOT_TYPE_COL_IDX = 11
    SPECIES_ID_COL_IDX = 12

    hsm = NOAA.HotSpotMap()

    del rows[0]  # remove col headers
    for row in rows:
        rgb = NOAA.Image(res_path + row[IMG_RGB_COL_IDX], "rgb")
        therm8 = NOAA.Image(res_path + row[IMG_THERMAL8_COL_IDX], "therm8")
        therm16 = NOAA.Image(res_path + row[IMG_THERMAL16_COL_IDX], "therm16")

        hotspot = NOAA.HotSpot(row[HOTSPOT_ID_COL_IDX], int(row[XPOS_IDX]), int(row[YPOS_IDX]), int(row[LEFT_IDX]),
                               int(row[TOP_IDX]),
                               int(row[RIGHT_IDX]), int(row[BOT_IDX]), row[HOTSPOT_TYPE_COL_IDX],
                               row[SPECIES_ID_COL_IDX], rgb, therm8, therm16,
                               row[TIMESTAMP])

        hsm.add([rgb, therm8, therm16], hotspot)
    return hsm


# Call main function
if __name__ == '__main__':
    main()
