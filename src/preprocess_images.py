import os
import sys
import cv2
import numpy as np
import csv
import random
import NOAA

# configs
config = {
    "tile_size": {
        "width": 512,
        "height": 512
    }
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

    tile_and_label(hsm)


def tile_and_label(hsm):
    # Check if output directory exists, if not create
    if not os.path.exists("tiles/"):
        os.mkdir("tiles/")

    processed_images = 0
    skipped_images = 0

    tilew = config["tile_size"]["width"]
    tileh = config["tile_size"]["height"]
    for id in hsm.hotspots:
        hs = hsm.hotspots[id]



        if hs.classIndex == 4 or hs.classIndex == 3:
            # if anomaly or NA skip for now
            skipped_images+=1
            continue

        if (not hs.rgb.load_image()):
            skipped_images+=1
            print("Failed to load images for hotspot" + hs.id)
            continue

        imgh = hs.rgb.image.shape[0]  # image height
        imgw = hs.rgb.image.shape[1]  # image width
        remainder_w = imgw % tilew   # image remainder for width
        remainder_h = imgh % tileh  # image remainder for height
        cw = imgw/tilew  # number of tiles width
        ch = imgh/tileh  # number of tiles height

        # center points of bounding box
        center_y = (hs.rgb_bb_b - hs.rgb_bb_t) / 2 + hs.rgb_bb_t
        center_x = (hs.rgb_bb_r - hs.rgb_bb_l) / 2 + hs.rgb_bb_l

        # Tile image
        processed_images += 1
        for y in range(0,ch+1):
            for x in range(0, cw+1):
                top = tileh * y
                bot = tileh * y
                if y != ch:  # if last row then use the remainder height
                    bot += tileh
                else:
                    bot += remainder_h
                left = tilew * x
                right = tilew * x
                if  x != cw:
                    right += tilew
                else:
                    right += remainder_w  # if last col use remainder width

                cropped_img = hs.rgb.image[top: bot, left: right]

                img_name = "tiles/tile_" + id

                # recalculate bounding box in tile

                if center_x > left and center_x < right and center_y < bot and center_y > top:
                    # save image
                    tile_center_x = center_x - x * tilew
                    tile_center_y = center_y - y * tileh
                    cv2.circle(hs.rgb.image, (center_x, center_y), 5, (0, 255, 0), 4)
                    cv2.imwrite(img_name + ".jpg", cropped_img)
                    # create label
                    with open(img_name + ".txt", 'a') as file:
                        file.write(str(hs.classIndex) + " " + str(tile_center_x) + " " +
                                   str(tile_center_y) + " " +
                                   str((right - left) / tilew) + " " +
                                   str((bot - top) / tileh))
                    # add image to training list
                    with open('training_list.txt', 'a') as file:
                        file.write(img_name + ".jpg" + "\n")

                #TODO pick random empty images to train on as well maybe one tile per image

        hs.rgb.free()


        i = 0


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
        rgb = NOAA.Image(res_path+row[IMG_RGB_COL_IDX], "rgb")
        therm8 = NOAA.Image(res_path+row[IMG_THERMAL8_COL_IDX], "therm8")
        therm16 = NOAA.Image(res_path+row[IMG_THERMAL16_COL_IDX], "therm16")

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
