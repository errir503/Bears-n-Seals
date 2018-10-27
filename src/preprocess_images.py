import os
import sys
import NOAA
from util import LabelParser as parser, visuals
from util.image_registration import *

config = {
    "output_dir": "images/results/",
    "offset": 80
}

def main():
    rows = list()

    f = open(sys.argv[1], 'r')
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)
    f.close()
    del rows[0]  # remove col headers

    global res_path
    res_path = sys.argv[2]
    hsm = make_hotspots(rows)
    del rows
    #visuals.show_ir(hsm)
    register_images(hsm)
    # crop_hotspots(hsm)
    # align_images(hotspots)

def crop_hotspots(hsm):
    # Check if output directory exists, if not create
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    i = 0
    for hs in hsm.hotspots:
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
        #cv2.rectangle(crop_img, (center_x-config["offset"], center_y-config["offset"]), (center_x+config["offset"], center_y+config["offset"]), (0, 255, 0), 2) #draw rect

        img_name = config["output_dir"]+"crop_" + hs.id
        cv2.imwrite(img_name + ".jpg", crop_img)

        with open(img_name + ".txt", 'a') as file:
            file.write(str(hs.classIndex) + " " + str((center_x + 0.0) / cropw) + " " +
                       str((center_y + 0.0) / croph) + " " +
                       str((config["offset"] + 0.0) / cropw) + " " +
                       str((config["offset"] + 0.0)/ croph) + "\n")

        with open('training_list.txt', 'a') as file:
            file.write(os.getcwd() + "/" + img_name + ".jpg" + "\n")

        hs.rgb.free()


def make_hotspots(rows):
    hsm = NOAA.HotSpotMap()

    for row in rows:
        hotspot = parser.parse_hotspot(row, res_path)
        hsm.add(hotspot)
    return hsm




# Call main function
if __name__ == '__main__':
    main()
