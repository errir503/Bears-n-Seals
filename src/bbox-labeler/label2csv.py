import copy
import os

import cv2

from arcticapi import ArcticApi
from arcticapi.visuals import drawBBoxYolo

# Using the existing CSV goes through all labels in the Images directory and updates the existing CSV file
# with updates from the labels
csv = '/Users/yuval/Documents/XNOR/bounding-box-labeler-yolo/_CHESS_ImagesSelected4Detection.csv'
# csv = '/Users/yuval/Documents/XNOR/bounding-box-labeler-yolo/out.csv'
img_path = '/Users/yuval/Documents/XNOR/Bears-n-Seals/images/CHESS/'
output_csv = 'out.csv'
img_paths = ["Images", "relabel"]

# csv = 'out.csv'
api = ArcticApi(csv, img_path)
hotspots = api.hsm.hotspots

idx2row = {}
for idx, hs in enumerate(hotspots):
    idx2row[hs.id] = idx


crop_txt_files = []
for img_path in img_paths:
    for file in os.listdir(img_path):
        if file.endswith(".2label"):
            no_ext = os.path.splitext(file)[0]
            ids = no_ext.split("-")[0].split("_")[1:]
            crop_txt_files.append({'ids': ids, 'path': os.path.join(img_path, file)})

for file in crop_txt_files:
    ids = file['ids']
    ids_original_len = len(ids)
    if ids_original_len < 1:
        print("File has no ids: " + file["path"])
        continue

    lines = []
    with open(file['path']) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    # sometime I remove bad labels or remove labels from an image we don't want to update those
    # and we set the status to removed
    if len(lines) < len(ids):
        for id in ids:
            found = False
            for line in lines:
                items = line.split(" ")[0]
                if id == items[0]:
                    found = True
            if not found:
                idx = idx2row[id]
                hs = hotspots[idx]
                hs.status = "removed"

    # go through each line and calculate the global coordinates
    for line in lines:
        items = line.split(" ")
        idx = None
        if not items[0] in idx2row:
            id = str(int(float(items[0])))
            found = None
            for csvid in idx2row:
                if id in csvid:
                    found = csvid
            if found is None:
                print("Failed to find " + id)
                continue
            else:
                id = found
            idx2row[items[0]] = len(hotspots)
            idx = idx2row[id]
            newhs = copy.copy(hotspots[idx])
            newhs.id = items[0]
            newhs.status = "new"
            hotspots.append(newhs)

        idx = idx2row[items[0]]
        hs = hotspots[idx]
        x = float(items[2])
        y = float(items[3])
        w = float(items[4])
        h = float(items[5])
        t = int(items[6])
        b = int(items[7])
        l = int(items[8])
        r = int(items[9])
        chipw = r - l
        chiph = b - t
        bboxw = w * chipw
        bboxh = h * chiph
        centerx = l + (chipw * x)
        centery = t + (chiph * y)


        left = centerx - bboxw/2
        right = centerx + bboxw/2
        top = centery + bboxh/2
        bot = centery - bboxh/2
        hs.updated_top = int(top)
        hs.updated_bot = int(bot)
        hs.updated_left = int(left)
        hs.updated_right = int(right)
        hs.updated = True

        # debug
        if False and x != 0.5:
            if hs.rgb.load_image():
                img = hs.rgb.image
                (yolox, yoloy, yolow, yoloh) = hs.getYoloBBox()
                drawBBoxYolo(img, yolox, yoloy, yolow, yoloh, hs.classIndex)
                # pltIm(img)
                cv2.imwrite('outimages/'+ hs.id + ".jpg", img)
                hs.rgb.free()

        hotspots[idx] = hs
header = "hotspot_id,timestamp,filt_thermal16,filt_thermal8,filt_color,x_pos,y_pos,thumb_left,thumb_top,thumb_right," \
         "thumb_bottom,hotspot_type,species_id,updated_bot,updated_top,updated_left,updated_right,updated,status"

api.saveHotspotsToCSV(output_csv, header)