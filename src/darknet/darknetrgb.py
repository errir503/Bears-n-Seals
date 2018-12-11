import sys
from ctypes import *
import random
import cv2
import numpy as np
import os
import time
cfg_file = sys.argv[1]
cfg_file = sys.argv[2]
weight_file = sys.argv[3]
out_csv_file = sys.argv[4]

# cfg_file = b"cfg/sealsv3test.cfg"
# data_file = b"cfg/seals.data"
# weight_file = b"seal_weights/sealsv3_4000.weights"
# out_csv_file = "possible_color.csv"
pred_thres = 0.5

def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net, meta, image, thresh=pred_thres, hier_thresh=pred_thres, nms=.45):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                            (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


def tile_image(image):
    tilew = 800
    tileh = 800

    imgh = image.shape[0]  # image height
    imgw = image.shape[1]  # image width
    remainder_w = imgw % tilew  # image remainder for width
    remainder_h = imgh % tileh  # image remainder for height
    cw = imgw / tilew  # number of tiles width
    ch = imgh / tileh  # number of tiles height

    tiles = []
    # Tile image
    for y in range(0, ch + 1):
        for x in range(0, cw + 1):
            top = tileh * y
            bot = tileh * y
            if y != ch:  # if last row then use the remainder height
                bot += tileh
            else:
                bot += remainder_h
            left = tilew * x
            right = tilew * x
            if x != cw:
                right += tilew
            else:
                right += remainder_w  # if last col use remainder width

            cropped_img = image[top: bot, left: right]

            tiles.append((cropped_img, (top, bot, left, right)))

    return tiles


if __name__ == "__main__":
    files = []

    with open(sys.argv[1], 'r') as my_file:
        files = my_file.read().splitlines()

    # net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    # meta = load_meta(b"cfg/coco.data")

    net = load_net(cfg_file, weight_file, 0)
    meta = load_meta(data_file)
    start_el = int(sys.argv[2])
    files = files[start_el:]
    i = start_el
    for file_name in files:
        basename = os.path.splitext(os.path.basename(file_name))[0]
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        if img is None:
            print("Could not read file " + file_name)
            continue
        start = time.time()
        tiles = tile_image(img)
        end = time.time()

        print(str(i) + " File: " + file_name)
        print("Crop Time: " + str(end - start))
        start = time.time()
        for tile in tiles:
            detections = []
            r = detect(net, meta, tile[0])
            for k in range(len(r)):
                if len(r) > 0:
                    x = r[k][2][0]
                    y = r[k][2][1]
                    pred = r[k][1]
                    width = r[k][2][2]
                    height = r[k][2][3]
                    top, bot, left, right = tile[1]

                    print "Name: ", r[k][0], "Predict %: ", pred, "X: ", x, "Y: ", y, "W: ", width, "H: ", height, '\n'
                    detections.append((r[k], tile[1], file_name))
                    csv_row = str(i) + "," + file_name + "," + str(pred) + "," + str(x) + "," + str(y) + "," + \
                              str(width) + "," + str(height) + "," + \
                              str(top) + "," + str(bot) + "," + str(left) + "," + str(right) + ",UNCHECKED"+"\n"

                    with open(out_csv_file, 'a') as fd:
                        if os.stat(out_csv_file).st_size == 0:
                            fd.write(
                                "fnum,file_name,prediction,local_x,local_y,bbox_width,bbox_height,crop_top,crop_bot,crop_left,crop_right,status\n")
                        fd.write(csv_row)

                    cv2.imwrite("res/" + basename + "_" + str(tile[1][0]) + "_" + str(tile[1][2]) + ".jpg", tile[0])

            del tile
        end = time.time()
        total_time = end - start
        print("Detect Time: " + str(total_time) + " (" + str(len(tiles)) + " frames)  " + str(round(len(tiles)/total_time,2)) + "fps")
        print("")
        i += 1
        del tiles
        del img
    i += 1
