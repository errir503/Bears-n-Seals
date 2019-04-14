import sys
from ctypes import *
import math
import random
import cv2
import time
import numpy as np


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
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



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
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


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
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

def norm_matrix(m):
    if m < 0:
        return 0
    if m > 1:
        return 1
    return m

def lin_normalize_image(image_array, bit_8, bottom=None, top=None):
    """Linear normalization for an image array
    Inputs:
        image_array: np.ndarray, image data to be normalized
        bit_8: boolean, if true outputs 8 bit, otherwise outputs 16 bit
        bottom: float, value to map to 0 in the new array
        top: float, value to map to 2^(bit_depth)-1 in the new array
    Output:
        scaled_image: nd.ndarray, scaled image between 0 and 2^(bit_depth) - 1
    """

    if bottom is None:
        # bottom = np.min(image_array)
        bottom = np.percentile(image_array,1)
    if top is None:
        # top = np.max(image_array)
        top = np.percentile(image_array,100)
    print top
    scaled_image = (image_array - bottom + 0.0) / (top - bottom + 0.0)
    scaled_image = np.vectorize(norm_matrix)(scaled_image)

    if bit_8:
        scaled_image = np.floor(scaled_image * 255).astype(np.uint8)  # Map to [0, 2^8 - 1]
    else:
        scaled_image = np.floor(scaled_image * 65535).astype(np.uint16)  # Map to [0, 2^16 - 1]

    return scaled_image


if __name__ == "__main__":
    files = []
    detections = []
    with open(sys.argv[1], 'r') as my_file:
        files = my_file.read().splitlines()

    # net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    # meta = load_meta(b"cfg/coco.data")

    net = load_net(b"cfg/irtest.cfg", b"seal_weights/ir_900.weights", 0)
    meta = load_meta(b"cfg/seals.data")

    i = 0
    while i < len(files):
        i += 1
        file = files[i]

        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
        if img is None:
            print("Could not read file " + file)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.uint16)
        img = lin_normalize_image(img, True)
        print np.max(img)
        print np.min(img)
        r = detect(net, meta, img)
        if len(r) > 0 and r[0] > 0.3:
            r.append(file)
            detections.append(r)
            print r
        # for i in r:
        #     x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
        #     print (x,y,w,h)
        #     xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        #     pt1 = (xmin, ymin)
        #     pt2 = (xmax, ymax)
        #     cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        #     cv2.putText(img, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)
        cv2.imwrite("res/res"+str(i)+".jpg", img)
        del img

