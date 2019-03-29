from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config
import pandas as pd
import imgaug as ia

class Detection():
    def __init__(self, det_id, x1, y1, x2, y2, label, confidence):
        self.id = det_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.confidence = confidence
        self.bbox =  ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)

class Matches():
    def __init__(self, tp, fp, fn):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def get_above_thresh(self, iou_thresh, confidence_thresh):
        tps = []
        for t in self.tp:
            if t[2] > iou_thresh:
                tps.append(t)


        tps2 = []
        for t in tps:
            if t[1].confidence > confidence_thresh:
                tps2.append(t)
        fps = []
        for t in self.fp:
            if t[1].confidence > confidence_thresh:
                fps.append(t)
        return tps2, fps, self.fn

cfg = load_config("new_data")
viame_csv = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/tinyout.csv"

api = ArcticApi(cfg)
# dict of viame image name to detections
viame_detections = {}
with open(viame_csv) as f:
    lis=[line.split(',') for line in f]        # create a list of lists
    lis = lis[2:]
    for det in lis:
        det_id, img_name, frame_id, x1, y1, x2, y2, conf, _ = det[:9]

        det_id = int(det_id)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        conf = float(conf)
        label = "ERR"
        multilabels = det[9:]
        for i in xrange(0, len(multilabels), 2):
            label_conf = float(multilabels[i+1])
            if label_conf == conf:
                label = multilabels[i]
        d = Detection(det_id, x1, y1, x2, y2, label, conf)
        if not img_name in viame_detections:
            viame_detections[img_name] = []
        viame_detections[img_name].append(d)

missing = 0
res = {}
for det in viame_detections:
    path = cfg.rgb_dir+det
    if not path in api.rgb_images:
        missing += 1
        continue
    predicted = viame_detections[det]
    ground_truth = [hs.rgb_bb for hs in api.rgb_images[path].hotspots]
    # find matches
    matches = []
    FN = []
    for t in ground_truth:
        best_iou = 0
        best_match = None
        for p in predicted:
            iou = p.bbox.iou(t)
            if iou > best_iou:
                best_iou = iou
                best_match = p
        if best_match is None:
            hs = api.hsm.get_hs(t.hsId)
            if not hs.updated:
                continue
            if "removed" in hs.status:
                continue
            FN.append((t, best_match, best_iou))
        else:
            matches.append((t, best_match, best_iou))
    # filter out duplicate prediction matches by chosing highest confidence
    filter_matches = {}
    for m in matches:
        if not m[1].id in filter_matches:
            filter_matches[m[1].id] = m
        elif filter_matches[m[1].id][2] < m[2]:
            filter_matches[m[1].id] = m
    new_matches =[]
    for k in filter_matches:
        new_matches.append(filter_matches[k])


    matches = new_matches
    false_positives = []
    for p in predicted:
        found = False
        for m in matches:
            if m[1].id == p.id:
                found = True

        if not found:
            false_positives.append((None, p, 0))
    if not len(false_positives) + len(matches) == len(predicted):
        print("issue fp+matches!=predicted")
    r = Matches(matches, false_positives, FN)
    res[path] = r

fps = 0
tps = 0
fns = 0
for k in res:
    matches = res[k]
    tp, fp, fn = matches.get_above_thresh(.5, .8)
    tps += len(tp)
    fps += len(fp)
    fns += len(matches.fn)

precision = float(tps)/(tps+fps)
recall = float(tps)/(tps+fns)
print("precision %f" % precision)
print("recall %f" % recall)
print("FP %d  TP %d  FN %d" % (fps, tps, fns))
print("Missing %d images" % missing)