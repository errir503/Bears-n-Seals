import matplotlib.patches as mpatches
import os

import numpy as np
from PIL import Image

from src.VIAME.EVAL.model import DetectionResults
from src.arcticapi import ArcticApi
from src.arcticapi import HotSpot
from src.arcticapi.config import load_config
import matplotlib.pyplot as plt

from src.arcticapi.model.HotSpot import SpeciesList




def get_mean_confidences(tp, fp, fn):
    pass


IOU_THRESH = 0.2 # only matters for optical
OPTICAL_CONFIDENCE_THRESH = .8
THERMAL_CONFIDENCE_THRESH = .5
threshes = [.2 ,.4,.6,.8,.9,.95]
cfg = load_config("new_data")
rgb_csv = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/results_optical_fullyolov3.csv"
rgb_csv1 = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/results_optical_17tiny.csv"
rgb_csv2 = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/results_optical_17tiny_50000.csv"
rgb_csv3 = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/rgb_18/results_optical_18tiny_20000.csv"
rgb_csv4 = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/rgb_18/results_optical_18tiny_last.csv"
rgb_csv5 = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/rgb_18/results_optical_18tiny_48000.csv"
rgb_csv6 = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results/rgb_18/results_optical_18tiny_60000.csv"
rgb_ims = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_optical_images.txt"
ir_csv = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/results_thermal.csv"
ir_ims = "/home/yuval/Documents/XNOR/VIAME/build/install/examples/darknet/detectors/input_thermal_images.txt"
api = ArcticApi(cfg)
res = []
res.append(DetectionResults(rgb_ims, rgb_csv, "rgb", api))
res.append(DetectionResults(rgb_ims, rgb_csv1, "rgb", api))
res.append(DetectionResults(rgb_ims, rgb_csv2, "rgb", api))
res.append(DetectionResults(rgb_ims, rgb_csv3, "rgb", api))
res.append(DetectionResults(rgb_ims, rgb_csv4, "rgb", api))
res.append(DetectionResults(rgb_ims, rgb_csv5, "rgb", api))
res.append(DetectionResults(rgb_ims, rgb_csv6, "rgb", api))
# thermal_results = DetectionResults(ir_ims, ir_csv, "thermal", api)

plotmap = {}
for thresh in threshes:
    for optical_results in res:
        if not optical_results.output in plotmap:
            plotmap[optical_results.output] = [[],[],[]]

        print(optical_results.output)
        tp, fp, fn = optical_results.confidence_filter(thresh,.1)
        new_hotspots = []
        for f in fp:
            fp_dets = fp[f]
            copy_hs = api.rgb_images[f].hotspots[0]
            i = 0
            for det in fp_dets:
                type = 0 if "ringed" in det.label else 1
                type = SpeciesList[type]
                hs = HotSpot(copy_hs.id + "fp%d"%i, copy_hs.thermal_loc[0], copy_hs.thermal_loc[1],int(det.x1), int(det.y1), int(det.x2), int(det.y2), "Animal", type,
                             copy_hs.rgb, copy_hs.ir, copy_hs.timestamp, copy_hs.project_name, copy_hs.aircraft, int(det.y1), int(det.y2), int(det.x1), int(det.x2), False, confidence=str(det.confidence))
                new_hotspots.append(hs)
                # xpos, ypos, thumb_left, thumb_top, thumb_right, thumb_bottom, type, species_id, rgb
                # , ir, timestamp, project_name, aircraft,
                # updated_top = -1, updated_bot = -1, updated_left = -1, updated_right = -1,
                # updated = False, status = "none", confidence = "NA"):
                i+=1
        header = "id,color_image,thermal_image,hotspot_id,hotspot_type,species_id,species_confidence,fog,thermal_x," \
                         "thermal_y,color_left,color_top,color_right,color_bottom, updated_left, updated_top, updated_right, updated_bottom, " \
                         "updated, status"
        api.saveHotspots(new_hotspots, "/fast/fps/fps.csv", header)
        # for f in fp:
        #     fp_dets = fp[f]
        #     image = Image.open(f)
        #     open_cv_image = np.array(image)
        #     for det in fp_dets:
        #         cv2.rectangle(open_cv_image, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (255, 0, 0), 6)
        #     for hs in api.rgb_images[f].hotspots:
        #         cv2.rectangle(open_cv_image, (hs.rgb_bb.x1-2, hs.rgb_bb.y1-2), (hs.rgb_bb.x2-2, hs.rgb_bb.y2-2), (0, 255, 0), 6)
        #
        #     image = Image.fromarray(open_cv_image)
        #     image.save("/fast/fps/" + os.path.basename(f))



        precision = float(len(tp))/(len(tp) + len(fp))
        recall = float(len(tp))/(len(tp) + len(fn))
        f1 = 2* ((precision*recall)/(precision+recall))
        print("IOU_THRESH %.2f" % IOU_THRESH)
        # print("CONFIDENCE_THRESH %.2f" % CONFIDENCE_THRESH)
        print
        print("precision %.3f" % precision)
        print("recall %.3f" % recall)
        plotmap[optical_results.output][0].append(precision)
        plotmap[optical_results.output][1].append(recall)
        plotmap[optical_results.output][2].append(f1)
        print("FP %d  TP %d  FN %d" % (len(fp), len(tp), len(fn)))
        # avg_confidence_tp = np.average(tps_confidence_thresh)
        # stddev_confidence_tp = np.std(tps_confidence_thresh)
        # avg_iou_tp = np.average(tps_iou_thresh)
        # stddev_iou_tp = np.std(tps_iou_thresh)
        # print("True Positive confidence avg: %.3f  std: %.3f" % (avg_confidence_tp, stddev_confidence_tp))
        # print("True Positive iou avg: %.3f  std: %.3f" % (avg_iou_tp, stddev_iou_tp))

        print
        # print("Missing %d images" % missing)
colors = ['olive', 'red', 'blue', 'green', 'orange', 'yellow', "black", "gray"]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
legend_dict = {}
for idx, d in enumerate(plotmap):
    color = colors[idx]
    ys = plotmap[d]
    legend_dict[os.path.basename(d)] = color
    ax.plot(threshes, ys[0],  linestyle=(0, (1, 1)), color=color)
    ax.plot(threshes, ys[1],  linestyle=(0, ()), color=color)
    ax.plot(threshes, ys[2], linestyle=(0, (5, 5)), color=color)
patchList = []
for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)
plt.xlabel("Confidence Thresh")
plt.ylabel("Percent")
plt.legend(handles=patchList)
plt.show()
