import os

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

class DetectionResults():
    def __init__(self, input, output, type, api):
        self.api = api
        self.api_images = api.rgb_images if type == "rgb" else api.ir_images
        self.type = type
        self.output = output
        self.viame_detections = {}
        last_added = ""
        with open(output) as f:
            base_path = self.api.cfg.rgb_dir if type == "rgb" else self.api.cfg.ir_dir
            lis = [line.split(',') for line in f]  # create a list of lists
            lis = lis[2:]
            for det in lis:
                det_id, img_name, frame_id, x1, y1, x2, y2, conf, _ = det[:9]
                img_name = base_path+img_name
                last_added = img_name

                det_id = int(det_id)
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                conf = float(conf)
                label = "ERR"
                multilabels = det[9:]
                for i in xrange(0, len(multilabels), 2):
                    label_conf = float(multilabels[i + 1])
                    if label_conf == conf:
                        label = multilabels[i]
                d = Detection(det_id, x1, y1, x2, y2, label, conf)
                if not img_name in self.viame_detections:
                    self.viame_detections[img_name] = []
                self.viame_detections[img_name].append(d)

        # last added, remove rest from input images
        with open(input) as f:
            self.input_images_fullpath = f.readlines()
        self.input_images_fullpath = [x.strip() for x in self.input_images_fullpath]
        last_index = self.input_images_fullpath.index(last_added)

        self.input_images_fullpath = self.input_images_fullpath[:last_index]
        self.input_images = [os.path.basename(path) for path in self.input_images_fullpath]

        # create ground truth image lists
        self.ground_truth_background = []
        self.ground_truth_seal = []
        self.ground_truth_other = []
        for im in self.input_images_fullpath:
            if im not in self.api_images:
                self.ground_truth_background.append(im)
                continue
            if im not in self.viame_detections:
                # print(im)
                continue
            hasSeal = False
            for hs in self.api_images[im].hotspots:
                if hs.classIndex < 2:
                    hasSeal = True
                    break

            if hasSeal:
                self.ground_truth_seal.append(im)
            else:
                self.ground_truth_other.append(im)

        # filter out duplicate predictions by iou
        removed_ct = 0
        for k in self.viame_detections:
            duplicates = {}
            predictions = self.viame_detections[k]
            filtered_predictions = []

            for p in predictions:
                is_duplicate = False
                for p2 in predictions:
                    if not p.id == p2.id:
                        if p.bbox.iou(p2.bbox) > .8 or p2.bbox.iou(p.bbox) > .8:
                            is_duplicate = True
                            if p.id in duplicates or p2.id in duplicates:
                                continue
                            best = p if p.bbox.area > p2.bbox.area else p2
                            duplicates[best.id] = best

                if not is_duplicate:
                    filtered_predictions.append(p)


            for d in duplicates:
                filtered_predictions.append(duplicates[d])

            removed_ct += len(predictions) - len(filtered_predictions)
            self.viame_detections[k] = filtered_predictions
        print("removed %d duplicate bboxes > .8 iou" % removed_ct)



        self.tp = {}
        self.fp = {}
        self.fn = {}

        if type == "rgb":
            for im in self.viame_detections:
                if not im in self.ground_truth_seal:
                    continue
                skip = False
                for hs in self.api_images[im].hotspots:
                    if not hs.updated:
                        skip = True
                if skip:
                    continue

                detections = self.viame_detections[im]
                ground_truth = [hs.rgb_bb for hs in api.rgb_images[im].hotspots
                                if not "removed" in hs.status]


                truth_to_det = {}
                for t in ground_truth:
                    best_iou = 0
                    best_match = None
                    for det in detections:
                        if im in self.ground_truth_seal:
                            iou = det.bbox.iou(t)
                            if iou > best_iou:
                                best_iou = iou
                                best_match = det
                    truth_to_det[t.hsId] = best_match

                for det in detections:
                    best_iou = 0
                    best_match = None
                    for t in ground_truth:
                        iou = det.bbox.iou(t)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = det


                    if best_match is None:
                        if not im in self.fp:
                            self.fp[im] = []
                        self.fp[im].append(det)

                for t in truth_to_det:
                    res = truth_to_det[t]
                    if res is None:
                        if not im in self.viame_detections:
                            continue
                        if not im in self.fn:
                            self.fn[im] = []
                        hs = api.hsm.get_hs(t)
                        self.fn[im].append(hs)
                    else:
                        if not im in self.tp:
                            self.tp[im] = []
                        self.tp[im].append((res, t))

            for im in self.ground_truth_seal:
                if not im in self.tp:
                    if im not in self.fn:
                            self.fn[im] =[]
                            for hs in self.api_images[im].hotspots:
                                self.fn[im].append(hs)

        else:
            for im in self.viame_detections:
                detections = self.viame_detections[im]
                for det in detections:
                    if im in self.ground_truth_seal:
                        if not im in self.tp:
                            self.tp[im] = []
                        self.tp[im].append(det)
                    elif im in self.ground_truth_background:
                        if not im in self.fp:
                            self.fp[im] = []
                        self.fp[im].append(det)

            for im in self.ground_truth_seal:
                if not im in self.tp:
                    self.fn[im] = self.api_images[im].hotspots

    def confidence_filter(self, confidence_thresh, iou_thresh):
        tp = {}
        fp = {}
        fn = {}
        for tpim in self.tp:
            for tp_det in self.tp[tpim]:
                if tp_det[0].confidence > confidence_thresh:
                    if not tpim in tp:
                        tp[tpim] = []
                    tp[tpim].append(tp_det)
                else:
                    if not tpim in fn:
                        fn[tpim] = []
                    fn[tpim].append(tp_det[1])
        for fnim in self.fn:
            if not fnim in fn:
                fn[fnim] = []
            for fn_det in self.fn[fnim]:
                fn[fnim].append(fn_det)
        for fpim in self.fp:

            for fp_det in self.fp[fpim]:
                if fp_det.confidence > confidence_thresh:
                    if not fpim in fp:
                        fp[fpim] = []
                    fp[fpim].append(fp_det)
        return tp, fp, fn