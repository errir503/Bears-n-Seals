from arcticapi.augmnetation.TrainingImage import TrainingImage
from utils import *



def is_in_box(hotspot, top, bot, left, right):
    x, y = hotspot.getRGBCenterPt()
    return left < x < right and bot > y > top

def crop_rgb_hotspot(cfg, aeral_image):
    """
    :param cfg: CropCfg
    :type hs: HotSpot
    """
    if not aeral_image.load_image():
        return
    img = aeral_image.image
    hotspots = aeral_image.getHotSpots(cfg)
    drawn = []
    for hs in hotspots:
        # already drawn skip
        found = False
        for hs in drawn:
            for hs2 in hotspots:
                if hs.id == hs2.id:
                    found = True
        if found:
            continue

        classIndex = hs.classIndex
        imgh = img.shape[0]
        imgw = img.shape[1]

        tcrop, bcrop, lcrop, rcrop, center_x, center_y, dx, dy = recalculate_crops(hs.rgb_bb_b, hs.rgb_bb_t, hs.rgb_bb_l,
                                                                           hs.rgb_bb_r,
                                                                           imgh, imgw, cfg.maxShift, cfg.minShift,
                                                                           cfg.crop_size)
        todraw = []
        for hs in hotspots:
            isinbox = is_in_box(hs, tcrop, bcrop, lcrop, rcrop)
            if isinbox:
                drawn.append(hs)
                todraw.append(hs)

        crop_img = img[tcrop:bcrop, lcrop: rcrop]

        croph = crop_img.shape[0]
        cropw = crop_img.shape[1]
        ids = ""
        for item in todraw:
            ids += item.id + "_"
        bboxes = []
        for hs in todraw:
            x, y = hs.getRGBCenterPt()
            y = y - tcrop
            x = x - lcrop
            bboxes.append((hs.id, classIndex, (x + 0.0) / cropw, (y + 0.0) / croph, (cfg.bbox_size + 0.0) / cropw,
                           (cfg.bbox_size + 0.0) / croph))

        tr = TrainingImage(crop_img, cfg, aeral_image.path, bboxes, (tcrop, bcrop, lcrop, rcrop))

        # tr.random_hue_adjustment(0.05)
        tr.save()

        # # Generate negative image(with no object) and labels for training
        # tcrop, bcrop, lcrop, rcrop = negative_bounds(tcrop, bcrop, lcrop, rcrop, imgw, imgh, cfg.crop_size)
        # crop_img_neg = img[tcrop:bcrop, lcrop: rcrop]
        # tr_neg = TrainingImage(crop_img_neg, cfg, [], (tcrop, bcrop, lcrop, rcrop))
        # # tr.random_hue_adjustment(0.05)
        # tr_neg.save()

    # free image from memory
    aeral_image.free()


