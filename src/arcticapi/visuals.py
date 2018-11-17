import cv2


def show_ir(hsm, colorJet = False):
    for hs in hsm.hotspots:
        if not hs.ir.load_image(colorJet):
            continue
        cv2.imshow('norm', hs.ir.image[0])
        cv2.imshow('anydepth', hs.ir.image[1])
        cv2.imshow('normglobal', hs.ir.image[2])
        cv2.imshow('normlocal', hs.ir.image[3])
        cv2.waitKey(0)