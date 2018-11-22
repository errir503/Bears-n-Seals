import cv2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def show_ir(hsm, colorJet = False):
    for hs in hsm.hotspots:
        if not hs.ir.load_image(colorJet):
            continue
        cv2.imshow('norm', hs.ir.image[0])
        cv2.imshow('anydepth', hs.ir.image[1])
        cv2.imshow('normglobal', hs.ir.image[2])
        cv2.imshow('normlocal', hs.ir.image[3])
        cv2.waitKey(0)