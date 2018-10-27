import os
from datetime import datetime

import NOAA

HOTSPOT_ID_COL_IDX = 0
TIMESTAMP = 1
IMG_THERMAL16_COL_IDX = 2
IMG_THERMAL8_COL_IDX = 3
IMG_RGB_COL_IDX = 4
XPOS_IDX = 5
YPOS_IDX = 6
LEFT_IDX = 7
TOP_IDX = 8
RIGHT_IDX = 9
BOT_IDX = 10
HOTSPOT_TYPE_COL_IDX = 11
SPECIES_ID_COL_IDX = 12

def parse_meta_deta(filename):
    tokens = filename.split('_')
    if(len(tokens) < 3):
        return None, None, None
    project_name = tokens[0]
    aircraft = tokens[1]
    camera_pos = tokens[2]

    return project_name, aircraft, camera_pos




def parse_ts(ts):
    if len(ts) != 21 or ts[18:21] != 'GMT':
        return None
    year = int(ts[0:4])
    month = int(int(ts[4:6]) - 1)
    date = int(ts[6:8])
    hours = int(ts[8:10])
    minutes = int(ts[10:12])
    seconds = int(ts[12:14])
    ms = int(ts[15:18])
    return datetime(year, month, date, hours, minutes, seconds, ms)


def parse_hotspot(row, res_path):
    # get camera positions, project name, and aircraft
    time = parse_ts(row[TIMESTAMP])
    project_name, aircraft, rgb_pos = parse_meta_deta(row[IMG_RGB_COL_IDX])
    project_name, aircraft, thermal_pos = parse_meta_deta(row[IMG_THERMAL8_COL_IDX])
    project_name, aircraft, ir_pos = parse_meta_deta(row[IMG_THERMAL16_COL_IDX])
    # create each image object
    rgb = NOAA.Image(res_path + row[IMG_RGB_COL_IDX], "rgb", rgb_pos)
    thermal = NOAA.Image(res_path + row[IMG_THERMAL8_COL_IDX], "thermal", thermal_pos)
    ir = NOAA.Image(res_path + row[IMG_THERMAL16_COL_IDX], "ir", ir_pos)
    return NOAA.HotSpot(row[HOTSPOT_ID_COL_IDX],
                        int(row[XPOS_IDX]),
                        int(row[YPOS_IDX]),
                        int(row[LEFT_IDX]),
                        int(row[TOP_IDX]),
                        int(row[RIGHT_IDX]),
                        int(row[BOT_IDX]),
                        row[HOTSPOT_TYPE_COL_IDX],
                        row[SPECIES_ID_COL_IDX],
                        rgb,
                        thermal,
                        ir,
                        time,
                        project_name, aircraft)