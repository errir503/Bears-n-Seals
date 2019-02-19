from datetime import datetime

from model import AerialImage, HotSpot

# Original columns
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
# New columns that I added
UPDATED_TOP_IDX = 13
UPDATED_BOT_IDX = 14
UPDATED_LEFT_IDX = 15
UPDATED_RIGHT_IDX = 16
UPDATED_IDX = 17
STATUS_IDX = 18


def parse_meta_data(filename):
    tokens = filename.split('_')
    if (len(tokens) < 3):
        return None, None, None
    project_name = tokens[0]
    aircraft = tokens[1]
    camera_pos = tokens[2]

    return project_name, aircraft, camera_pos


def parse_ts(ts):
    try:
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
    except:
        return None


def parse_hotspot_new_dataset(row, cfg):
    rgb_name_meta_deta = row[1].split("_")
    ir_name_meta_deta = row[2].split("_")
    rgb = AerialImage(cfg.rgb_dir + row[1], "rgb", rgb_name_meta_deta[3], fog=row[7])
    ir = AerialImage(cfg.rgb_dir + row[2], "ir", ir_name_meta_deta[3], fog=row[7])
    if row[8] == "NA":
        return None
    return HotSpot(row[3],
                   int(row[8]),
                   int(row[9]),
                   int(row[10]),
                   int(row[11]),
                   int(row[12]),
                   int(row[13]),
                   row[4],
                   row[5],
                   rgb,
                   ir,
                   rgb_name_meta_deta[5],
                   rgb_name_meta_deta[1], rgb_name_meta_deta[2], confidence=row[6])

def parse_hotspot(row, cfg):
    # get camera positions, project name, and aircraft
    # time = parse_ts(row[TIMESTAMP])
    time = row[TIMESTAMP]
    project_name, aircraft, rgb_pos = parse_meta_data(row[IMG_RGB_COL_IDX])
    project_name, aircraft, thermal_pos = parse_meta_data(row[IMG_THERMAL8_COL_IDX])
    project_name, aircraft, ir_pos = parse_meta_data(row[IMG_THERMAL16_COL_IDX])
    # create each image object
    rgb = AerialImage(cfg.rgb_dir + row[IMG_RGB_COL_IDX], "rgb", rgb_pos)
    ir = AerialImage(cfg.ir_dir + row[IMG_THERMAL16_COL_IDX], "ir", ir_pos)

    if len(row) == 13:
        return HotSpot(row[HOTSPOT_ID_COL_IDX],
                       int(row[XPOS_IDX]),
                       int(row[YPOS_IDX]),
                       int(row[LEFT_IDX]),
                       int(row[TOP_IDX]),
                       int(row[RIGHT_IDX]),
                       int(row[BOT_IDX]),
                       row[HOTSPOT_TYPE_COL_IDX],
                       row[SPECIES_ID_COL_IDX],
                       rgb,
                       ir,
                       time,
                       project_name, aircraft)
    else:
        updated =  row[UPDATED_IDX]
        isUpdated = (updated == 'true')
        return HotSpot(row[HOTSPOT_ID_COL_IDX],
                       int(row[XPOS_IDX]),
                       int(row[YPOS_IDX]),
                       int(row[LEFT_IDX]),
                       int(row[TOP_IDX]),
                       int(row[RIGHT_IDX]),
                       int(row[BOT_IDX]),
                       row[HOTSPOT_TYPE_COL_IDX],
                       row[SPECIES_ID_COL_IDX],
                       rgb,
                       ir,
                       time,
                       project_name, aircraft,
                       int(row[UPDATED_BOT_IDX]),
                       int(row[UPDATED_TOP_IDX]),
                       int(row[UPDATED_LEFT_IDX]),
                       int(row[UPDATED_RIGHT_IDX]),
                       isUpdated,
                       row[STATUS_IDX])


