import csv
import os
from datetime import datetime
from operator import itemgetter
from src.arcticapi import ArcticApi
from src.arcticapi.config import load_config

csv_me = '/home/yuval/Documents/XNOR/Bears-n-Seals/src/bbox-labeler/updated_live_out.csv'
csv_microsoft = '/data/raw_data/TrainingAnimals_WithSightings.csv'
img_path = '/data/raw_data/CHESS/'
out_path = "/data/raw_data/merged.csv"
cfg = load_config("full")

api = ArcticApi(csv_me, img_path)

ms_hs = {}
odd_rows = []
f = open(csv_microsoft, 'r')
reader = csv.reader(f)
for idx, row in enumerate(reader):
    if idx == 0:
        continue
    if row[5] != "Bearded Seal" and row[5] != "Ringed Seal":
        print(row[5])
        continue
    if row[3] == "NA":
        odd_rows.append(row)
        continue
    if not row[3] in ms_hs:
        ms_hs[row[3]] = row
f.close()

for hs in api.hsm.hotspots:
    if hs.id in ms_hs:
        ms_hs[hs.id].append(hs.updated_left)
        ms_hs[hs.id].append(hs.updated_top)
        ms_hs[hs.id].append(hs.updated_right)
        ms_hs[hs.id].append(hs.updated_bot)
        ms_hs[hs.id].append(hs.updated)
        ms_hs[hs.id].append(hs.status)
        ms_hs[hs.id].append("merged_unverified")

    else:
        if hs.type == "Anomaly":
            continue
        base_rgb = os.path.basename(hs.rgb.path)
        split = base_rgb.split('_')
        base_ir = os.path.basename(hs.ir.path)
        year = int(hs.timestamp[0:4])
        month = int(hs.timestamp[5:7])
        date = int(hs.timestamp[8:10])
        hours = int(hs.timestamp[11:13])
        minutes = int(hs.timestamp[14:16])
        seconds = int(hs.timestamp[17:19])
        ms = int(hs.timestamp[20:])
        t = datetime(year, month, date, hours, minutes, seconds, ms)
        strDate=t.strftime("%Y%m%d%H%M%S.") + str(ms) +"GMT"
        new_rgb_name = split[0] +str(year) +"_N94S_"+ split[1] +"_"+ split[2] +"_"+ strDate +"_"+ split[5]
        ms_hs[hs.id] =  [-1, base_rgb, base_ir, hs.id, hs.type, hs.species,"NA", "NA", hs.rgb_bb_l, hs.rgb_bb_t, hs.rgb_bb_r, hs.rgb_bb_b, hs.thermal_loc[0], hs.thermal_loc[1],
                         hs.updated_left, hs.updated_top, hs.updated_right, hs.updated_bot, hs.updated, hs.status, "Yuval Only"]






for k in ms_hs:
    if len(ms_hs[k]) == 14:
        ms_hs[k].append(-1)
        ms_hs[k].append(-1)
        ms_hs[k].append(-1)
        ms_hs[k].append(-1)
        ms_hs[k].append(False)
        ms_hs[k].append("none")
        ms_hs[k].append("Microsoft Only")

#CHESS_FL23_S_160517_235529.370_COLOR-8-BIT.JP old image name
# CHESS_FL23_S_160517_235529.370_COLOR-8-BIT.JP new image name

temp = []
for key, value in ms_hs.iteritems():
    temp.append(value)
temp.sort(key=lambda x: float(x[3]))
wtr = csv.writer(open ('scripts/out.csv', 'w'), delimiter=',', lineterminator='\n')
header = "id,color_image,thermal_image,hotspot_id,hotspot_type,species_id,species_confidence,fog,thermal_x,thermal_y,color_left,color_top,color_right,color_bottom, updated_left, updated_top, updated_right, updated_bottom, updated, status, merge_status"
wtr.writerow(header.split(','))
for row in temp:
    wtr.writerow(row)

pass