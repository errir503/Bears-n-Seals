import csv
import os
from PIL import Image

csv_file = 'new_file.csv'
outDir = 'newlabels/'
if not os.path.exists(outDir):
    os.mkdir(outDir)

rows = []
f = open(csv_file, 'r')
reader = csv.reader(f)
for row in reader:
    rows.append(row)
f.close()

del rows[0]  # remove col headers
d = {'SEAL':0, 'NOTSEAL':0, 'MAYBESEAL':0, 'UNCHECKED':0}

row_objects = []
for row in rows:
    status = row[11]
    d[status] += 1
    obj = lambda: None
    obj.num = int(row[0])
    obj.file = row[1]
    obj.pred = float(row[2])
    obj.local_x = float(row[3])
    obj.local_y = float(row[4])
    obj.bbox_width = float(row[5])
    obj.bbox_height = float(row[6])
    obj.crop_top = int(row[7])
    obj.crop_bot = int(row[8])
    obj.crop_left = int(row[9])
    obj.crop_right = int(row[10])
    if len(row) != 12:
        obj.status = 'UNCHECKED'
    else:
        status = row[11]
        obj.status = status
    row_objects.append(obj)
del rows
print(d)

for row in row_objects:
    status = row.status
    basename = os.path.splitext(os.path.basename(row.file))[0] + "_" + str(row.num)
    txtname = basename + ".txt"
    imgname = basename + ".jpg"
    if status == "NOTSEAL":
        # Marked as not a seal, generate negative
        img = Image.open(row.file)
        img = img.crop((row.crop_left, row.crop_top, row.crop_right, row.crop_bot))


        img.save(outDir + imgname)
        open(outDir + txtname, 'a').close()
    elif status == "SEAL":
        img = Image.open(row.file)
        img = img.crop((row.crop_left, row.crop_top, row.crop_right, row.crop_bot))

        basename = os.path.basename(row.file)
        noext = os.path.splitext(basename)[0]

        cropw = (row.crop_right - row.crop_left)
        croph = (row.crop_bot - row.crop_top)
        with open(outDir + txtname, 'a') as file:
            file.write(str(0) + " " + str((row.local_x + 0.0) / cropw) + " " +
                       str((row.local_y + 0.0) / croph) + " " +
                       str((row.bbox_width + 0.0) / cropw) + " " +
                       str((row.bbox_height + 0.0) / croph) + "\n")
        img.save(outDir + imgname)




