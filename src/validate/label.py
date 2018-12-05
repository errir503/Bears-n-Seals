import os
import sys
from PIL import Image
sys.path.append("../")
from validate import Parser

csv_file = 'new_file.csv'
outDir = 'newlabels/'
if not os.path.exists(outDir):
    os.mkdir(outDir)

p = Parser(csv_file)
rows = p.get_objects()

for row in rows:
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




