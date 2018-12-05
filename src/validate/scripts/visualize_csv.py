from PIL import Image
import cv2
import sys
sys.path.append("../../")
from validate import Parser

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

p = Parser('good.csv')
rows = p.get_objects()
for row in rows:
    img = cv2.imread(row.file)

    img = img[row.crop_top:row.crop_bot,row.crop_left:row.crop_right]
    cv2.rectangle(img, (int(row.local_x - row.bbox_width / 2), int(row.local_y - row.bbox_height / 2)),
                  (int(row.local_x + row.bbox_width / 2), int(row.local_y + row.bbox_height / 2)), (0, 255, 0), 2)  # draw rect

    cv2.putText(img,row.status, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.imshow('Window', img)

    key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break
    del img
