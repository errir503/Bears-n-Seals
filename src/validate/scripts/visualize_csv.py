import cv2

from validate import Parser

p = Parser('../new_file.csv')

rows = p.get_objects()
for row in rows:
    img = cv2.imread(row.file)
    cv2.rectangle(img, (row.local_x - row.bbox_width / 2, row.local_y - row.bbox_height / 2),
                  (row.local_x + row.bbox_width / 2, row.local_y + row.bbox_height / 2),
                  (0, 255, 0), 1)  # draw rect
    cv2.imshow('Window', img)

    key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break
    del img