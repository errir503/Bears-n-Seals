Required Libraries

Python 2.7
cv2
numpy


Pipeline for training:
- Attempt to align RGB and Thermal images
- Break images into tiles with same resolution as input images
- Generate Labels for Tiles
- Train
- Repeat



Useful Commands:
insert file path at beginning of every line
:%s!^!/home/yuval/Bears-n-Seals/tiles/!

export PATH=/usr/local/cuda-9.1/bin:$PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

head -n 1000 items.txt > valid.txt
tail -n +1001 items.txt > train.txt