
# Darknet integration
This package contains scripts to run detection with darknet.

To use these files:
* Copy the file you want to use to the root of your darknet directory(where you trained your model)
* Modify the file variables for the `.data`, `.cfg`, and `.weights` files
* Generate a file containing all the images you want to run detection on, one per line using `\n'` delimiter

### RGB
For detecting on RGB images use `darknetrgb.py`.  It generates tiles and feeds them into the network.  You can modify the tile
size but it's best to use the same size as the images you trained on and for the width and height to be divisible by 32(for small obects).

To run type `python darknetrgb.py image_list.txt`

### IR
Incomplete

To run type `python darknetir.py image_list.txt`
