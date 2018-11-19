
# Bears-n-Seals
This a project collaboration between NOAA and XNOR AI.  The goal of this
project is to detect seals and polar bears in aerial imagery from different locations in the arctic.


## Data
All images were gathered from a plane flying ~1,000 feet above the ice.

* `_CHESS_ImagesSelected4Detection.csv` : All raw hotspot data.

The column schema is as follows(each record in unique):

* `hotspot_id`: unique ID
* `timestamp`: GMT/UTC timestamp (always corresponds to thermal image timestamp)
* `filt_thermal16`: Filename of the 16-bit PNG containing the raw FLIR image data
* `filt_thermal8`: Filename of the 8-bit JPG containing the annotated FLIR image data (hotspots circled)
* `filt_color`: Filename of the 8-bit JPG containing a color image taken at or near the same time as the thermal image. The timestamp encoded in the filename may be different from the thermal timestamp by up to 60 seconds (but typically less than 1 second).
* `x_pos`/`y_pos`: Location of the hotspot in the thermal image
* `thumb_*`: Bounding box of the hotspot in the color image. **NOTE**: some of these values are negative, as the bounding box is always 512x512 even if the hotspot is at the edge of the image.
* `hotspot_type`: "Animal" or "Anomaly", classified by human "Animal" (true positive) or "Anomaly" (false positive)
* `species_id`: "Bearded Seal", "Ringed Seal", "UNK Seal", "Polar Bear" or "NA" (for anomalies)

The file names for `filt_color, filt_thermal8, filt_thermal16`:

Example: `CHESS_FL1_C_160408_000946.314_THERM-16BIT.PNG`
* [CHESS] : project name
* [FL1] : plane tail number
* [c] : camera - plane caries 3 pairs of thermal and EO cameras
* [160408] : flight id
* [000946.314] : timestamp  - sometimes formatted like 000946.314GMT
* [THERM-16BIT]: image type = [THERM-16BIT] is for IR, [COLOR-8-BIT] is for RGB, [THERM-8-bit] is for thermal 8-bit

**NOTE:** timestamps are taken from thermal and applied to IR/therm image names

## Preprocessing:
This script crops images and generates bounding box labels in darknet/yolo label foramt.
```usage src/preprocess.py

usage: preprocess.py

required arguments:
  --csv CSV      csv file: relative path to the seal image data csv file
                 (default: None)
  --imdir IMDIR  image dir: relative path to the directory containing all
                 images (default: None)
  --out OUT      out dir: relative path to the directory to store cropped
                 images (default: None)
                 
optional arguments:
  --bb BB        bounding box size: size of bounding box width and height
                 around the center point (default: 70)
  --min MIN      min shift: min value shift center point dx and dy, calculated
                 as random value between min and max (default: 100)
  --max MAX      max shift: max value shift center point dx and dy, calculated
                 as random value between min and max (default: 250)
  --cs CS        crop size: size of croped region (default: 512)
  --label LABEL  label: output file with all absolute label paths for training
                 (default: training_list.txt)
```

## TODOs
* Preprocess test/train split option
* Darknet cfg file generator for quicker training
* Image registration?


###Info
[afsc.noaa.gov/News/iceseal_pop_assess.htm](https://www.afsc.noaa.gov/News/iceseal_pop_assess.htm)

### Useful Commands:

Test/Train Split:
`head -n 1000 training_list.txt > sealvalid.txt`
`tail -n +1000 training_list.txt > sealtrain.txt`



Generat map score (Must use "AB Darknet fork")
`./darknet detector map cfg/bearsnseals.data cfg/bearsnseals.cfg weights/backup.weights`



