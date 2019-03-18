# for all images in directory make 640x640 crops 1000px from the left and 1000px from the top
# limit memory to 2gb and output in another folder
for k in /data/raw_data/TrainingBackground_ColorImages_00/*.JPG; do convert $k -crop 640x640 -repage -limit memory 2gb -limit map 2gb -verbose crop_%02d_$k; done