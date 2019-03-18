# for all images in directory make 640x640 crops 1000px from the left and 1000px from the top
# limit memory to 2gb and output in another folder
for k in *.JPG; do convert $k -crop 640x640+1000+1000 -limit memory 2gb -limit map 2gb -verbose ../640_negs/cropped_$k; done



