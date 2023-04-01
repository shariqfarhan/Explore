'''
The following set of code was run in terminal with the base files being changed to get the required outputs
'''
# Code to check if the setup is working on the base dataset
python train.py --data data/smalcoco/smalcoco.data--batch 3 --cache --epochs 3 --nosave

# We then modify the base data for custom training
python train.py --data data/customdata/custom.data--batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 50 --nosave

# Detect was changed to detect_new to include the new location of the video output
python detect_new.py --conf-thres 0.1 --output video_out_out

