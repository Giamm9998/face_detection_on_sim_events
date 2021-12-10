# Face_detection_on_sim_events

This repository contains code that implements the conversion of RGB video to event-based video with face detection. It can be used to create an annotated dataset of event-based frames (or videos). The code for the simulation comes from this other repository https://github.com/uzh-rpg/rpg_vid2e therefore you need to install all the packages/requirements specified there in order to use this repository.

![conc](https://user-images.githubusercontent.com/49485831/145583080-5b6ae20f-1b00-448b-8ef6-c5869d271b9b.png)
![conc2](https://user-images.githubusercontent.com/49485831/145583083-4f5cd3b0-1c8e-4915-a54f-aa0768519483.png)
![conc3](https://user-images.githubusercontent.com/49485831/145583084-acc86de4-11d5-4bef-bbbb-8eb975c9e454.png)


## Important requirements
These are the most important required packages to install:
* opencv
* face-alignment -> https://github.com/1adrianb/face-alignment
* pytorch, torchvision, cudatoolkit -> if you want to use gpu (suggested)
* moviepy
* scipy

## Usage
First of all you need to set your ffmpeg path, modifying the "skvideo.setFFmpegPath" in two files: simulator.py, upsampling/utils/dataset.py. <br />
The file to exectute for the actual conversion is simulator.py. Example:
```
python3 simulator.py --input_file /path/to/video.mp4 --acc_time 0.01
```
The parameters *input_file* and *acc_time* are required. Explanation of every parameter:
* input_file  -> path to the video to convert
* acc_time  -> accumulation time used for the frame creation. Lower acc_time = More resulting frames and less events for each frame
* device  -> default is cpu, change it if you want to use cuda (ex. "cuda:0")
* ups_fps  -> fps used during the upsampling operation
* resize-h  -> The frames of the input video are resized before the upsampling to keep execution time reasonable. This parameter allows to choose the resize height. Default (and suggested) is 240, the width is obtained keeping the original ratio.
* video_res  -> Set this parameter True if you want to generate a video from the resulting frames. It's useful in order to check the correctness of the detections and the overall result.

The converted frames are stored in the result directory, whereas the detections (landmarks and bounding boxes) are stored in Detections directory as numpy files. <br /> The bounding boxes are arrays of the type: \[x1, y1, x2, y2\] where (x1,y1) are the coordinates of the top-left point and (x2,y2) correspond to the bottom-right point.

## Train a face detector
In the project from which this code comes, with the resulting frames a face detector has been trained (for more info check the doc). The face detection system that has been used can be found at https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/df98840d000bcc98c1b2db5df10f430550caf937. If you want to use the annotated dataset obtained with the code of this repository to train YOLOv3 follow the instructions on his repository. The only preprocessing operations needed before the training regards the bounding box format (to adapt to YOLOv3 requirements).
