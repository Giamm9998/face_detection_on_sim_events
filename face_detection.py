import ntpath
import shutil

import face_alignment
import cv2
import numpy as np
import os
from os.path import isfile, join
import tqdm


def convert_frames_to_video(pathIn, pathOut, fps, video_name):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    dets = np.load("Data/Detections/bboxs/" + video_name + ".npy", allow_pickle=True)
    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        if dets[i][0] is not None :
            img = cv2.rectangle(img, (round(dets[i][0]), round(dets[i][1])), (round(dets[i][2]), round(dets[i][3])),
                                (255, 255, 255), 2)
        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    return


def detect_face(H, W, video_dir, video_path, device):
    # frame generation
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (W, H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # generate face detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='sfd')
    detections = []
    pbar = tqdm.tqdm(total=len(frames))
    pbar.set_description("Frames detected")
    for frame_ in frames:
        # face detection
        det = fa.get_landmarks_from_image(frame_)
        detections.append(det)
        pbar.update()
    video_name = ntpath.basename(video_path).split(".")[0]
    np.save("Data/Detections/landmarks/" + video_name, detections)
    try:
        shutil.rmtree(video_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    return detections, len(frames)
