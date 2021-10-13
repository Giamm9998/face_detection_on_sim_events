import argparse
import os
import shutil
import scipy.interpolate

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from upsampling.utils import Upsampler
import numpy as np
import torch
from face_detection import detect_face, convert_frames_to_video
import cv2
import ntpath
import os
import tqdm
import moviepy.editor as mp
import skvideo
skvideo.setFFmpegPath('yor_ffmpeg_path')

try:
    import esim_py
except ImportError:
    print("esim_py not found, importing binaries. These do not correspond to source files in this repo")
    import sys

    binaries_folder = os.path.join(os.path.dirname(__file__), "..", "bin")
    sys.path.append(binaries_folder)
    print(sys.path)
    import esim_py

timestamps_file = "Data/Upsampled/seq0/timestamps.txt"
input_path = "Data/RGB_video/"
upsample_dir = "Data/Upsampled"


def upsample(fps, video_name, device):
    input_dir = input_path + video_name
    output_dir = upsample_dir

    # create fps file
    file = open(input_dir + "/seq0/fps.txt", "w")
    file.write(str(fps))
    file.close()

    upsampler = Upsampler(
        input_dir=input_dir,
        output_dir=output_dir,
        device=device
    )
    upsampler.upsample()

    sample = cv2.imread(upsample_dir + "/seq0/imgs/00000000.png")
    return sample.shape[0], sample.shape[1]


def generate_events(cp=0.1, cn=0.1, rp=1e-4, log_eps=1e-3, use_log=True):
    esim = esim_py.EventSimulator(cp, cn, rp, log_eps, use_log)
    events = esim.generateFromFolder(upsample_dir + "/seq0/imgs", timestamps_file)
    try:
        shutil.rmtree(upsample_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    return events


def viz_events(events, resolution):
    pos_events = events[events[:, -1] == 1]
    neg_events = events[events[:, -1] == -1]

    image_pos = np.zeros(resolution[0] * resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0] * resolution[1], dtype="uint8")

    np.add.at(image_pos, (pos_events[:, 0] + pos_events[:, 1] * resolution[1]).astype("int32"), pos_events[:, -1] ** 2)
    np.add.at(image_neg, (neg_events[:, 0] + neg_events[:, 1] * resolution[1]).astype("int32"), neg_events[:, -1] ** 2)

    image_rgb = np.stack(
        [
            image_pos.reshape(resolution),
            image_neg.reshape(resolution),
            np.zeros(resolution, dtype="uint8")
        ], -1
    ) * 50

    return image_rgb


def plot_events(events, H, W, i, xs_intrp, ys_intrp, video_name, is_not_none):
    image_rgb = viz_events(events, [H, W])
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("Data/Result/" + video_name + "/frame" + str(i) + ".png", gray_image)

    if is_not_none:
        xs = xs_intrp(i)
        ys = ys_intrp(i)

        xmax, ymax = xs.max(), ys.max()
        xmin, ymin = xs.min(), ys.min()

        return [xmin, ymin, xmax, ymax]
    else:
        print("No detections on frame " + str(i))
        return [None, None, None, None]


def adapt_events_array(events):
    tmp = []
    h, w = 0, 0  # maxh and maxw
    for evs in events:
        evs[0] = int(evs[0])
        if evs[0] > w:
            w = evs[0]
        evs[1] = int(evs[1])
        if evs[1] > h:
            h = evs[1]
        evs[3] = int(evs[3])
        evs[2] = int(evs[2] * 1000000)  # us
        tmp.append((evs[0], evs[1], evs[3], evs[2]))
    return tmp, int(h) + 1, int(w) + 1


def generate_frames(ac_time, events, H, W, xs_intrp, ys_intrp, video_name, nones):
    i = 0
    step = 0
    j = 0
    bboxs = []
    total = int(events[-1][2] / ac_time) + 1  # this number is not relevant
    indxs = [True for _ in range(total)]
    for x in nones:
        indxs[x] = False
    pbar = tqdm.tqdm(total=total)
    pbar.set_description("Frames generated")
    while j < len(events) - 1:
        while events[j][2] - events[step][2] < ac_time and j < len(events) - 1:
            j += 1
        next_step = j
        bbox = plot_events(events[step:next_step], H, W, i, xs_intrp, ys_intrp, video_name, indxs[i])
        bboxs.append(bbox)
        step = next_step
        i += 1
        pbar.update()
    return np.array(bboxs)


def manage_input_path(video_path, res):
    video_name = ntpath.basename(video_path).split(".")[0]
    video_format = ntpath.basename(video_path).split(".")[1]
    if not os.path.exists(input_path + video_name):
        os.mkdir(input_path + video_name)
    if not os.path.exists(input_path + video_name + "/seq0"):
        os.mkdir(input_path + video_name + "/seq0")
    if not os.path.exists("Data/Result/" + video_name):
        os.mkdir("Data/Result/" + video_name)
    clip = mp.VideoFileClip(video_path)
    clip_resized = clip.resize(height=res)
    clip_resized.write_videofile(input_path + video_name + "/seq0/" + video_name + "." + video_format)
    # shutil.copy(video_path, input_path + video_name + "/seq0")
    return video_name, video_format


def interpolate_detections(detections, k, res_frame_num):
    xs = []
    ys = []
    nones = []
    count = 0
    for i in range(len(detections)):
        if detections[i] is None:
            nones.append(i)
        else:
            xs.append(detections[i][0][:, 0])
            ys.append(detections[i][0][:, 1])
    p = [0]
    for i in range(1, res_frame_num):
        count += k
        if i not in nones:
            p.append(count)
    xs_intrp = scipy.interpolate.interp1d(p, np.vstack(xs), axis=0, fill_value='extrapolate')
    ys_intrp = scipy.interpolate.interp1d(p, np.vstack(ys), axis=0, fill_value='extrapolate')
    _nones = []
    for i in nones:
        for j in range(round(i * k), round(i * k + k)):
            _nones.append(j)
    return xs_intrp, ys_intrp, _nones


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True,
                        help='Path to input file. See README.md for output directory.')
    parser.add_argument("--device", type=str, default="cpu", help='Device to be used (cpu, cuda:X)')
    parser.add_argument("--ups_fps", type=int, default=25, help='Upsampling fps')
    parser.add_argument("--resize_h", type=int, default=240, help='Height of resized frames')
    parser.add_argument("--acc_time", type=float, required=True, help='Accumulation time for frame creation')
    parser.add_argument("--video_res", type=bool, default=False,
                        help='Set this True if you want to generate a video with the result frames')

    args = parser.parse_args()
    return args


def main():
    # parameters TODO parser

    flags = get_flags()
    video_path = flags.input_file
    output_path = "Data/Result/"
    resolution = flags.resize_h
    up_fps = flags.ups_fps
    acc_time = flags.acc_time
    # res_fps = 60

    video_name, video_format = manage_input_path(video_path, resolution)
    H, W = upsample(up_fps, video_name, flags.device)
    tmp_path = input_path + video_name + "/seq0/" + video_name + "." + video_format
    detections, video_frame_num = detect_face(H, W, input_path + video_name, tmp_path, flags.device)
    events = generate_events()
    video_len = events[-1][2]
    res_frame_num = video_len / acc_time
    k = res_frame_num / video_frame_num
    xs_intrp, ys_intrp, nones = interpolate_detections(detections, k, video_frame_num)
    del detections
    bboxs = generate_frames(acc_time, events, H, W, xs_intrp, ys_intrp, video_name, nones)
    np.save("Data/Detections/bboxs/" + video_name, bboxs)
    del events
    if flags.video_res:
        convert_frames_to_video(output_path + video_name + "/", output_path + video_name + "." + video_format,
                                int(1 / acc_time), video_name)


if __name__ == '__main__':
    main()
