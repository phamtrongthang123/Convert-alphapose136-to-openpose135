from asyncore import read
from errno import ENETDOWN
import os 
import json
from matplotlib.pyplot import axis 
import numpy as np 
from more_itertools.recipes import grouper, pairwise
from consta import *
from pathlib import Path 
import os 


def convert_video(path):
    print("=========== CONVERT VIDEO==============")
    filepath = os.path.join(path, 'alphapose-results.json')
    outdir = Path(os.path.join(path, 'converted'))
    os.makedirs(outdir, exist_ok=True)
    with open(filepath) as f:
        docs = json.load(f)
    ## remove dup 
    docsn = []
    check_score = True 
    p = None
    for doc in docs:
        # if doc['idx'] != 1:
        #     continue 
        if check_score:
            if p == None: 
                p = doc
                docsn.append(doc)
                continue
            if p['image_id'] == doc['image_id']:
                # print(doc['image_id'])
                if p['score'] > doc['score']:
                    continue
                else:
                    docsn.pop()
            p = doc
        docsn.append(doc)

    # convert
    for doc in tqdm(docsn):
        image_id = doc['image_id'].split(".")[0]
        doc_kpts = doc['keypoints']
        numarr = list(grouper(doc_kpts, 3))
        template = {"version":1.3,"people":[{"person_id":[-1],"pose_keypoints_2d":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[], "box":[]}]}

        # original format https://github.com/Fang-Haoshu/Halpe-FullBody 
        # idx is open, value is halpe
        body_halpe2open = [0,18,6,8,10,5,7,9,19,12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 20, 22, 24, 21, 23, 25]
        face_halpe2open = list(np.array(list(range(68)) + [37,44])+26)
        left_halpe2open = list(np.array(list(range(21)))+94)
        right_halpe2open = list(np.array(list(range(21)))+115)
        converted = template 
        subnumarr = [(0,0,0)]*len(body_halpe2open)
        for idx in range(len(subnumarr)):
            for other_idx in BODY_25_GRAPH.get(idx, set()):
                x1, y1, c1 = numarr[body_halpe2open[idx]]
                x2, y2, c2 = numarr[body_halpe2open[other_idx]]   
                c = min(c1, c2)
                if c < 0.05:
                    continue 
                subnumarr[idx] = numarr[body_halpe2open[idx]]
                subnumarr[other_idx] = numarr[body_halpe2open[other_idx]]
        converted['people'][0]['pose_keypoints_2d'] = [round(y,9) for x in subnumarr for y in x]

        subnumarr = [(0,0,0)]*len(face_halpe2open)
        for idx in range(len(subnumarr)):
            for other_idx in FACE_70_GRAPH.get(idx, set()):
                x1, y1, c1 = numarr[face_halpe2open[idx]]
                x2, y2, c2 = numarr[face_halpe2open[other_idx]]        
                c = min(c1, c2)
                if c < 0.05:
                    continue 
                subnumarr[idx] = numarr[face_halpe2open[idx]]
                subnumarr[other_idx] = numarr[face_halpe2open[other_idx]]
        converted['people'][0]['face_keypoints_2d'] = [round(y,9) for x in subnumarr for y in x]

        subnumarr = [(0,0,0)]*len(left_halpe2open)
        for idx in range(len(subnumarr)):
            for other_idx in HAND_21_GRAPH.get(idx, set()):
                x1, y1, c1 = numarr[left_halpe2open[idx]]
                x2, y2, c2 = numarr[left_halpe2open[other_idx]]    
                c = min(c1, c2)
                if c < 0.05:
                    continue 
                subnumarr[idx] = numarr[left_halpe2open[idx]]
                subnumarr[other_idx] = numarr[left_halpe2open[other_idx]]
        converted['people'][0]['hand_left_keypoints_2d'] = [round(y,9) for x in subnumarr for y in x]
        subnumarr = [(0,0,0)]*len(right_halpe2open)
        for idx in range(len(subnumarr)):
            for other_idx in HAND_21_GRAPH.get(idx, set()):
                x1, y1, c1 = numarr[right_halpe2open[idx]]
                x2, y2, c2 = numarr[right_halpe2open[other_idx]]       
                c = min(c1, c2)
                if c < 0.05:
                    continue 
                subnumarr[idx] = numarr[right_halpe2open[idx]]
                subnumarr[other_idx] = numarr[right_halpe2open[other_idx]]
        converted['people'][0]['hand_right_keypoints_2d'] = [round(y,9) for x in subnumarr for y in x]

        converted['people'][0]['box'] = doc['box']

        with open(outdir/f"{int(image_id):012d}_keypoints.json", 'w') as f: 
            json.dump(converted, f)

    print()
    return outdir



import json 
import sys
from math import ceil
import gizeh
from more_itertools.recipes import grouper, pairwise
import cv2
import numpy as np 
from consta import *
import os 
from pathlib import Path 
from tqdm import tqdm 
def body_vis(person, surface):
    ## POSE 
    numarr = list(grouper(person["pose_keypoints_2d"], 3))
    for idx in range(len(numarr)):
        for other_idx in BODY_25_GRAPH.get(idx, set()):
            x1, y1, c1 = numarr[idx]
            x2, y2, c2 = numarr[other_idx]
            
            c = min(c1, c2)
            if c < 0.05:
                continue
            if c == 0:
                continue
            drawcolor = colors_body[idx]
            line = gizeh.polyline(
                points=[(x1, y1), (x2, y2)], stroke_width=2,
                stroke=drawcolor
            )
            line.draw(surface)
            circ = gizeh.circle(r=5, xy=(x1,y1), fill=drawcolor)
            circ.draw(surface)
            circ = gizeh.circle(r=5, xy=(x2,y2), fill=drawcolor)
            circ.draw(surface)
    return surface

def face_vis(person, surface):
    ## FACE 
    numarr = list(grouper(person["face_keypoints_2d"], 3))
    for idx in range(len(numarr)):
        for other_idx in FACE_70_GRAPH.get(idx, set()):
            x1, y1, c1 = numarr[idx]
            x2, y2, c2 = numarr[other_idx]
            c = min(c1, c2)
            if c == 0:
                continue
            drawcolor = (1,1,1)
            line = gizeh.polyline(
                points=[(x1, y1), (x2, y2)], stroke_width=2,
                stroke=drawcolor
            )
            line.draw(surface)
            circ = gizeh.circle(r=2, xy=(x1,y1), fill=drawcolor)
            circ.draw(surface)
            circ = gizeh.circle(r=2, xy=(x2,y2), fill=drawcolor)
            circ.draw(surface)
    return surface 
def right_vis(person, surface):
        ## RIGHT HAND 
    numarr = list(grouper(person["hand_right_keypoints_2d"], 3))
    for idx in range(len(numarr)):
        for other_idx in HAND_21_GRAPH.get(idx, set()):
            x1, y1, c1 = numarr[idx]
            x2, y2, c2 = numarr[other_idx]
            c = min(c1, c2)
            if c == 0:
                continue
            drawcolor = colors_right[idx]
            line = gizeh.polyline(
                points=[(x1, y1), (x2, y2)], stroke_width=2,
                stroke=drawcolor
            )
            line.draw(surface)
            circ = gizeh.circle(r=2, xy=(x1,y1), fill=drawcolor)
            circ.draw(surface)
            circ = gizeh.circle(r=2, xy=(x2,y2), fill=drawcolor)
            circ.draw(surface)

    return surface

def left_vis(person, surface):
    ## LEFT HAND 
    numarr = list(grouper(person["hand_left_keypoints_2d"], 3))
    for idx in range(len(numarr)):
        for other_idx in HAND_21_GRAPH.get(idx, set()):
            x1, y1, c1 = numarr[idx]
            x2, y2, c2 = numarr[other_idx]
            c = min(c1, c2)
            if c == 0:
                continue
            drawcolor = colors_left[idx]
            line = gizeh.polyline(
                points=[(x1, y1), (x2, y2)], stroke_width=2,
                stroke=drawcolor
            )
            line.draw(surface)
            circ = gizeh.circle(r=2, xy=(x1,y1), fill=drawcolor)
            circ.draw(surface)
            circ = gizeh.circle(r=2, xy=(x2,y2), fill=drawcolor)
            circ.draw(surface)
    return surface 
def vis_video(filepath):
    print("============ VIS VIDEO ==========")
    fileroot = Path(f"{filepath}/converted")
    outroot = Path(f"{filepath}/converted_vis")
    os.makedirs(outroot, exist_ok=True)
    for filename in tqdm(fileroot.glob("*.json")):
        with open(filename) as f:
            doc = json.load(f)
        vcap = cv2.VideoCapture(str(list(Path(filepath).glob('*.mp4'))[0]))
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        surface = gizeh.Surface(width=width, height=height, bg_color=(0, 0, 0))
        for person in doc["people"]:
            surface = body_vis(person, surface)
            surface = face_vis(person, surface)
            surface = right_vis(person, surface)
            surface = left_vis(person, surface)
        surface.get_npimage()
        surface.write_to_png(str(outroot/f"{str(filename.name.split('.')[0]).zfill(10)}.png"))
    print()
    return outroot

def getfps(filepath):
    vcap = cv2.VideoCapture(str(list(Path(filepath).glob('*.mp4'))[0]))
    fps = vcap.get(cv2.CAP_PROP_FPS)
    return fps 

def make_video(filepath, fps):
    os.system(f"ffmpeg -y -framerate {fps} -i {filepath}/%10d.png {filepath}/vis.mp4")
    return f"{filepath}/vis.mp4"

def merge_vid(merge_root, original_vid, output_video):
    vido = str(list(Path(original_vid).glob('*.mp4'))[0])
    os.system(f"ffmpeg -y -i \"{vido}\" -i \"{output_video}\" -filter_complex hstack=inputs=2 \"{merge_root}/{Path(original_vid).name}.mp4\"")
    

def check_correct_pipeline(vids):
    for vid in vids: 
        converted = vid/"converted"
        converted_vis = vid/"converted_vis"
        num_file = len(list(converted.glob("*.json")))
        num_vis = len(list(converted_vis.glob("*.png")))
        assert num_file == num_vis 
    
def get_video(folder):
    return str(list(Path(folder).glob('*.mp4'))[0])
def get_frames_from_video(vidp):
    cap = cv2.VideoCapture(vidp )
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length
 

def get_frames_from_folder(vidf):
    length = len(list(vidf.glob("*.png"))) 
    return length

def clone_frames(total, vidp):
    os.system(f"cp -r {vidp}/converted_vis {vidp}/old_converted_vis")
    cvidp = vidp/"converted_vis"
    img = cv2.imread(str(list(Path(cvidp).glob('*.png'))[0]))
    height, width, channels = img.shape
    clonable = gizeh.Surface(width=width, height=height, bg_color=(0, 0, 0))
    for i in range(total):
        fn = str(cvidp/f"{str(i).zfill(10)}.png")
        if not os.path.exists(fn):
            clonable.write_to_png(fn)
    return cvidp

import imageio

def load_get_frames_from_folder(path):
    pngs = sorted(list(path.glob("*.png")))
    # frames = []
    # for fn in tqdm(pngs):
    #     image = imageio.imread(fn)
    #     frames.append(image)
    # return frames
    return pngs 

def load_get_frames_from_video(path):
    pass 


from skimage import img_as_ubyte
import moviepy.editor as me
def merge_horizontal_vid_frameslist(vidpath, framelist):
    reader = imageio.get_reader(vidpath)
    merge_frames = []
    count = 0 
    try:
        for i,im in tqdm(enumerate(reader)):
            left_frame = im 
            right_frame = imageio.imread(framelist[i])
            w, h = draw_text(left_frame, framelist[i].name, pos=(10, 10))
            # print(left_frame)
            merge = np.concatenate((left_frame, right_frame), axis=1)
            merge_frames.append(img_as_ubyte(merge))
            # imageio.imsave(folder/framelist[i].name, merge)
            # break
            # if count > 300:
            #     break 
            # count += 1
    except:
        pass
    return merge_frames

import cv2

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def add_audio_from_to(from_path, to_path):
    audio = me.VideoFileClip(from_path).audio

    video = me.VideoFileClip(to_path)
    # os.system(f"rm {to_path}")
    video = video.set_audio(audio)

    out_path = Path(to_path)
    parent_path = out_path.parent
    name_path = "ad-" + out_path.name
    out_path = str(parent_path.joinpath(name_path))
    video.write_videofile(out_path)

def save_video_from_frames(frames, folder, fname, fps=30):
    imageio.mimsave(os.path.join(folder,f"{fname}.mp4"), frames,
                    fps=fps) 
    return os.path.join(folder,f"{fname}.mp4")
