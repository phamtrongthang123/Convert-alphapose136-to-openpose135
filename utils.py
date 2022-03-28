import os 
import json 
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
        template = {"version":1.3,"people":[{"person_id":[-1],"pose_keypoints_2d":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}

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


        with open(outdir/f"{image_id}.json", 'w') as f: 
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
