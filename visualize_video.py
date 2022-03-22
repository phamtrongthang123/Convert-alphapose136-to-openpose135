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
def main():
    fileroot = Path('AlphaPose_test/converted/')
    outroot = Path('AlphaPose_test/converted_vis')
    os.makedirs(outroot, exist_ok=True)
    for filename in tqdm(fileroot.glob("*.json")):
        with open(filename) as f:
            doc = json.load(f)
        vcap = cv2.VideoCapture('AlphaPose_test/AlphaPose_BTSdynamite-cut_posetrack.mp4')
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

if __name__ == "__main__":
    main()