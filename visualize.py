import json 
import sys
from math import ceil
import gizeh
from more_itertools.recipes import grouper, pairwise
import cv2
import numpy as np 
from consta import *



def main():
    # filename = 'openpose/result_json/COCO_val2014_000000000459_keypoints.json'
    filename ='alphapose/converted-alphapose-results.json'
    with open(filename) as f:
        doc = json.load(f)
    img = cv2.imread('image/COCO_val2014_000000000459.jpg')
    height, width, channels = img.shape
    surface = gizeh.Surface(width=width, height=height, bg_color=(0, 0, 0))
    for person in doc["people"]:
        ## POSE 
        numarr = list(grouper(person["pose_keypoints_2d"], 3))
        for idx in range(len(numarr)):
            for other_idx in BODY_25_GRAPH.get(idx, set()):
                x1, y1, c1 = numarr[idx]
                x2, y2, c2 = numarr[other_idx]
                
                c = min(c1, c2)
                
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

    surface.get_npimage()
    surface.write_to_png('abc.png')

if __name__ == "__main__":
    main()