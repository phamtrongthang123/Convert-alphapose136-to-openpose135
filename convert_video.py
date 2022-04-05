import json 
import numpy as np 
from more_itertools.recipes import grouper, pairwise
from consta import *
from pathlib import Path 
import os 
from tqdm import tqdm

# filepath = Path('AlphaPose_test/AlphaPose_test_dance1_posetrack.json')
filepath = Path('AlphaPose_test/AlphaPose_BTSdynamite-cut_posetrack.json')
outdir = Path('AlphaPose_test/converted')
os.makedirs(outdir, exist_ok=True)
with open(filepath) as f:
    docs = json.load(f)

## remove dup 
docsn = []
check_score = False 
p = None
for doc in docs:
    if doc['idx'] != 1:
        continue 
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
    
    # break
