from pathlib import Path 
from utils import *
import gc



originalpath = Path("raw_vids")
vidpath = Path("alphapose_raw_vids/outputs")
original_vids = sorted(list(originalpath.glob("*")))
vids = sorted(list(vidpath.glob("*")))
zvids = zip(original_vids, vids)
for vid in tqdm(zvids):
    original_vid, vidp = vid
    if "8Ht3NoIxExY" != vidp.name:
        continue
    vis_frames = load_get_frames_from_folder(vidp/"converted_vis")
    # alphapose_frames = load_get_frames_from_video(list(vidp.glob("*.mp4"))[0])
    alphapose_vid = list(vidp.glob("*.mp4"))[0]
    merge_frames = merge_horizontal_vid_frameslist(alphapose_vid, vis_frames)
    saved_path = save_video_from_frames(merge_frames, "alphapose_merge", vidp.name)
    del merge_frames
    gc.collect()
    ori_vid_p = get_video(original_vid)
    add_audio_from_to(ori_vid_p, saved_path)
    break
    # merge_frames = draw_idx(merge_frames)
    # save_frames(merge_frames, vidp/"merge_vis")