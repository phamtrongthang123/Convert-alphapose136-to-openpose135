from pathlib import Path 
from utils import *


originalpath = Path("raw_vids")
vidpath = Path("alphapose_raw_vids/outputs")
original_vids = sorted(list(originalpath.glob("*")))
vids = sorted(list(vidpath.glob("*")))
check_correct_pipeline(vids)
zvids = zip(original_vids, vids)
for vid in tqdm(zvids):
    original_vid, vidp = vid
    ori_vid_p = get_video(original_vid)
    totalframes = get_frames_from_video(ori_vid_p)
    num_output_frames = get_frames_from_folder(vidp/"converted_vis")
    print(totalframes, num_output_frames)
    new_vidp = vidp/"converted_vis"
    if totalframes != num_output_frames:
        new_vidp = clone_frames(totalframes, vidp)

    
    fps = getfps(original_vid)
    output_video = make_video(new_vidp, fps)
    merge_root = "alphapose_merge"
    os.makedirs(merge_root, exist_ok=True)
    merge_vid(merge_root, original_vid, output_video)