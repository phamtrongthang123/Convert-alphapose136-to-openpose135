from pathlib import Path 
from utils import *
import gc



originalpath = Path("raw_vids")
vidpath = Path("first_delivery_batch2")
original_vids = sorted(list(originalpath.glob("*")))
vids = sorted(list(vidpath.glob("*")))
zvids = zip(original_vids, vids)

skips = ['vhH2P096CUM', 'wxKYvceauBg', 'x3cyRffU3_U', 'x9LCyHll9FM', 'xTfNBn6mQT0', 'yW03xxXaRHY', 'z3r0ksw5_os', 'zcFf9QaALc0']


for vid in tqdm(zvids):
    original_vid, vidp = vid
    original_vid = originalpath/f"{vidp.name}.mp4"
    if vidp.name not in skips:
        continue
    try:
        vis_frames = load_get_frames_from_folder(vidp/"converted_vis")
        # alphapose_frames = load_get_frames_from_video(list(vidp.glob("*.mp4"))[0])
        alphapose_vid = list(vidp.glob("*.mp4"))[0]
        merge_frames = merge_horizontal_vid_frameslist(alphapose_vid, vis_frames)
        saved_path = save_video_from_frames(merge_frames, "alphapose_merge_type2", vidp.name)
        del merge_frames
        gc.collect()
        # ori_vid_p = get_video(original_vid
        ori_vid_p = str(original_vid)

        add_audio_from_to(ori_vid_p, saved_path)
    except Exception as e:
        print("skip ", vidp)
        print("Error ", e)
        try:
            del merge_frames
            gc.collect()
        except:
            pass 
    # break
    # merge_frames = draw_idx(merge_frames)
    # save_frames(merge_frames, vidp/"merge_vis")