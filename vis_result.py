import os
from pathlib import Path

import cv2


def get_pics(path):
    # Combines directory listing with path joining in one step
    return [os.path.join(path, filename) for filename in sorted(os.listdir(path))]

def load_text(file_path, delimiter):
    with open(file_path, 'r') as file:
        return [line.strip().split(delimiter) for line in file]

# Set paths
dataset_base = Path('/home/qiuyang/datasets/HUT290/HUT290')
results_base = Path('/home/qiuyang/comparison/raw_results/tracking_results')
test_set = 'fish51'

# Load data
ground_truth = load_text(dataset_base / test_set / 'groundtruth.txt', delimiter=',')
img_paths = get_pics(str(dataset_base / test_set / ''))

# Load tracking results
trackers = {
    'PU-Artrack_seq': 'PU-Artrack_seq',
    'PU-OSTrack': 'baseline',
    'artrack_seq': 'artrack_seq_256_full',
    #'odtrack': 'baseline_300',
    'ostrack': 'vitb_256_mae_32x4_ep300',
    #'mixformer_cvt_online': 'baseline',
    #'mixformer2_vit_online': '288_depth8_score',


    # add tracker which you want at here ......
}

name_map = {
    'PU-Artrack_seq': 'PU-ARTrack',
    'PU-OSTrack': 'PU-OSTrack',
    'artrack_seq': 'ARTrck',
    'odtrack': 'ODTrack',
    'ostrack': 'OSTrack',
    'mixformer_cvt_online': 'MIXFormer',
    'mixformer2_vit_online': 'MIXFormerV2',
}
tracker_data = {}
for key, subdir in trackers.items():
    tracker_path = results_base / f'{key}/{subdir}/{test_set}.txt'
    tracker_data[key] = load_text(tracker_path, delimiter='\t') # some tracker may use delimiter = ',' to split result index

# Prepare video writer ,if you do not need this, annotate them
# first_frame = cv2.imread(img_paths[0])
# height, width, layers = first_frame.shape
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' if .avi
# video = cv2.VideoWriter(test_set + '.mp4', fourcc, 30, (width, height))

# Drawing rectangles on images
for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)

    # Draw ground truth rectangle (red)
    gt = list(map(int, ground_truth[i]))
    cv2.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 255, 0), 4) # green

    # Draw rectangles for each tracker
    colors = {'PU-Artrack_seq': (0, 0, 255) , # red
              'PU-OSTrack': (0, 255, 255), # yellow
              'artrack_seq': (255,255,0), #light green
              'odtrack': (255, 0, 0), # blue
              'ostrack':(255,0,255), # pink
              'mixformer_cvt_online': (255, 255, 255),  # white
              'mixformer2_vit_online': (0, 0, 0), # black
              # set the rectangle color of tracker at here ......
              }
    legend_y = 30  # Starting y-coordinate for the legend
    for tracker, data in tracker_data.items():
        track = list(map(int, data[i]))
        cv2.rectangle(img, (track[0], track[1]), (track[0] + track[2], track[1] + track[3]), colors[tracker], 4)

    cv2.putText(img, 'Sequence Name: ' + test_set, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    legend_y += 30# Increment y-coordinate for next legend entr

    cv2.putText(img,'GT', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    legend_y += 30

    # Iterate over the colors dictionary to print each tracker name in its respective color
    for tracker_name, color in colors.items():
        if tracker_name not in trackers.keys():
            continue
        display_name = name_map.get(tracker_name)
        cv2.putText(img, display_name, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        legend_y += 30

    # video.write(img)  # Write frame to video

    cv2.imshow("Tracking", img)
    # Wait indefinitely until a key is pressed, check if it's 'enter', 'space', or 'q'
    key = cv2.waitKey(0)
    if key == 13 or key == 32:  # ASCII values for Enter and Space respectively
        continue  # Go to the next frame
    elif key & 0xFF == ord('q'):
        break  # Exit the loop

# video.release()  # Release the video writer
cv2.destroyAllWindows()  # Close all OpenCV windows


