from utils import read_video, save_video
from trackers import PlayerTracker

def main():
    # Read Video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)

    # Detect PLayers
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path='tracker_stubs/player_detections.pkl'
                                                     )


    # Draw output
    
    ## Draw Player Bouding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    
    output_video_path = 'output_videos/output_video.avi' 
    save_video(output_video_frames, output_video_path)



if __name__ == '__main__':
    main()