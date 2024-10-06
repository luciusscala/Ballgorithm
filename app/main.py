from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_tranformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from random import randint

app = FastAPI()

# Directory to save the uploaded and processed files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_videos'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve the HTML homepage
@app.get("/", response_class=HTMLResponse)
async def read_home():
    with open("app/static/index.html") as f:  # Adjust the path as necessary
        return f.read()


@app.post('/convert/')
async def convert_video(file: UploadFile = File(...)):
    if file.filename.endswith('.mp4'):
        # Save the uploaded MP4 file
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(input_path, 'wb') as f:
            f.write(await file.read())

        # Call the main processing function with the uploaded video
        output_path = os.path.join(OUTPUT_FOLDER, file.filename.replace('.mp4', '.avi'))
        main(input_path, output_path)

        # Return the processed AVI file
        return FileResponse(output_path, media_type="video/x-msvideo", filename=output_path)

    return {"error": "Invalid file type"}

def main(input_path, output_path):
    video_frames = read_video(input_path)

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=False,
                                        stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    
    #intepret camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=False,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjusted_position_to_tracks(tracks, camera_movement_per_frame)

    #intepret angle of camera
    view_transformer = ViewTransformer()
    view_transformer.add_tranformed_position_to_tracks(tracks)

    #predict ball positions when empty
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    #add spead and distance
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    #assign team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #assign ball
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(randint(1,2))
        
    
    team_ball_control = np.array(team_ball_control)

    
         
    #draw annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)


    #output
    save_video(output_video_frames, output_path)

if __name__ == '__main__':
    main()


