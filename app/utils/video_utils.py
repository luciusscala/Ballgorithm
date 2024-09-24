import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while  True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(out_frames, out_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 24, (out_frames[0].shape[1], out_frames[0].shape[0]))
    for frame in out_frames:
        out.write(frame)
    out.release()