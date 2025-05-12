import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

def detect_and_track_cars(video_path, model_path, device='cpu'):
    """
    Detects and tracks cars in a video using YOLOv8 and ByteTrack.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the YOLOv8 model.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        List of (frame_number, car_id, frame_image)
    """
    model = YOLO(model_path).to(device=device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    frame_number = 0
    results = []
    
    for frame in frame_generator:
        # Inference
        result = model.predict(frame, verbose=False)[0]

        # Filter detections by car classes
        detections = sv.Detections.from_ultralytics(result)

        # Track objects
        tracked = tracker.update_with_detections(detections)

        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            track_id = int(tracked.tracker_id[i])
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            results.append((frame_number, track_id, class_id, confidence, frame.copy()))

        frame_number += 1

    return results


if __name__ == "__main__":
    video = "videos/madeup.mp4"
    detections = detect_and_track_cars(video, model_path='models/Car_Detection_Model.pt', device='cpu')

    for frame_num, car_id, class_id, confidence, frame in detections:  # Show first 5 results for demo
        print(f"Frame: {frame_num}, Car ID: {car_id}")
        cv2.imshow("Tracked Frame", frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
