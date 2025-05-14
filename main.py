import cv2
from ultralytics import YOLO
import supervision as sv

def plate_detection_model(video_path, model_path, device='cpu'):
    """
    Detects plates in a video using pre-trained yolo models and tracked using ByteTrack.

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
    first_frame = next(frame_generator)
    height, width = first_frame.shape[:2]
    
    # Reset generator to first frame again
    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    for frame in frame_generator:
        # Inference
        result = model.predict(frame, verbose=False)[0]

        # Filter detections by car classes
        detections = sv.Detections.from_ultralytics(result)

        # Track objects
        tracked = tracker.update_with_detections(detections)
        frame = frame.copy()
        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            bbox = (x1, y1, x2, y2)
            track_id = int(tracked.tracker_id[i])
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            results.append((frame_number, track_id, confidence, frame, bbox))
        frame_number += 1

    # return list(best_results.values()), width, height
    return results, width, height

def crop_plate_box(detections, output_size=(640, 640)):
    """
    Crops and resizes license plate regions from frames.

    Args:
        detections (list): List of tuples from plate_detection_model.
        output_size (tuple): Desired size for output cropped images.

    Returns:
        List of tuples: (frame_number, car_id, cropped_resized_plate_image)
    """
    cropped_plates = []

    for frame_num, car_id, _, frame_img, plate_bbox in detections:
        x1, y1, x2, y2 = plate_bbox

        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_img.shape[1], x2)
        y2 = min(frame_img.shape[0], y2)

        # Crop and resize
        plate_crop = frame_img[y1:y2, x1:x2]
        resized_plate = cv2.resize(plate_crop, output_size)
        cropped_plates.append((frame_num, car_id, resized_plate))

    return cropped_plates

def detect_plate_number(detections, text_model_path, device='cpu'):
    """
    Runs text detection on cropped license plates.

    Args:
        detections (list): Output from plate_detection_model.
        text_model_path (str): Path to the text/number YOLO model.
        device (str): Device for inference.

    Returns:
        List of tuples: (frame_number, car_id, detected_texts)
    """
    cropped_plates = crop_plate_box(detections)
    text_model = YOLO(text_model_path).to(device=device)
    results = []

    for frame_num, car_id, plate_img in cropped_plates:
        result = text_model.predict(plate_img, verbose=False, conf=0.25, iou=0.40)[0]
        detections = sv.Detections.from_ultralytics(result)
        # Apply NMS to remove overlapping predictions
        nms_detections = detections.with_nms(threshold=0.7, class_agnostic=True)
        texts = []
        for i in range(len(nms_detections.xyxy)):
            
            x1, y1, x2, y2 = map(int, nms_detections.xyxy[i])
            bbox = (x1, y1, x2, y2)
            class_id = int(nms_detections.class_id[i]) if nms_detections.class_id is not None else -1
            label = text_model.model.names[class_id] if class_id in text_model.model.names else "?"

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            texts.append((label, center_x, center_y, bbox))

            texts.sort(key=lambda x: x[1])  # Sort by center_x
        results.append((frame_num, car_id, plate_img, texts))

    return results

def split_text_number_predictions(plate_predictions):
    """
    Splits detected plate components into text and numbers based on label.

    Args:
        detections (list): Output from detect_plate_number()

    Returns:
        List of tuples:
            (frame_number, car_id, plate_img, number_detections, text_detections)
    """
    split_results = []

    for frame_num, car_id, plate_img, texts in plate_predictions:
        numbers = []
        texts_only = []

        for label, _, _, _ in texts:
            if label.isdigit():  # classify digits as numbers
                numbers.append(label)
            else:  # anything else is text
                texts_only.append(label)

        split_results.append((frame_num, car_id, plate_img, numbers, texts_only[::-1]))

    return split_results    

def plate_detection_from_image(image_path, model_path, device='cpu'):
    """
    Detects license plates in a single image using YOLO.
    Args:
        image_path (str): Path to input image.
        model_path (str): Path to YOLOv8 model.
        device (str): Device to run the model on.

    Returns:
        List of (frame_number, car_id, confidence, frame_image, bbox)
    """
    model = YOLO(model_path).to(device=device)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    height, width = image.shape[:2]
    result = model.predict(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    results = []
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        bbox = (x1, y1, x2, y2)
        track_id = i  # Fake ID for single image
        confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        results.append((0, track_id, confidence, image, bbox))  # frame_number = 0
    return results, width, height


if __name__ == "__main__":
    video = "videos/digi.mp4"
    detections, width, height = plate_detection_model(video, model_path='models/Plate_Box_Model.pt', device='cuda')
    plate_predictions = detect_plate_number(detections, text_model_path='models/Plate_Text_Numbers_Model.pt', device='cuda')
    split_results = split_text_number_predictions(plate_predictions)
    for frame_num, car_id, _, numbers, texts in split_results:
        print(f"Frame: {frame_num}, Car ID: {car_id}")
        print(f"  Numbers: {numbers}")
        print(f"  Texts: {texts}")


# if __name__ == "__main__":
#     image_path = "images/1.jpg"

#     detections, width, height = plate_detection_from_image(
#         image_path,
#         model_path='models/Plate_Box_Model.pt',
#         device='cuda'
#     )   
#     print(f"Width: {width}, Height: {height}, Detections: {detections}")

#     plate_predictions = detect_plate_number(
#         detections,
#         text_model_path='models/Plate_Text_Numbers_Model.pt',
#         device='cuda'
#     )

#     split_results = split_text_number_predictions(plate_predictions)
#     for frame_num, car_id, _, numbers, texts in split_results:
#         print(f"Frame: {frame_num}, Car ID: {car_id}")
#         print(f"  Numbers: {numbers}")
#         print(f"  Texts: {texts}")