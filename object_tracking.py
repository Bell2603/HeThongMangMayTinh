import cv2
import torch
import numpy as np
import argparse, os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO

# BIẾN TOÀN CỤC
FRAME_WIDTH=30        # Đang cho là đoạn đường dài 100m và rộng 30m
FRAME_HEIGHT=100
MAX_SPEED_REF = 250.0  # 60
#MAX_DENSITY_REF = 10.0
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        nargs="?",
        default="content/highway.mp4",
        #default="content/Vid1.mp4",
        #default="content/Video_1.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        help="path to output video",
        default="content/output.mp4"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50, # nhiều nhiễu thì tăng lên, bỏ lỡ vật thì giảm xuống
        help="confidence threshold",
    )
    parser.add_argument(
        "--blur_id",
        type=int,
        default=None,
        help="class ID to apply Gaussian Blur",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,  # None, hiện chỉ đang track xe máy
        help="class ID to track",  # python object_tracking.py --class-id 3 2
    )
    opt = parser.parse_args()
    return opt



def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h # khoanh vùng vật thể bằng cách xác định điểm trên bên trái và dưới bên phải

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left  x, y vẽ các đường viên bao quanh 
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img  

def calculate_speed(distance, fps):
    return (distance *fps)*3.6


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  # PYTAGORE


def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame 

# Vận tốc trung bình
def average_speed_all_vehicles_fair(speed_accumulator):
    vehicle_avgs = []
    for speeds in speed_accumulator.values():
        if len(speeds) > 0:
            vehicle_avgs.append(sum(speeds) / len(speeds))

    if len(vehicle_avgs) == 0:
        return 0.0

    return sum(vehicle_avgs) / len(vehicle_avgs)

def count_motorcycle_car_simple(tracks):
    
    motorcycle_count = 0
    car_count = 0
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        class_id = track.get_det_class()
        
        if class_id == 3:  # Xe máy
            motorcycle_count += 1
        elif class_id == 2:  # Ô tô
            car_count += 1
    
    return motorcycle_count, car_count


def main(_argv):
    
    SOURCE_POLYGONE = np.array([[18, 550], # góc dưới trái
                                 [1852, 608], # góc dưới phải
                                 [1335, 370], # góc trên phải
                                 [534, 343]], # góc trên trái
                                dtype=np.float32)
    
    """SOURCE_POLYGONE = np.array([[523, 607], # góc dưới trái
                                 [1168, 971], # góc dưới phải
                                 [1628, 482], # góc trên phải
                                 [1004, 302]], # góc trên trái
                                dtype=np.float32)
    """
    """SOURCE_POLYGONE = np.array([[716, 826], # góc dưới trái, lấy làn bên trái
                                 [1407, 826], # góc dưới phải
                                 [1364, 427], # góc trên phải
                                 [734, 428]], # góc trên trái
                                dtype=np.float32)
    """
    BIRD_EYE_VIEW = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT],[0, FRAME_HEIGHT]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(SOURCE_POLYGONE, BIRD_EYE_VIEW)


    # Initialize the video capture
    video_input = opt.video

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return
    
    # lấy thông số của video, HD hay full HD,...
    frame_generator = read_frames(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    pts = SOURCE_POLYGONE.astype(np.int32) 
    pts = pts.reshape((-1, 1, 2))     

    polygon_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [pts], 255)
    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(opt.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)   # quá 50 frame không thấy thì xoá track
    # Load YOLO model
    model = YOLO("yolov10n.pt")
    # Load the COCO class labels
    classes_path = "configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 
    # FPS calculation variables
    frame_count = 0              # đếm số frame đã xử lý
    start_time = time.time()
    prev_positions={}            # vị trí trước đó của ID
    speed_accumulator={}
    
    while True:
        try:
            frame = next(frame_generator)       # lấy từng fram từ video
        except StopIteration:                   # hết video thì thoát
            break
        # Run model on each frame
        with torch.no_grad():       # không tính toán gradient để tiết kiệm bộ nhớ
            results = model(frame)  # chạy mô hình YOLO để phát hiện đối tượng trong frame
        detect = []
        for pred in results:        # duyệt qua các kết quả của yolo
            for box in pred.boxes:  # duyệt qua các hộp giới hạn được phát hiện    
                x1, y1, x2, y2 = map(int, box.xyxy[0] )   # tọa độ hộp giới hạn X1, Y1 GÓC TRÊN TRÁI, X2, Y2 GÓC DƯỚI PHẢI
                confidence = box.conf[0]     # độ tin cậy của phát hiện
                label = box.cls[0]           # lấy nhãn lớp của đối tượng được phát hiện

                # Filter out weak detections by confidence threshold and class_id
                if opt.class_id is None:    # nếu không chỉ định class_id thì chỉ lọc theo độ tin cậy
                    if confidence < opt.conf:
                        continue
                else:
                    if class_id != opt.class_id or confidence < opt.conf:   # nếu có chỉ định class_id thì lọc theo cả class_id và độ tin cậy
                        continue            
                    
                if polygon_mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255:    # lấy vị trí trung tâm xem có nằm trong vùng quan tâm không
                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label)])            
        tracks = tracker.update_tracks(detect, frame=frame)
        for track in tracks:
            if not track.is_confirmed():   # dùng để tránh bị nhiễu
                continue
            track_id = track.track_id      # lấy ID của từng xe
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            if polygon_mask[(y1+y2)//2,(x1+x2)//2] == 0:  # nếu tâm không nằm trong vùng quan tâm thì xoá track
                tracks.remove(track)
            color = colors[class_id]       # màu sắc cho từng lớp đối tượng
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"   # hiển thị ID và tên lớp
            center_pt = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], M)    # chuyển đổi từ điểm ảnh ở 2D sang 3D tức là môi trường thật
            if track_id in prev_positions:    # kiểm tra xem ID đã có vị trí trước đó chưa
                prev_position = prev_positions[track_id]   # lấy vị trí trước đó
                distance = calculate_distance(prev_position, transformed_pt[0][0])  # TÍNH TOÁN KHOẢNG CÁCH ĐÃ ĐI QUA
                speed = calculate_speed(distance, fps)  # TÍNH TOÁN VẬN TỐC
                if track_id in speed_accumulator:
                    speed_accumulator[track_id].append(speed) # lưu trữ vận tốc để tính trung bình
                    if len(speed_accumulator[track_id]) > 100:
                        speed_accumulator[track_id].pop(0)  # giữ lại 100 giá trị gần nhất
                else:
                    speed_accumulator[track_id] = []  # khởi tạo danh sách lưu trữ vận tốc mới cho đối tượng mới
                    speed_accumulator[track_id].append(speed)
            prev_positions[track_id] = transformed_pt[0][0]
            # Draw bounding box and text
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3, rect_thickness=1, rect_color=(B, G, R), line_color=(R, G, B))
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            # vận tốc trung bình
            moto_count, car_count = count_motorcycle_car_simple(tracks)
            avg_speed_all = average_speed_all_vehicles_fair(speed_accumulator) # tính vận tốc trung bình của tất cả các xe
            # density = len(speed_accumulator) / MAX_DENSITY_REF  # mật độ giao thông
            density = (moto_count * 3.0 + car_count * 18.0) / (FRAME_WIDTH * FRAME_HEIGHT)  # mật độ giao thông, xe máy tính 3 đơn vị, ô tô tính 18 đơn vị
            CI = 0.5 * density + 0.5 * (1 - avg_speed_all / MAX_SPEED_REF)   # chỉ số tắc nghẽn giao thông
            CI = max(0, min(1, CI))   # giới hạn CI trong khoảng 0-1
            if CI < 0.3: 
            #level = 0
                level, label = 0, "Thong thoang" # Thông thoáng, 0
            elif CI < 0.6:
            #level = 1
                level, label = 1, "Dong" # Đông, 1
            else:
                #level = 2
                level, label = 2, "Tac nghen" #Tắc nghẽn, 2
            # Vẽ hình chữ nhật màu đen đè lên vùng text cũ
            cv2.rectangle(frame, (25, 15), (700, 150), (0, 0, 0), -1)  # vẽ hộp đen để làm nền cho văn bản
            cv2.putText(
            frame,
            f"Avg speed (all vehicles): {avg_speed_all:.1f} km/h",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
             1,
            (0, 165, 255),
            2
            )
            cv2.putText(frame, f"CI: {CI:.2f}", (30, 70), # chỉ số tắc nghẽn giao thông
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(frame, f"Level {level}: {label}", (30, 100), # mức độ tắc nghẽn
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            #####
            if track_id in speed_accumulator :
                avg_speed = sum(speed_accumulator[track_id]) / len(speed_accumulator[track_id])
                #print(avg_speed)
                cv2.rectangle(frame, (x1 - 1, y1-40 ), (x1 + len(f"Speed: {avg_speed:.0f} km/h") * 10, y1-20), (0, 0, 255), -1)
                cv2.putText(frame, f"Speed: {avg_speed:.0f} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Apply Gaussian Blur
            if opt.blur_id is not None and class_id == opt.blur_id:
                print("true")
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, f"Height: {FRAME_HEIGHT}", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {FRAME_WIDTH}", (1530, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('speed_estimation', frame)
        writer.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"FPS: {fps_calc:.2f}")
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)