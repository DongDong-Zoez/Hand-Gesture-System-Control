import argparse
import os
import time
import cv2 
import mediapipe as mp
import matplotlib.pyplot as plt

def argparser() -> argparse.Namespace:
    """
    Function that defines and parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Face mask detection using YOLOv8 and OpenCV")
    parser.add_argument("-dc", "--detection_confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    parser.add_argument("-tc", "--tracking_confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

    parser.add_argument("--num_hands", type=int, default=1, help="number hands to detect")
    parser.add_argument("--num_detections", type=int, default=100, help="number hands to detect")

    parser.add_argument("--cls_idx", type=int, default=5, help="number hands to detect")
    parser.add_argument("--save_path", type=str, help="path to save image")

    parser.add_argument("--h", type=int, default=720, help="window height for web camera")
    parser.add_argument("--w", type=int, default=1280, help="window width for web camera")
    return parser.parse_args()

class AIVideo(object):

    def __init__(self, h, w, min_detection_confidence, min_tracking_confidence, num_hands, num_detections, save_path, format="yolo", cls_idx=0):

        # Video frame dimensions
        self.h = h
        self.w = w

        # MediaPipe Hands module
        self.mpHands = mp.solutions.hands

        # Confidence thresholds and number of hands to detect
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_hands = num_hands

        # MediaPipe Drawing utilities
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles

        self.num_detections = num_detections
        self.save_path = save_path
        self.format = format
        self.cls_idx = cls_idx


    def liveStream(self):
        # Open webcam for live streaming
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        # Initialize time variables
        cTime = 0
        pTime = 0

        # Create necessary directories for data collection
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(os.path.join(self.save_path, "train")):
            os.makedirs(os.path.join(self.save_path, "train", "images"))
            os.makedirs(os.path.join(self.save_path, "train", "labels"))
        if not os.path.exists(os.path.join(self.save_path, "val")):
            os.makedirs(os.path.join(self.save_path, "val", "images"))
            os.makedirs(os.path.join(self.save_path, "val", "labels"))
        nd = 0

        # Initialize MediaPipe Hands module
        with self.mpHands.Hands(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            max_num_hands=self.num_hands,
        ) as hands:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Empty frame.")
                    continue

                # Process frame with MediaPipe Hands
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frameRGB = cv2.flip(frameRGB, 1)
                result = hands.process(frameRGB)
                frame = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks:
                    if nd > self.num_detections:
                        return 
                    for handLms in result.multi_hand_landmarks:
                        # Draw hand landmarks and connections
                        self.mpDraw.draw_landmarks(
                            frame,
                            handLms,
                            self.mpHands.HAND_CONNECTIONS,
                            self.mpDrawStyle.get_default_hand_landmarks_style(),
                            self.mpDrawStyle.get_default_hand_connections_style(),
                        )

                        # yolo format [x (center), y (center), w, h]
                        if self.format == "yolo":
                            
                            # extract (x, y) coordinate
                            landmarks += [lm.x for lm in handLms.landmark] + [lm.y for lm in handLms.landmark]
                            xs = landmarks[0::2]
                            ys = landmarks[1::2]

                            # calcu top left corner (xmin, ymin) and bot right corner (xmax, ymax)
                            xmin, ymin = min(xs), min(ys)
                            xmax, ymax = max(xs), max(ys)

                            # calc yolo format [x (center), y (center), w, h]
                            xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
                            w, h =  (xmax - xmin), (ymax - ymin)

                            # write annotation txt file with lines "cls xc yc w h lms..."
                            landmarks = [str(k) for k in landmarks]
                            root = os.path.join(self.save_path, "train") if nd % 2 else os.path.join(self.save_path, "val")
                            with open(os.path.join(root, "labels", f"{self.cls_idx}_{nd}.txt"), 'w') as f:
                                lines = f"{self.cls_idx} " + " ".join([str(xc), str(yc), str(w), str(h)]) + " " + " ".join(landmarks)
                                f.write(lines)
                                plt.imsave(os.path.join(root, "images", f"{self.cls_idx}_{nd}.jpg"), frameRGB)
                                nd += 1

                # Calculate and display frames per second
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # Display the frame
                cv2.imshow('img', frame)

                # Check for keyboard interrupt to stop the program
                if cv2.waitKey(1) == ord('q'):
                    print("Keyboard Interrupt.")
                    return
            cap.release()

def main():
    args = argparser()
    live = AIVideo(
        h=args.h,
        w=args.w,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        num_hands=args.num_hands,
        save_path=args.save_path,
        num_detections=args.num_detections,
        cls_idx=args.cls_idx,
    )
    live.liveStream()

if __name__ == "__main__":
    main()
