import argparse
from threading import Thread
import time
import cv2 
import os
from ultralytics import YOLO

def argparser() -> argparse.Namespace:
    """
    Function that defines and parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Face mask detection using YOLOv8 and OpenCV")
    parser.add_argument("--max_det", type=int, default=1, help="number hands to detect")
    parser.add_argument("--action_time", type=float, default=2, help="number hands to detect")
    parser.add_argument("--src", type=int, default=0, help="number hands to detect")
    parser.add_argument("--h", type=int, default=720, help="window height for web camera")
    parser.add_argument("--w", type=int, default=1280, help="window width for web camera")
    return parser.parse_args()

class ActionSpace(object):

    # Dictionary to map action indices to corresponding functions
    action_mapping = {
        0: lambda: os.system("taskkill /f /im code.exe"), # close vscode 
        1: lambda: os.system("code ."), # open vscode 
        2: lambda: os.system("Calc"), # open calculator
        3: lambda: os.system("notepad"), # open txt note
        4: lambda: os.system("write"), # open word
        5: lambda: os.system("control"), # open control panel
    }

class StreamController(object):

    def __init__(self, h, w, src=0):
        # Initialize video stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Read the first frame
        self.success, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread for capturing frames
        Thread(target=self.get, args=()).start()
        return self
    
    def get(self):
        # Continuously capture frames from the video stream
        while not self.stopped:
            if not self.success:
                self.stop()
            else:
                self.success, self.frame = self.stream.read()

    def stop(self):
        # Stop the thread and release the video stream
        self.stopped = True

class AIVideo(object):

    def __init__(self, src, h, w, max_det, action_time):
        # Video frame dimensions
        self.h = h
        self.w = w
        self.max_det = max_det

        # Initialize the video stream controller
        self.cap = StreamController(h, w, src).start()


        # Initialize YOLO model
        self.model = YOLO('best.pt')

        # Set action space
        self.action = ActionSpace.action_mapping
        self.action_time = action_time
        
    def liveStream(self):
        # Initialize time variables
        cTime = 0
        pTime = 0
        aTime = 0

        # Previous detected class index
        pcls = -1

        while self.cap.stream.isOpened():
            success, frame = self.cap.success, self.cap.frame
            if not success:
                print("Empty frame.")
                continue
            
            frame = cv2.flip(frame, 1)

            # Run YOLO model on the frame
            result = self.model(frame, max_det=self.max_det)[0].cpu().numpy()
            frame = result.plot()

            if result:
                boxes = result.boxes  
                if boxes:
                    xyxy = boxes.xyxy[0]
                    cls_idx = boxes.cls[0]
                    _ = boxes.conf 
                    
                    if cls_idx == pcls:
                        # Accumulate time if the same class is detected
                        aTime += time.time() - pTime
                        end_angle = int(aTime / self.action_time * 360)
                        cv2.ellipse(frame, (int(xyxy[0]) + 20, int(xyxy[1]) - 70), (20,20), 0, 0, end_angle, (0,255,0), thickness=3)
                        cv2.putText(frame, f"loading...", (int(xyxy[0]) + 50, int(xyxy[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                        if aTime > self.action_time:
                            # Execute the action if the time threshold is reached
                            self.action[cls_idx]()
                            aTime = 0
                    else:
                        aTime = 0
                    pcls = cls_idx
                
                keypoints = result.keypoints  
                if keypoints:
                    _ = keypoints.xy[0]

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime


            # Display the frame
            cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('img', frame)

            # Check for keyboard interrupt to stop the program
            if cv2.waitKey(1) & 0xFF == ord("q") and self.cap.stopped:
                self.cap.stop()
                print("Keyboard Interrupt.")
        
        # Release the video stream and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    args = argparser()
    live = AIVideo(
        h=args.h,
        w=args.w,
        action_time=args.action_time,
        src=args.src,
        max_det=args.max_det,
    )
    live.liveStream()

if __name__ == "__main__":
    main()
