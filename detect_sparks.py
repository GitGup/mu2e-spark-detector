import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import argparse
import time

sns.set_theme(style="white")

frame_count = 0
spark_count = 0

def process_video(path):
    global frame_count
    cap = cv2.VideoCapture(path)

    frame_dir = "AMB_Sparks"

    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
        
    if not cap.isOpened():
        print("Error")
        exit()

    prev_frame = None
    frame_skip_threshold = 5
    skip_counter = 0

    while cap.isOpened():
        
        ret, curr_frame = cap.read()

        if ret:
            #if the prev frame is similar to curr
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, curr_frame)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                non_zero_count = np.count_nonzero(thresh)
                if non_zero_count < frame_skip_threshold:
                    skip_counter += 1
                    continue
                
            detect_spark(curr_frame)

            prev_frame = curr_frame
            frame_count += 1
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Done Processing!")

def extract_frame_number(file_path):
    match = re.search(r'frame_(\d+)', file_path)
    return int(match.group(1)) if match else None

#detect AMB spark given an image/frame
def detect_spark(image):
    global frame_count
    global spark_count

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #adjsut mask if we miss sparks
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    found_spark = False
    for contour in contours:
        if cv2.contourArea(contour) > 25 and found_spark == False:
            
            #means we found a spark so draw a red square
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            #save the sparked AMB image/frame
            output_dir = "AMB_Sparks"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            spark_image_path = os.path.join(output_dir, f"AMB_Spark_{frame_count}.png")
            plt.imsave(spark_image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            found_spark = True

            print(f"Spark Detected on frame {frame_count}")
            spark_count += 1

    found_spark = False

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Process video and detect sparks in images.")
    parser.add_argument('video_path', type=str, help='Path to the video file')

    args = parser.parse_args()

    raw_video_path = args.video_path
    
    #check if arg dir given is a directory or a path
    if os.path.isdir(raw_video_path):
        print("Getting Videos...")
        for filename in os.listdir(raw_video_path):
            if filename.endswith('.mp4') or filename.endswith('.AVI'):
                video_path = os.path.join(raw_video_path, filename)
                print("Processing video from: ", video_path)
                process_video(video_path)

    elif os.path.isfile(raw_video_path) and raw_video_path.endswith('.mp4') or raw_video_path.endswith('.AVI'):
        print("Processing video from: ", raw_video_path)
        process_video(raw_video_path)

    else:
        ("Please provide a video path or path to dir containing videos")
    
end_time = time.time()
time_elapsed = end_time - start_time
print(f"Finished in {time_elapsed:.2f} seconds!")
print(f"Total Spark Count: {spark_count}")