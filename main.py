import asyncio
import os
import cv2
import numpy as np
import mediapipe as mp
import ffmpeg
from glob import glob

mp_pose = mp.solutions.pose


def extract_wobbles(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    skip_until_frame = -1
    wobble_times = []

    prev_angle = None
    prev_center = None
    prev_right_wrist = None
    prev_left_wrist = None
    prev_left_knee = None
    prev_right_knee = None

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while cap.isOpened():
            if frame_idx < skip_until_frame:
                frame_idx += 1
                cap.grab()
                continue

            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                cx = np.mean([left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x])
                cy = np.mean([left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y])
                center = np.array([cx, cy])

                dx = right_shoulder.x - left_shoulder.x
                dy = right_shoulder.y - left_shoulder.y
                angle = np.degrees(np.arctan2(dy, dx))

                right_wrist = lm[mp_pose.PoseLandmark.RIGHT_THUMB]
                left_wrist = lm[mp_pose.PoseLandmark.LEFT_THUMB]

                # 손목 속도 계산
                right_speed = np.linalg.norm([
                    right_wrist.x - prev_right_wrist.x,
                    right_wrist.y - prev_right_wrist.y
                ]) if prev_right_wrist else 0

                left_speed = np.linalg.norm([
                    left_wrist.x - prev_left_wrist.x,
                    left_wrist.y - prev_left_wrist.y
                ]) if prev_left_wrist else 0

                max_speed = max(right_speed, left_speed)

                left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]

                # 무릎 좌표 변화량 (속도)
                if prev_left_knee and prev_right_knee:
                    left_leg_speed = np.linalg.norm([
                        left_knee.x - prev_left_knee.x,
                        left_knee.y - prev_left_knee.y
                    ])
                    right_leg_speed = np.linalg.norm([
                        right_knee.x - prev_right_knee.x,
                        right_knee.y - prev_right_knee.y
                    ])
                    leg_speed = max(left_leg_speed, right_leg_speed)
                else:
                    leg_speed = 0

                if prev_angle is not None and prev_center is not None:
                    if max_speed > 0.5 or leg_speed > 0.5:
                        time = frame_idx / fps
                        print(f"🎯 타격+휘청 감지 at {time:.2f}s in {os.path.basename(video_path)}")
                        wobble_times.append(time)

                        # score = angle_diff + max_speed * 50
                        # if score > 300:
                        #     skip_sec = 5
                        # elif score > 200:
                        #     skip_sec = 3
                        # else:
                        #     skip_sec = 2

                        skip_sec = 3
                        skip_until_frame = frame_idx + int(skip_sec * fps)

                        output_path = os.path.join(output_dir, f"wobble_{len(wobble_times)}.mp4")
                        (
                            ffmpeg
                            .input(video_path, ss=time, t=skip_sec)
                            .output(output_path)
                            .run(overwrite_output=True, quiet=True)
                        )

                prev_angle = angle
                prev_center = center
                prev_right_wrist = right_wrist
                prev_left_wrist = left_wrist
                prev_left_knee = left_knee
                prev_right_knee = right_knee

            frame_idx += 1

    cap.release()


def merge_all_clips(base_output_dir, final_output_path):
    all_clips = []
    for dirpath in sorted(glob(os.path.join(base_output_dir, "*"))):
        all_clips += sorted(glob(os.path.join(dirpath, "wobble_*.mp4")))

    if not all_clips:
        print("⚠️ 추출된 클립이 없습니다.")
        return

    with open("concat_list.txt", "w", encoding="utf-8") as f:
        for path in all_clips:
            f.write(f"file '{os.path.abspath(path)}'\n")

    (
        ffmpeg
        .input("concat_list.txt", format="concat", safe=0)
        .output(final_output_path, c="copy")
        .run(overwrite_output=True)
    )
    os.remove("concat_list.txt")
    print(f"✅ 모든 클립 합쳐서 저장 완료: {final_output_path}")


if __name__ == "__main__":
    input_dir = "input_videos"
    output_base = "output_clips"
    os.makedirs(output_base, exist_ok=True)

    video_files = glob(os.path.join(input_dir, "*.mp4"))

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_base, video_name)
        extract_wobbles(video_path, video_output_dir)

    merge_all_clips(output_base, "final_highlight.mp4")
