import os
import ffmpeg
import cv2
import mediapipe as mp
import numpy as np

# 하이라이트 저장 폴더
output_dir = 'wobble_highlights'
os.makedirs(output_dir, exist_ok=True)

# 분석 대상 동영상 리스트 (폴더나 직접 리스트로 지정)
video_files = [
    '1round.mp4',
    '2round.mp4',
    '3round.mp4',
    'spar3.mp4'
]

# 미디어파이프 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

highlight_files = []


def extract_wobbles(video_path, video_index):
    global highlight_files
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    wobble_frames = []
    prev_angle = None
    prev_center = None
    prev_right_wrist = None
    prev_left_wrist = None
    skip_until_frame = -1

    while cap.isOpened():

        if frame_idx < skip_until_frame:
            # 건너뛰기
            frame_idx += 1
            cap.grab()
            continue

        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

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

            # 손목 (오른손) 위치
            right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrist_speed = 0

            left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]

            if prev_angle is not None and prev_center is not None:
                angle_diff = abs(angle - prev_angle)
                move_dist = np.linalg.norm(center - prev_center)

                # 손목 속도 계산
                if prev_right_wrist:
                    wrist_speed = max(wrist_speed, np.linalg.norm([
                        right_wrist.x - prev_right_wrist.x,
                        right_wrist.y - prev_right_wrist.y
                    ]))

                # 손목 속도 계산
                if prev_left_wrist:
                    wrist_speed = max(wrist_speed, np.linalg.norm([
                        left_wrist.x - prev_left_wrist.x,
                        left_wrist.y - prev_left_wrist.y
                    ]))

                # 새로운 휘청 + 타격 조건
                if angle_diff > 20 and move_dist > 0.05 and wrist_speed > 0.1:
                    time = frame_idx / fps
                    print(f"🎯 타격+휘청 감지 at {time:.2f}s")
                    wobble_frames.append(time)

                    # 3초 하이라이트 추출 후 2초간 스킵
                    skip_duration = 2  # seconds
                    skip_until_frame = frame_idx + int(skip_duration * fps)

            prev_angle = angle
            prev_center = center
            prev_right_wrist = right_wrist
            prev_left_wrist = left_wrist

        frame_idx += 1

    cap.release()

    wobble_times = sorted(set(int(t) for t in wobble_frames))

    # 하이라이트 클립 저장
    for i, sec in enumerate(wobble_times):
        start = max(sec - 1, 0)
        out_filename = f"video{video_index}_wobble_{i + 1}.mp4"
        output_path = os.path.abspath(os.path.join(output_dir, out_filename))
        highlight_files.append(output_path)

        (
            ffmpeg
            .input(os.path.abspath(video_path), ss=start, t=3)
            .output(output_path)
            .run(overwrite_output=True)
        )


# 모든 영상에 대해 처리 실행
for idx, video in enumerate(video_files):
    print(f"▶ Processing {video} ...")
    extract_wobbles(video, idx + 1)

# 합칠 파일 리스트 생성
concat_file_path = os.path.join(output_dir, "file_list.txt")
with open(concat_file_path, 'w') as f:
    for path in highlight_files:
        f.write(f"file '{path}'\n")

# 최종 하이라이트 영상으로 합치기
final_output_path = os.path.join(output_dir, "final_highlight.mp4")
ffmpeg.input(concat_file_path, format='concat', safe=0).output(final_output_path, c='copy').run()

print(f"\n✅ 모든 하이라이트를 합친 영상이 저장되었습니다: {final_output_path}")
