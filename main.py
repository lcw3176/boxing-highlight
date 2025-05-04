import os
import ffmpeg
import cv2
import mediapipe as mp
import numpy as np

video_path = '1round.mp4'
output_dir = video_path.replace(".mp4", "") + '_highlights'
os.makedirs(output_dir, exist_ok=True)

# 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0
wobble_frames = []
time_to_record = 5

prev_angle = None
prev_center = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        # 어깨-엉덩이 중심
        cx = np.mean([left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x])
        cy = np.mean([left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y])
        center = np.array([cx, cy])

        # 어깨 기울기
        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        angle = np.degrees(np.arctan2(dy, dx))

        if prev_angle is not None and prev_center is not None:
            angle_diff = abs(angle - prev_angle)
            move_dist = np.linalg.norm(center - prev_center)

            # 휘청 조건: 어깨 각도 급변 + 중심점 급변
            if angle_diff > 20 and move_dist > 0.1:
                time = frame_idx / fps

                if wobble_frames:
                    before_time = int(wobble_frames[-1])
                    now_time = int(time)
                    already_exist = False

                    for i in range(before_time, before_time + time_to_record):
                        if i == now_time:
                            already_exist = True
                            break

                    if already_exist:
                        continue

                wobble_frames.append(time)
                print(f"휘청 감지 at {time:.2f}s")
        # | 상황               | 예상 `move_dist` |
        # | ---------------- | -------------- |
        # | 가만히 있음           | 0.001–0.01   |
        # | 팔 휘두름 등 소폭 이동    | 0.02–0.04    |
        # | 몸이 휘청, 중심 이동     | 0.05–0.1     |
        # | 완전히 쓰러지거나 화면 벗어남 | 0.2–0.5 이상   |

        prev_angle = angle
        prev_center = center

    frame_idx += 1

cap.release()

# 중복 제거 후 하이라이트 저장
wobble_times = sorted(set(int(t) for t in wobble_frames))


# 하이라이트 영상 추출
highlight_files = []
for i, sec in enumerate(wobble_times):
    start = max(sec - 1, 0)
    output_path = os.path.join(output_dir, f"wobble_{i+1}.mp4")
    highlight_files.append(output_path)

    (
        ffmpeg
        .input(video_path, ss=start, t=time_to_record)
        .output(output_path)
        .run(overwrite_output=True)
    )

# 여러 개의 하이라이트 영상 합치기
concat_file = os.path.join(output_dir, "file_list.txt")

# 합칠 파일 목록 작성
with open(concat_file, 'w') as f:
    for file in highlight_files:
        f.write(f"file '{os.path.abspath(file)}'\n")

# 합친 영상 출력 경로
final_output_path = os.path.join(output_dir, "final_highlight.mp4")

# 영상 합치기
ffmpeg.input(concat_file, format='concat', safe=0).output(final_output_path, c='copy').run()

# 완료
print(f"최종 하이라이트 영상이 {final_output_path}에 저장되었습니다.")
