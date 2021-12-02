import json

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from protobuf_to_dict import protobuf_to_dict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


video_output = "../../../data/processed/examples/"
video_input = "../../../data/examples/RostoIntensidade-01Primeira-Acalmar.avi"
video_name = video_input.split("/")[-1]

df_keys = pd.DataFrame(columns=["frame", "video_name", "keys"])


cap = cv2.VideoCapture(video_input)

if cap.isOpened() == False:
    print("Error opening video stream or file")

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if success:
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )

                    keypoints = protobuf_to_dict(face_landmarks)
                    # arr_external = np.array([[0,0,0]])

                    print("Frame " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    print("Qt Keys " + str(len(keypoints["landmark"])))

                    arr_external = np.array(
                        [
                            [
                                keypoints["landmark"][0]["x"],
                                keypoints["landmark"][0]["y"],
                                keypoints["landmark"][0]["z"],
                            ]
                        ]
                    )

                    for i in range(1, len(keypoints["landmark"])):
                        arr_internal_ = np.array(
                            [
                                [
                                    keypoints["landmark"][i]["x"],
                                    keypoints["landmark"][i]["y"],
                                    keypoints["landmark"][i]["z"],
                                ]
                            ]
                        )

                        arr_external = np.concatenate(
                            (arr_external, arr_internal_), axis=0
                        )

                    new_row = {
                        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        "video_name": video_name,
                        "keys": arr_external,
                    }

                    print("------------------------")
                    df_keys = df_keys.append(new_row, ignore_index=True)

            # print(arr_external)
            print(arr_external.shape)
            cv2.imshow("MediaPipe FaceMesh", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            break

df_keys.to_csv(
    str(video_output) + str(video_name.split(".")[0]) + str(".csv"), sep=";"
)

cap.release()


# keypoints = []
# for data_point in face_landmarks.landmark:
# 	keypoints.append({
# 						'X': data_point.x,
# 						'Y': data_point.y,
# 						'Z': data_point.z,
# 						'Visibility': data_point.visibility,
# 						})
