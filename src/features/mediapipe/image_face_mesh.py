import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


image = cv2.imread("../../../data/examples/me.jpg")

# For static images:
IMAGE_FILES = [image]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
print("0000")
with mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
) as face_mesh:
    print("222")
    for idx, file in enumerate(IMAGE_FILES):
        print("333")
        # image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
        continue
    annotated_image = image.copy()

    for face_landmarks in results.multi_face_landmarks:
        print("face_landmarks:", face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    cv2.imshow("Output", annotated_image)
    cv2.waitKey(0)
    cv2.imwrite(
        "../../../data/processed/examples/mediapipe_annotated_image"
        + str(idx)
        + ".png",
        annotated_image,
    )
