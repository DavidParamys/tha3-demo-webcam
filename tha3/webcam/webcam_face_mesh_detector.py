import traceback
import cv2
import numpy as np
import mediapipe as mp
import time

RESIZE_SOLUATION = (320, 320)

class FaceMeshDetector(object):

    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        super()
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1,
                                                 refine_landmarks=True,
                                                 min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findMaceMesh(self, frame, isDraw=True, is_resize=False):
        if is_resize:
            frame = cv2.resize(frame, RESIZE_SOLUATION)

        # Face tracking
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)
        faceLmsList = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if isDraw:
                    self.mpDraw.draw_landmarks(image=frame,
                                               landmark_list=faceLms,
                                               connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mpDrawingStyles
                                               .get_default_face_mesh_tesselation_style())

                    self.mpDraw.draw_landmarks(image=frame,
                                               landmark_list=faceLms,
                                               connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mpDrawingStyles
                                               .get_default_face_mesh_contours_style())

                    self.mpDraw.draw_landmarks(image=frame,     # 繪製到 output
                                               landmark_list=faceLms,
                                               connections=self.mpFaceMesh.FACEMESH_IRISES,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mpDrawingStyles
                                               .get_default_face_mesh_iris_connections_style())
                faceLmsList.append(faceLms)
        return frame, faceLmsList


if __name__ == '__main__':
    # Unit test
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    ret, _ = cap.read()
    while ret:
        # Get frame
        ret, frame = cap.read()

        # Detect
        frame, faceLmsList = detector.findMaceMesh(frame=frame,
                                                   isDraw=True)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime

        # Show
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
