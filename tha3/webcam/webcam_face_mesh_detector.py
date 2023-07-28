import traceback
import cv2
import numpy as np
import mediapipe as mp
import time
import math

RIGHT_EYE_BONE_X = "rightEyeBoneX"
RIGHT_EYE_BONE_Y = "rightEyeBoneY"
RIGHT_EYE_BONE_Z = "rightEyeBoneZ"

LEFT_EYE_BONE_X = "leftEyeBoneX"
LEFT_EYE_BONE_Y = "leftEyeBoneY"
LEFT_EYE_BONE_Z = "leftEyeBoneZ"

HEAD_BONE_X = "headBoneX"
HEAD_BONE_Y = "headBoneY"
HEAD_BONE_Z = "headBoneZ"

RIGHT_EYE_BONE_QUAT = "rightEyeBoneQuat"
LEFT_EYE_BONE_QUAT = "leftEyeBoneQuat"
HEAD_BONE_QUAT = "headBoneQuat"

RESIZE_SOLUATION = (320, 320)

def create_default_ifacialmocap_pose():
    data = {}

    data[HEAD_BONE_X] = 0.0
    data[HEAD_BONE_Y] = 0.0
    data[HEAD_BONE_Z] = 0.0
    data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[LEFT_EYE_BONE_X] = 0.0
    data[LEFT_EYE_BONE_Y] = 0.0
    data[LEFT_EYE_BONE_Z] = 0.0
    data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[RIGHT_EYE_BONE_X] = 0.0
    data[RIGHT_EYE_BONE_Y] = 0.0
    data[RIGHT_EYE_BONE_Z] = 0.0
    data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    return data

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

    def findFaceMesh(self, frame, output, isDraw=False, isResize=False, isShowText=False):
        if isResize:
            frame = cv2.resize(frame, RESIZE_SOLUATION)

        # Face tracking
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(frameRGB)
        ih, iw, ic = frame.shape
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # DEBUG: Draw landmakrs
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
                # Get value directly
                """
                target_ids = [4,
                              195,
                              386]
                for id, lm in enumerate(faceLms.landmark):
                    if id == target_ids[0]:
                        output[HEAD_BONE_X] = lm.x
                        output[HEAD_BONE_Y] = lm.y
                        output[HEAD_BONE_Z] = lm.z
                    elif id == target_ids[1]:
                        output[LEFT_EYE_BONE_X] = lm.x
                        output[LEFT_EYE_BONE_Y] = lm.y
                        output[LEFT_EYE_BONE_Z] = lm.z
                    elif id == target_ids[2]:
                        output[RIGHT_EYE_BONE_X] = lm.x
                        output[RIGHT_EYE_BONE_Y] = lm.y
                        output[RIGHT_EYE_BONE_Z] = lm.z
                """
                # calculate Bone
                face_bone_ids = [10, 162, 389, 136, 365, 152]
                left_bone_ids = [33, 160, 158, 133, 153, 144]
                right_bone_ids = [163, 387, 385, 362, 380, 373]
                face_bone_ids.sort()
                left_bone_ids.sort()
                right_bone_ids.sort()
                nose_2d = []
                nose_3d = []
                face_3d = []
                face_2d = []
                left_eye_3d = []
                left_eye_2d = []
                right_eye_3d = []
                right_eye_2d = []
                for id, lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x * iw), int (lm.y * ih)
                    # DEBUG: Draw Points on frame
                    if isShowText:                       
                        if id in face_bone_ids:
                            cv2.putText(frame, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        else:
                            cv2.putText(frame, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
                    # Get point for head bone
                    if id in face_bone_ids:
                        if id == 1:
                            nose_2d = (lm.x * iw, lm.y * ih)
                            nose_3d = (lm.x * iw, lm.y * ih, lm.z * 3000)
                        face_2d.append([x ,y])
                        face_3d.append([x ,y, lm.z])
                    
                    elif id in left_bone_ids:
                        left_eye_2d.append([x ,y])
                        left_eye_3d.append([x, y, lm.z])
                    elif id in right_bone_ids:
                        right_eye_2d.append([x ,y])
                        right_eye_3d.append([x, y, lm.z])

                # Calculate 
                components= self.calculate_bone(face_2d, face_3d, frame.shape)
                left_components = self.calculate_bone(left_eye_2d, left_eye_3d, frame.shape)
                right_components = self.calculate_bone(right_eye_2d, right_eye_3d, frame.shape)
                
                # Save to output dict
                float(components[0]) * math.pi / 180
                output[HEAD_BONE_X] = float(components[0]) * math.pi / 180
                output[HEAD_BONE_Y] = float(components[1]) * math.pi / 180
                output[HEAD_BONE_Z] = float(components[2]) * math.pi / 180
                output[RIGHT_EYE_BONE_X] = float(right_components[0]) * math.pi / 180
                output[RIGHT_EYE_BONE_Y] = float(right_components[1]) * math.pi / 180
                output[RIGHT_EYE_BONE_Z] = float(right_components[2]) * math.pi / 180
                output[LEFT_EYE_BONE_X] = float(left_components[0]) * math.pi / 180
                output[LEFT_EYE_BONE_Y] = float(left_components[1]) * math.pi / 180
                output[LEFT_EYE_BONE_Z] = float(left_components[2]) * math.pi / 180
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking down"
                elif x > 10:
                    text = "Looking up"
                else: 
                    text = "Forward"

                # Display the nose direction
                # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                if isDraw:
                    if nose_2d:
                        p1 = ((int(nose_2d[0]), int(nose_2d[1])))
                        p2 = ((int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10)))
                        cv2.line(frame, p1, p2, (255,0,0), 3)
                    cv2.putText(frame, text, (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    cv2.putText(frame, f"x:{np.round(x, 2):.2f}", (iw -100, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                    cv2.putText(frame, f"y:{np.round(y, 2):.2f}", (iw -100, 100),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                    cv2.putText(frame, f"z:{np.round(z, 2):.2f}", (iw -100, 150),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
            output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
            output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
        return frame, output
    
    def calculate_bone(self, list_2d:list, list_3d:list, frame_shape):
        ih, iw, ic = frame_shape
        list_2d = np.array(list_2d, dtype=np.float64)
        list_3d = np.array(list_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * iw

        cam_matrix = np.array([[focal_length, 0 , ih/2],
                                [0, focal_length , iw/2],
                                [0, 0, 1]])
        # The distortion parameters
        dist_matrix = np.zeros((4,1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(list_3d, list_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles 
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the roation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        return [x, y, z]



if __name__ == '__main__':

    # Unit test
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    ret, _ = cap.read()
    output = create_default_ifacialmocap_pose()
    while ret:
        # Get frame
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1) # Flip horizontally

        # Detect
        frame, faceLmsList = detector.findFaceMesh(frame=frame,
                                                   output=output,
                                                   isDraw=True,
                                                   isShowText=True)

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
