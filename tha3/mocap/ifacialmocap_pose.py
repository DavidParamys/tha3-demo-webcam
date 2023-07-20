from tha3.mocap.ifacialmocap_constants import BLENDSHAPE_NAMES, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_X, \
    RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, RIGHT_EYE_BONE_QUAT


def create_default_ifacialmocap_pose():
    data = {}

    for blendshape_name in BLENDSHAPE_NAMES:
        data[blendshape_name] = 0.0

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

def print_ifacialmocap_pose(data):
    print("print_ifacialmocap_pose:")
    for blendshape_name in BLENDSHAPE_NAMES:
        print(f"data[{blendshape_name}]:{data.get(blendshape_name):.2f} ", end ="")
    print("")

    print(f"data[{HEAD_BONE_X}]:{data.get(HEAD_BONE_X):.2f} ", end="")
    print(f"data[{HEAD_BONE_Y}]:{data.get(HEAD_BONE_Y):.2f} ", end="")
    print(f"data[{HEAD_BONE_Z}]:{data.get(HEAD_BONE_Z):.2f}", end="")
    print(f"data[{HEAD_BONE_QUAT}]:{data.get(HEAD_BONE_QUAT)} ")

    print(f"data[{LEFT_EYE_BONE_X}]:{data.get(LEFT_EYE_BONE_X):.2f} ", end="")
    print(f"data[{LEFT_EYE_BONE_Y}]:{data.get(LEFT_EYE_BONE_Y):.2f} ", end="")
    print(f"data[{LEFT_EYE_BONE_Z}]:{data.get(LEFT_EYE_BONE_Z):.2f} ", end="")
    print(f"data[{LEFT_EYE_BONE_QUAT}]:{data.get(LEFT_EYE_BONE_QUAT)}")

    print(f"data[{RIGHT_EYE_BONE_X}]:{data.get(RIGHT_EYE_BONE_X):.2f} ", end="")
    print(f"data[{RIGHT_EYE_BONE_Y}]:{data.get(RIGHT_EYE_BONE_Y):.2f} ", end="")
    print(f"data[{RIGHT_EYE_BONE_Z}]:{data.get(RIGHT_EYE_BONE_Z):.2f} ", end="")
    print(f"data[{RIGHT_EYE_BONE_QUAT}]:{data.get(RIGHT_EYE_BONE_QUAT)}")