import numpy as np
import numpy.linalg as linalg

def scaleMatrix(scale: np.array) -> np.array:
    return np.diag(scale)

def rotMatrix(rotEuler: np.array) -> np.array:
    # Unpack the Euler angles
    theta, phi, psi = np.radians(rotEuler[:3])  # Convert angles from degrees to radians

    # Rotation matrix around the X-axis (Roll)
    rotX = np.array([[1,             0,              0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta),  np.cos(theta), 0],
                     [0,             0,              0, 1]])

    # Rotation matrix around the Y-axis (Pitch)
    rotY = np.array([[ np.cos(phi), 0, np.sin(phi), 0],
                     [           0, 1,           0, 0],
                     [-np.sin(phi), 0, np.cos(phi), 0],
                     [           0, 0,           0, 1]])

    # Rotation matrix around the Z-axis (Yaw)
    rotZ = np.array([[np.cos(psi), -np.sin(psi), 0, 0],
                     [np.sin(psi),  np.cos(psi), 0, 0],
                     [          0,            0, 1, 0],
                     [          0,            0, 0, 1]])

    # Combined rotation matrix: rotX @ rotY @ rotZ
    return np.dot(np.dot(rotX, rotY), rotZ)

def translationMatrix(pos: np.array) -> np.array:
    translationMatrix = np.identity(4)
    translationMatrix[:, 3] = pos
    return translationMatrix

class Camera:
    def __init__(self, pos: np.array = np.array([0.0, 0.0, 0.0, 1.0]), rot: np.array = np.array([0.0, 0.0, 0.0, 1.0]), vFov: float = 70.0, aspectRatio: float = 1700.0 / 900.0, nearPlaneOffset: float  = 0.1, farPlaneOffset: float = 200.0):
        self.transform = Transform(pos, rot)
        self.vFov = vFov
        self.aspectRatio = aspectRatio
        self.nearPlaneOffset = nearPlaneOffset
        self.farPlaneOffset = farPlaneOffset
    
    def viewMatrix(self) -> np.array:
        modelNoScaleMat = translationMatrix(self.transform.pos) @ rotMatrix(self.transform.rot)
        return linalg.inv(modelNoScaleMat)
    
    def projectionMatrix(self) -> np.array:
        X = linalg.inv(np.array([[1.0,  0.0,  0.0, 0.0],
                                 [0.0, -1.0,  0.0, 0.0],
                                 [0.0,  0.0, -1.0, 0.0],
                                 [0.0,  0.0,  0.0, 1.0]]))

        focalLength = 1.0 / (np.tan(np.radians(self.vFov * 0.5)))
        x = focalLength / self.aspectRatio
        y = focalLength
        A =  self.farPlaneOffset / (self.farPlaneOffset - self.nearPlaneOffset)
        B = -self.nearPlaneOffset * A
        perspective = np.array([[  x, 0.0, 0.0, 0.0],
                         [0.0,   y, 0.0, 0.0],
                         [0.0, 0.0,   A,   B],
                         [0.0, 0.0, 1.0, 0.0]])

        return perspective @ X

class Transform:
    def __init__(self, pos: np.array = np.array([0.0, 0.0, 0.0, 1.0]), rot: np.array = np.array([0.0, 0.0, 0.0, 1.0]), scale: np.array = np.array([1.0, 1.0, 1.0, 1.0])):
        self.pos = pos
        self.rot = rot
        self.scale = scale

    def modelMatrix(self) -> np.array:
        return translationMatrix(self.pos) @ rotMatrix(self.rot) @ scaleMatrix(self.scale)

    def renderingMatrix(self, cam: Camera) -> np.array:
        return cam.projectionMatrix() @ cam.viewMatrix() @ self.modelMatrix()

    def transformPoint(self, point: np.array, cam: Camera) -> np.array:
        result4 = self.renderingMatrix(cam) @ point
        if (result4[3] == 0.0):
            return np.array([float('nan'), float('nan'), float('nan'), 1.0])
        return result4 / result4[3]

LEFT     = np.array([-1.0, 0.0, 0.0, 1.0])
RIGHT    = np.array([1.0, 0.0, 0.0, 1.0])
UP       = np.array([0.0, 1.0, 0.0, 1.0])
DOWN     = np.array([0.0, -1.0, 0.0, 1.0])
FORWARD  = np.array([0.0, 0.0, -1.0, 1.0])
BACKWARD = np.array([0.0, 0.0,  1.0, 1.0])

# Camera looking down the -z axis
# Also offset along the +z axis
cam1 = Camera(pos = BACKWARD * np.array([10.0, 10.0, 10.0, 1.0]))

print(f"projectionMat:\n{cam1.projectionMatrix()}\n")
print(f"viewMat:\n{cam1.viewMatrix()}\n")

modelTransform = Transform()

v1 = np.array([0.0, 0.0, 0.0, 1.0])

triV1 = np.array([-0.5, -0.5, 0.0, 1.0])
triV2 = np.array([0.5, -0.5, 0.0, 1.0])
triV3 = np.array([0.0, 0.5, 0.0, 1.0])

print(f"modelMat: {modelTransform.modelMatrix()}\n")

# Notice how the y value gets its sign flipped on the transformation from world to NDC space
print(f"{triV1} -> {modelTransform.transformPoint(triV1, cam1)}\n")

modelTransform.pos[2] = 10.0

print(f"modelMat: {modelTransform.modelMatrix()}\n")

# Notice how the z value is nan, bc its before the near plane, bc its not in the viewing frustrum
print(f"{v1} -> {modelTransform.transformPoint(v1, cam1)}\n")

# New camera looking down the +z axis instead
# Also offset along the -z axis instead
cam2 = Camera(pos = FORWARD * np.array([10.0, 10.0, 10.0, 1.0]), rot = UP * np.array([180.0, 180.0, 180.0, 1.0]))
print(f"projectionMat:\n{cam2.projectionMatrix()}\n")
print(f"viewMat:\n{cam2.viewMatrix()}\n")

# Notice how the z value is between [0,1] bc its infront of the camera
# Its also quite large, its bc we are essentially doing 1/z, where z is in [0,1] for valid depths values
# which is anything in view of the camera and also within the near and far plane offsets
print(f"{v1} -> {modelTransform.transformPoint(v1, cam2)}")