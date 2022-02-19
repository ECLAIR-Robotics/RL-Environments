import pybullet as p
import time
import pybullet_data

# Connect to the physics server and set gravity
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)

# Load in the floor
planeId = p.loadURDF("plane.urdf")

# Set start positio and of the test robot
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("./urdfs/xArm.urdf", startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
