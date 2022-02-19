import pybullet as p
import time
import pybullet_data

log_file = open("log.txt", "w")

# Connect to the physics server and set gravity
physicsClient = p.connect(p.GUI)
mode = p.VELOCITY_CONTROL
maxForce = 1000
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)

# Load in the floor
planeId = p.loadURDF("plane.urdf")

# Set start positio and of the test robot
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robot_arm = p.loadURDF("./urdfs/xArm.urdf", startPos, startOrientation)

# Print robotic joint parameters
log_file.write("Robot Arm ID: {}\n".format(robot_arm))
for i in range(p.getNumJoints(robot_arm)):
    log_file.write("Joint {}: {}\n".format(i, p.getJointInfo(robot_arm, i)))

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
stop = False
joint0_vel = 0
while not stop:
    # Step forward in the simulation
    p.stepSimulation()
    time.sleep(1./240.)
    # Get the current key press
    keys = p.getKeyboardEvents()
    # Quit
    if ord('q') in keys:
        stop = True
    # Adjust motors
    elif ord('j') in keys:
        joint0_vel += 10
    elif ord('k') in keys:
        joint0_vel -= 10
    elif ord('r') in keys:
        joint0_vel = 0

    p.setJointMotorControl2(robot_arm, 2, controlMode=mode, targetVelocity=joint0_vel, force=maxForce)
p.disconnect()
