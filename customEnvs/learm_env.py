import pybullet as p

physicsClient = p.connect(p.GUI)
# Set the physics engine parameters
p.setGravity(0,0,-10)

# Load in urdf model
urdf = p.loadURDF("./urdfs/xARM.urdf")

