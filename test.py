from pinocchio import pin

model = pin.buildModelFromUrdf('/home/djh/robot/robot_control/resources/robot/h1/urdf/h1.urdf')
data = model.createData()
print(pin.computeTotalMass(model,data))