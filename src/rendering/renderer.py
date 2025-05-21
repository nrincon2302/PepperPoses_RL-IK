import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from kinematics.forward import forward_kinematics

class ArmRenderer:
    def __init__(self, params: dict):
        """
        params debe contener 'shoulder_offset_y','upper_arm_length','elbow_offset_y','lower_arm_length'
        """
        self.params = params
        self.max_reach = params['upper_arm_length'] + params['lower_arm_length']
        self.fig = None
        self.ax = None

    def render(self, angles, target, step, curr_level):
        ee_pos, frames = forward_kinematics(angles, self.params)
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(6,6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            limit = self.max_reach*1.1
            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_zlim(-limit, limit)
        pts = np.array([f[:3,3] for f in frames])
        self.ax.cla()
        self.ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o')
        self.ax.scatter(*target, color='r', marker='x', s=50)
        self.ax.set_title(f"Step {step}, Level {curr_level}")
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)