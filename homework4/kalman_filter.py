from ex4_utils import kalman_step
import numpy as np 
import sympy as sp
from matplotlib import pyplot as plt
import math

class KalmanFilter: 

    def __init__(self, model_type, dt=1.0, q=1.0, r=1.0):
        self.dt = dt
        self.q = q
        self.r = r # noise intensity 
        self.model_type = model_type
        self.state_dim = None
        self.set_model()

    def set_model(self):

        dt = self.dt
        dt2 = dt ** 2 
        dt3 = dt ** 3 
        dt4 = dt ** 4 
        dt5 = dt ** 5


        if self.model_type == 'RW':

            self.A = np.eye(2)  
            self.C = np.eye(2)  
            self.Q = np.eye(2) * self.q
            self.R = np.eye(2) * self.r
            self.state_dim = 2

        elif self.model_type == 'NCV':

            self.A = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

            self.C = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])

            self.Q = self.q * np.array([ [dt3 / 3, 0,   dt2 / 2, 0],
                                        [0,   dt3 / 3, 0,   dt2 / 2],
                                        [dt2 / 2, 0,   dt,  0],
                                        [0,   dt2 / 2, 0,   dt]])

            self.R = np.eye(2) * self.r
            self.state_dim = 4

        elif self.model_type == 'NCA':
            
            self.A = np.array([
                [1, 0, dt, 0, dt2/2, 0],
                [0, 1, 0, dt, 0, dt2/2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            self.C = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ])

            self.Q = self.q * np.array([
            [dt5/20, 0,     dt4/8,  0,     dt3/6,  0],
            [0,     dt5/20, 0,     dt4/8, 0,     dt3/6],
            [dt4/8, 0,     dt3/3,  0,     dt2/2,  0],
            [0,     dt4/8, 0,     dt3/3,  0,     dt2/2],
            [dt3/6, 0,     dt2/2,  0,     dt,     0],
            [0,     dt3/6, 0,     dt2/2,  0,     dt]
            ])

            self.R = np.eye(2) * self.r
            self.state_dim = 6

        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim)

    def step(self, measurement):

        self.x, self.P, _, _ = kalman_step(self.A, self.C, self.Q, self.R,measurement.reshape(-1, 1),self.x,self.P)
        return self.x.flatten()
    
def spiral_function(N = 40):

    v = np.linspace (5*math.pi , 0 , N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    return x, y


def star_shape(N= 60):
    t = np.linspace(0, 2 * np.pi, N)
    r = 10 + 5 * np.cos(10 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

def looping_curve(N=60, a=3, b=2, delta=np.pi/2):
    t = np.linspace(0, 2 * np.pi, N)
    x = 10 * np.sin(a * t + delta)
    y = 10 * np.sin(b * t)
    return x, y

def plot_results(ax, x, y, model_type, q, r):

    kf = KalmanFilter(model_type=model_type, q=q, r=r)
    sx = np.zeros_like(x, dtype=np.float32)
    sy = np.zeros_like(y, dtype=np.float32)

    kf.x[0] = x[0]
    kf.x[1] = y[0]
    sx[0] = x[0]
    sy[0] = y[0]

    for j in range(1, len(x)):
        meas = np.array([x[j], y[j]])
        est = kf.step(meas)
        sx[j] = est[0]
        sy[j] = est[1]

    ax.plot(x, y, color='red', linestyle='-', marker='o', markersize=5,
             markerfacecolor='white', markeredgewidth=0.6)
    ax.plot(sx, sy, color='blue', linestyle='-', marker='o', markersize=5,
            markerfacecolor='white', markeredgewidth=0.6)
    
    ax.set_title(f"{model_type}: q={q}, r={r}", fontsize=10)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off') 
    plt.savefig("spiral_function.pdf",  bbox_inches='tight')
    ax.grid(True)

def main():

    x, y = spiral_function()
    models = ['RW', 'NCV', 'NCA']
    param_sets = [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]

    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    for i, model in enumerate(models):
        for j, (q, r) in enumerate(param_sets):
            ax = axes[i, j]
            plot_results(ax, x, y, model, q, r)
            if i == 2:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()


if __name__ == "__main__":

    
    main()








  


