import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

g = 100
L = .5
mu = .043

T = 2*np.pi*np.sqrt(L/g)

def pendulum_eq(y, t, L, g):
    theta, omega = y
    dydt = [omega, - (g/L)*np.sin(theta)]
    return dydt

theta0 = 0
omega0 = 3
y0 = [theta0, omega0]

t = np.linspace(0, T, 500)

solution = odeint(pendulum_eq, y0, t, args=(L, g))

theta = solution[:, 0]

x = L*np.sin(theta)
y = -L*np.cos(theta)

plt.style.use('dark_background')  

fig0 = plt.figure(figsize=(8, 10))

fig0.canvas.manager.set_window_title('behavior and trajectory')

plt.subplot(2, 1, 1)
plt.plot(t, theta, label=r"$\theta(t)$")
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, y, label="pendulum", color="b")
plt.legend()

fig = plt.figure(figsize=(5, 5))

fig.canvas.manager.set_window_title('pendulum')

ax = fig.add_subplot(111)

ax.set_xlim(-1.2*L, 1.2*L)
ax.set_ylim(-0.2*L, 2.2*L)
ax.set_aspect("equal", "box")
ax.axis('off')

line, = ax.plot([], [], color="white", lw=2)
point, = ax.plot([], [], 'bo', markersize=17)
origin = ax.plot([0], [L], 'wo', markersize=3)

def init():
    global solution, theta0, omega0, theta, y0
    line.set_data([], [])
    point.set_data([], [])

    y0 = solution[-1]
    theta0 = y0[0]
    omega0 = y0[1]

    solution = odeint(pendulum_eq, y0, t, args = (L, g))
    theta = solution[:, 0]

    return line, point

def update(frame):
    current_theta = theta[frame]

    x = L * np.sin(current_theta)
    y = -L * np.cos(current_theta) + L

    line.set_data([0, x], [L, y])
    point.set_data([x], [y])

    return line, point

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1)

plt.tight_layout()
plt.show()