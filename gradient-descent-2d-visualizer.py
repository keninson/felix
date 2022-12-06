import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

w = np.array([-2.0], dtype=np.float32)
scale = 0.1 * np.abs(w[0])
xs = np.arange(-np.abs(w[0]) - scale, np.abs(w[0]) + scale, 0.01)
ys = xs**2

# сама функция x^2
def forward():
    return w**2


# производная x^2 = 2*x
def gradient():
    return w * 2


def fn(x):
    return x * x


lr = 0.8
epochs = 10
aweights = np.array([], dtype=np.float32)
aweights = np.append(aweights, w)
grads = np.array([gradient()], dtype=np.float32)
for i in range(epochs):
    y_pred = forward()
    w -= np.dot(gradient(), lr)
    aweights = np.append(aweights, w)
    grads = np.append(grads, gradient())
    # print(w, y_pred)


#                        анимация
# ______________________________________________________ #
# plt.style.use("dark_background")
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.autoscale(False)


def animate(i):
    ax1.clear()
    # ax1.quiver(
    #     aweights[i],
    #     fn(aweights[i]),
    #     -aweights[i],
    #     -fn(aweights[i]),
    #     color=["r"],
    #     scale=2,
    #     scale_units="xy",
    #     angles="xy",
    #     zorder=2,
    # )
    # ax1.plot(
    #     [aweights[i] + scale * (np.sign(aweights[i])), 0],
    #     [
    #         grads[i] * (aweights[i] + scale * np.sign(aweights[i]))
    #         + fn(aweights[i])
    #         - grads[i] * aweights[i],
    #         0 + fn(aweights[i]) - grads[i] * aweights[i],
    #     ],
    #     c="blue",
    #     zorder=1,
    # )
    ax1.axline(
        (aweights[i], fn(aweights[i])), slope=grads[i], zorder=2, linestyle=(0, (5, 5))
    )
    ax1.plot(xs, ys, c="black", zorder=1)
    ax1.scatter(aweights[i], fn(aweights[i]), c="red", zorder=2)


ani = animation.FuncAnimation(fig, animate, interval=1000, frames=epochs + 1)
plt.show()
