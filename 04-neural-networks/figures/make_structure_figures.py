"""
Draws neurons-as-questions.png for structure-of-a-neural-network.md.

A square plate with a hole in the middle. Five layer-1 neurons each draw one line
around the hole and answer "how far am I past my line, away from the hole?" (a ReLU
hinge: 0 on the hole side, growing on the other). Adding the five answers gives a
bowl: 0 inside the pentagon the lines make, growing outside it. A layer-2 neuron then
asks a follow-up question about those answers, "is the total still small?", which
is a "close to the hole" detector that no single line could draw.

Run:  python make_structure_figures.py   (writes the PNG next to this script)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def relu(a):
    return np.maximum(a, 0)


n = 400
x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
centre, hole_r, line_r = np.array([0.5, 0.5]), 0.08, 0.22
outside_hole = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 > hole_r**2

# Five neurons. Neuron k has weights w = normal_k and bias b = -(line_r + normal_k . centre),
# so its pre-activation w . (x, y) + b is the signed distance past its line, away from the hole.
angles = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
hinges = []
for nk in normals:
    pre_activation = (x - centre[0]) * nk[0] + (y - centre[1]) * nk[1] - line_r
    hinges.append(relu(pre_activation))
hinges = np.array(hinges)

layer1_sum = hinges.sum(axis=0) / line_r  # weights of 1/line_r keep the numbers readable
layer2 = relu(1 - layer1_sum)


def draw_plate(ax, values, title, cmap="magma"):
    shown = np.where(outside_hole, values, np.nan)
    ax.imshow(shown, origin="lower", extent=(0, 1, 0, 1), cmap=cmap)
    ax.add_patch(plt.Circle(centre, hole_r, color="white", ec="0.3", lw=1))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))

# Panel 1: the five lines, with an arrow pointing to the side where each answer grows
ax = axes[0]
ax.set_facecolor("0.96")
ax.add_patch(plt.Circle(centre, hole_r, color="white", ec="0.3", lw=1))
t = np.linspace(-0.6, 0.6, 2)
colours = plt.cm.tab10(np.arange(5))
for k, (nk, col) in enumerate(zip(normals, colours)):
    foot = centre + line_r * nk
    along = np.array([-nk[1], nk[0]])
    ax.plot(foot[0] + t * along[0], foot[1] + t * along[1], color=col, lw=2)
    ax.annotate("", xy=foot + 0.07 * nk, xytext=foot, arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.text(*(foot - 0.05 * nk), f"q{k + 1}", color=col, ha="center", va="center", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Layer 1: 5 neurons = 5 lines\n(arrow = the side where the answer grows)", fontsize=10)

draw_plate(axes[1], hinges[0] / line_r, "One neuron's answer for q1, ReLU(w·x + b):\n0 on one side, 'how far past' on the other")
draw_plate(axes[2], layer1_sum, "Add the 5 answers:\n0 inside the pentagon, growing outside")
draw_plate(axes[3], layer2, "Layer 2 follow-up, ReLU(1 − sum):\n'inside all 5 lines?' = close to the hole", cmap="viridis")

fig.tight_layout()
out = Path(__file__).with_name("neurons-as-questions.png")
fig.savefig(out, dpi=130)
print("wrote", out)
