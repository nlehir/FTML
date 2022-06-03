import numpy as np
import matplotlib.pyplot as plt
import math
from TP7_utils import forward_pass

"""
    define target
"""
def target(x):
    return forward_pass(x, wh, theta)[-1]

xmin = -1
xmax = 1
m_target = 5

phi = np.random.uniform(-math.pi, math.pi, (m_target, 1))
# wh = 1 / math.sqrt(m_target) * np.column_stack((np.cos(phi), np.sin(phi)))
wh = np.column_stack((np.cos(phi), np.sin(phi)))
# theta = np.random.uniform(-1 / math.sqrt(m_target), 1 / math.sqrt(m_target), m_target + 1)
theta = np.random.uniform(-2, 2, m_target + 1)

sigma = 0.04
inputs = np.linspace(xmin, xmax, num=80)
targets = [target(x) for x in inputs]
noise = np.random.normal(0, sigma, len(inputs))
outputs = noise + targets
plt.plot(inputs, outputs, "o", label="data", alpha=0.8)
plt.plot(inputs, targets, label="target", color="aqua")
plt.xlabel("input")
plt.ylabel("output")
title = f"target function\n"+r"$\sigma=$"+f"{sigma}"
plt.title(title)
plt.legend(loc="best")
# figname = f"target_function_sigma={sigma}"
# edited_figname = figname.replace(".", "_")
plt.savefig("target_function.pdf")
plt.close()
np.save("data/inputs", inputs)
np.save("data/outputs", outputs)
np.save("data/targets", targets)
