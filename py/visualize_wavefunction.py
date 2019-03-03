import sys
from collections import defaultdict
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl


def read_data(fpath: str) -> dict:
    out = defaultdict(list)
    with open(fpath) as file:
        for line in file:
            line = line.split()
            header = line[0].replace(":", "")
            value = float(line[1])
            out[header].append(value)
    for key, value in out.items():
        out[key] = np.array(value)
    return out


coordinates = read_data(sys.argv[1])

def interleave(first, second):
    return np.vstack((first, second)).reshape((-1), order='F')

x1 = coordinates["x1"]
y1 = coordinates["y1"]
z1 = coordinates["z1"]

x2 = coordinates["x2"]
y2 = coordinates["y2"]
z2 = coordinates["z2"]

pos = np.c_[
        interleave(x1, x2), 
        interleave(y1, y2), 
        interleave(z1, z2)
]

# visualize with PyQt for performance
color = np.zeros((pos.shape[0], 4), dtype=np.float32)
color[:, 0] = color[:, 1] = color[:, 2] = 1

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()

sp = gl.GLScatterPlotItem(pos=pos)
w.addItem(sp)
particle = 0

def update():
    global color, particle
    color[particle % pos.shape[0], 3] = 0.5
    sp.setData(color=color)
    particle += 1

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(1)
app.processEvents()

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
