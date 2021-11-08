import numpy as np


cords = cords.T



y = np.insert(y, 0, values=np.zeros(len(y)), axis=1)
y_grad = np.gradient(y, axis=0)

diff = np.diff(cords, axis=0)
grad = np.gradient(cords, axis=0)

grad[:, 0] = cords[:, 0]
grad[:, 1] = np.abs(grad[:, 1])

y_grad[:, 0] = cords[:, 0]
y_grad[:, 1] = np.abs(y_grad[:, 1])

mm = y_grad[y_grad[:,1].argsort()]
mm = mm[mm[:,0].argsort()]
mm = mm[:int(0.01 * len(mm))]
mm = np.diff(mm)
w = mm.mean()
i = 0