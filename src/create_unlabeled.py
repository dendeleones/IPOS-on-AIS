import numpy as np
data = np.load("gpss_dataset.npz")
X = data['X'][:200]  # возьмём 200 примеров без меток
np.savez("unlabeled.npz", X=X)
print("unlabeled.npz создан")