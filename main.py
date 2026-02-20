import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

y_true = torch.zeros((10, 10))
y_true[:, 2] = 1
y_true[:, 3] = 1
y_true[:, 4] = 1

y_pred = y_true.clone()
y_pred[:, 2] = 0
y_pred[:, 3] = 1
y_pred[:, 4] = 1
y_pred[:, 5] = 1
print(f"{y_pred=}")
print(f"{y_true=}")

overlap = (y_pred * y_true).sum()
total_true = y_true.sum()
total_pred = y_pred.sum()

dice_score = (2 * overlap) / (total_true + total_pred)
# print(f"{dice_score=}")

skel_true = skeletonize(y_true.numpy())
skel_pred = skeletonize(y_pred.numpy())
# print("Predicted Skeleton:", skel_pred.astype(int))

# topology_precision = (skel_pred * y_true).sum()
# print(f"{topology_precision=}")
