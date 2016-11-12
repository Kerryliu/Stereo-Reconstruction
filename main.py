import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calibration import calibration
from featureMatch import featureMatch
from itertools import permutations

#Obtain K and F using calibration.py and featureMatch.py
K = calibration()
images = glob.glob("./FixedImages/*.jpg")
image1 = cv2.imread(images[0]) #queryimage
image2 = cv2.imread(images[1]) #trainimage
F, points1, points2 = featureMatch(image1, image2)
#Get Essential matrix, and possible transformation matrixes
E = cv2.findEssentialMat(points1, points2)
print("Fundamental Matrix:\n", F)
print("Essential Matrix:\n", E[0])

R1, R2, t = cv2.decomposeEssentialMat(E[0])

print("R1:\n", R1, "\nR2:\n", R2, "\nt1:\n", t, "\nt2\n", -t)
Rt1 = np.append(R1, t, axis=1)
Rt2 = np.append(R1, -t, axis=1)
Rt3 = np.append(R2, t, axis=1)
Rt4 = np.append(R2, -t, axis=1)
print("Rt1:\n", Rt1, "\nRt2:\n", Rt2, "\nRt3:\n", Rt3, "\nRt4\n", Rt4)

#Triangulate
Rt0 = np.append(np.eye(3), [[0], [0], [0]], axis=1)
listofRts = [Rt1, Rt2, Rt3, Rt4]
for Rti in listofRts:
    tPoints = cv2.triangulatePoints(K@Rt0, Rti, points1.transpose(), points2.transpose())
    if np.mean(tPoints[2] > 0):
        break
tPoints = tPoints[:-1]/tPoints[-1]
print("tPoints:\n", tPoints)

#Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = tPoints[0]
ys = tPoints[1]
zs = tPoints[2]
ax.scatter(xs, ys, zs, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
