import numpy as np
import cv2

def drawEpilines(image1, image2, lines, points1, points2):
    m, n = image1.shape
    for i, p1, p2 in zip(lines, points1, points2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -i[2]/i[1]])
        x1, y1 = map(int, [n, -(i[2]+i[0]*n)/i[1]])
        image1 = cv2.line(image1, (x0, y0), (x1, y1), color, 1)
        image1 = cv2.circle(image2, tuple(p1), 5, color, -1)
        image2 = cv2.circle(image2, tuple(p2), 5, color, -1)
    return image1, image2
