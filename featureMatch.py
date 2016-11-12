import numpy as np
import cv2

def featureMatch(image1, image2):
    detector = cv2.AKAZE_create()

    keyPoints1, descriptor1 = detector.detectAndCompute(image1, None)
    keyPoints2, descriptor2 = detector.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptor1, descriptor2, k = 2)

    #Ratio test
    good = []
    points1 = []
    points2 = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
            points1.append(keyPoints1[m.queryIdx].pt)
            points2.append(keyPoints2[m.trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    #Plot it out
    image3 = cv2.drawMatches(image1, keyPoints1, image2, keyPoints2, good, None, flags = 2)
    # cv2.imshow("Corresponding Points", image3)
    # cv2.waitKey(500)
    print("Writing matches to correspondingPoints.jpg")
    cv2.imwrite("correspondingPoints.jpg", image3)
    return F, points1, points2
