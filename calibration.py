import numpy as np
import cv2
import glob

def calibration():
    #Margin of error
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #7x6 checkerboard
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

    objpoints = [] #World
    imgpoints = [] #Image

    images = glob.glob("./CalibrationImages/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Find corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            print(fname, " is OK!")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

            #Draw and show found corners
            cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            # cv2.imshow("Checkerboard",img)
            # cv2.waitKey(1)
        else:
            print(fname, " is bad...")

    #Do calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    #Generate camera matrix
    img = cv2.imread("./CalibrationImages/IMG_20161110_181537.jpg")
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    #Get error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    #Print out camera matrix and error
    print("Camera Matrix:\n", newcameramtx)
    print("Total error: ", mean_error/len(objpoints))

    images = glob.glob("./TakenImages/*.jpg")
    for fname in images:
        print("Fixing up ", fname)
        #Get image
        img = cv2.imread(fname)
        # cv2.imshow(fname, img)
        # cv2.waitKey(100)

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # cv2.imshow(fname, dst)
        # cv2.waitKey(100)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        # cv2.imshow(fname, dst)
        # cv2.waitKey(100)

        #write image
        newfname = fname.replace("TakenImages", "FixedImages")
        cv2.imwrite(newfname,dst)

    cv2.destroyAllWindows()
    return newcameramtx
