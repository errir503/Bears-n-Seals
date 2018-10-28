import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
RATIO_TEST = .7
MATCH_HEIGHT = 1024
MIN_MATCHES = 4
MIN_INLIERS = 4


def computeTransform(imgRef, img, id, warp_mode=cv2.MOTION_HOMOGRAPHY, matchLowRes=True):
    # Convert images to grayscale
    if (len(img.shape) == 3 and img.shape[2] == 3):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgGray = img

    if (len(imgRef.shape) == 3 and imgRef.shape[2] == 3):
        imgRefGray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    else:
        imgRefGray = imgRef

    # resize if requested
    if (matchLowRes):
        aspect = imgRefGray.shape[1] / imgRefGray.shape[0]
        imgRefGray = cv2.resize(imgRefGray, (int(MATCH_HEIGHT * aspect), MATCH_HEIGHT))

    # Detect SIFT features and compute descriptors.
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, descriptors1 = sift.detectAndCompute(imgGray, None)
    kp2, descriptors2 = sift.detectAndCompute(imgRefGray, None)

    if (len(kp1) < 2):
        print("not enough keypoints")
        return False, np.identity(3), 0

    if (len(kp2) < 2):
        print("not enough keypoints")
        return False, np.identity(3), 0

    # scale feature points back to original size
    if (matchLowRes):
        scale = imgRef.shape[0] / imgRefGray.shape[0]
        for i in range(0, len(kp2)):
            kp2[i].pt = (kp2[i].pt[0] * scale, kp2[i].pt[1] * scale)

    # Pick good features
    if (RATIO_TEST < 1):
        # ratio test
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        goodMatches = []
        for m, n in matches:
            if m.distance < RATIO_TEST * n.distance:
                goodMatches.append(m)

        matches = goodMatches
    else:
        # top percentage matches
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

    print("%d matches" % len(matches))

    if (len(matches) < MIN_MATCHES):
        print("not enough matches")
        return False, np.identity(3), 0

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    print("%d inliers" % sum(mask))

    if sum(mask) < MIN_INLIERS:
        print("not enough inliers")
        return False, np.identity(3), 0

    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=4)

    img3 = cv2.drawMatches(imgGray, kp1, imgRef, kp2, goodMatches, None, **draw_params)
    cv2.imwrite("results_sift" + id + ".jpg", img3)

    # Check if we have a robust set of inliers by computing the area of the convex hull
    import scipy.spatial
    # Good area is 11392
    try:
        print('Inlier area ', scipy.spatial.ConvexHull(points2[np.isclose(mask.ravel(), 1)]).area)
        if scipy.spatial.ConvexHull(points2[np.isclose(mask.ravel(), 1)]).area < 1000:
            print("Inliers seem colinear or too close, skipping")
            return False, np.identity(3), 0
    except:
        print("Inliers seem colinear or too close, skipping")
        return False, np.identity(3), 0

    # if non homography requested, compute from inliers
    if warp_mode != cv2.MOTION_HOMOGRAPHY:
        points1Inliers = []
        points2Inliers = []

        for i in range(0, len(mask)):
            if (int(mask[i]) == 1):
                points1Inliers.append(points1[i, :])
                points2Inliers.append(points2[i, :])

        a = cv2.estimateRigidTransform(np.asarray(points1Inliers), np.asarray(points2Inliers),
                                       (warp_mode == cv2.MOTION_AFFINE))
        if a is None:
            return False, np.identity(3), 0
        h = np.identity(3)

        # turn in 3x3 transform
        h[0, :] = a[0, :]
        h[1, :] = a[1, :]

    return True, h, sum(mask)


# projective transform of a point
def warpPoint(pt, h):
    pt = [pt[0], pt[1], 1]
    ptT = np.dot(h, pt)
    ptT = [ptT[0] / ptT[2], ptT[1] / ptT[2]]
    return ptT
