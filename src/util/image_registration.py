import cv2
import matplotlib.pyplot as plt
import csv
import numpy as np

from src.util import normalizer as norm

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
RATIO_TEST = .85
MATCH_HEIGHT = 512
MIN_MATCHES = 4
MIN_INLIERS = 4

def computeTransform(imgRef, img, hs, warp_mode=cv2.MOTION_HOMOGRAPHY, matchLowRes=False,
                     showImgs = False):
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
    keypoints1, descriptors1 = sift.detectAndCompute(imgGray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(imgRefGray, None)

    if (len(keypoints1) < 2):
        print("not enough keypoints")
        return False, np.identity(3), 0

    if (len(keypoints2) < 2):
        print("not enough keypoints")
        return False, np.identity(3), 0

    # scale feature points back to original size
    if (matchLowRes):
        scale = imgRef.shape[0] / imgRefGray.shape[0]
        for i in range(0, len(keypoints2)):
            keypoints2[i].pt = (keypoints2[i].pt[0] * scale, keypoints2[i].pt[1] * scale)

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
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if showImgs:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=mask.ravel().tolist(),  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(imgGray, keypoints1, imgRefGray, keypoints2, matches, None, **draw_params)
        cv2.imwrite("images/registration/matches-" +str(hs.id) + ".jpg", img3)


        print("%d inliers" % sum(mask))

    if sum(mask) < MIN_INLIERS:
        print("not enough inliers")
        return False, np.identity(3), 0

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


# read and normlize IR image
def imreadIR(fileIR, percent=0.01):
    img = cv2.imread(fileIR, cv2.IMREAD_ANYDEPTH)

    if (not img is None):
        imgNorm = np.floor((img - np.percentile(img, percent)) / (
                    np.percentile(img, 100 - percent) - np.percentile(img, percent)) * 256)

    return imgNorm.astype(np.uint8), img


def registerThermalAndColorImages(file, fileOut, folder, displayResults=False):
    hotspots = []

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # header row
        headers = next(reader, None)

        for row in reader:
            hotspots.append(row)

    for i in range(0, len(hotspots)):

        hotspot = hotspots[i]

        fileIR = folder + hotspot[2]
        fileRGB = folder + hotspot[4]
        x = float(hotspot[5])
        y = float(hotspot[6])

        print('%d:\t%s\n\t%s' % (i, fileIR, fileRGB))

        thumb = [-1, -1, -1, -1]

        # Read the images to be aligned
        img, img16bit = imreadIR(fileIR)

        if (img is None):
            print('\nnot found\n')
            hotspots[i][7:11] = thumb
            continue

        imgRef = cv2.imread(fileRGB)

        if (imgRef is None):
            print('\nnot found\n')
            hotspots[i][7:11] = thumb
            continue

        # omcpute transform
        ret, transform = computeTransform(imgRef, img)

        if (ret):
            pt = [x, y]
            ptWarped = np.round(warpPoint(pt, transform))

            thumb = [int(ptWarped[0] - 256), int(ptWarped[1] - 256), int(ptWarped[0] + 256), int(ptWarped[1] + 256)]

            if (displayResults):
                # warp IR image
                imgWarped = cv2.warpPerspective(img, transform, (imgRef.shape[1], imgRef.shape[0]))
                # img16bitWarped = cv2.warpPerspective(img16bit, transform, (imgRef.shape[1], imgRef.shape[0]))

                # display everything
                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.plot(pt[0], pt[1], color='red', marker='o')
                plt.title("Orig IR")

                plt.subplot(2, 2, 2)
                plt.imshow(imgWarped, cmap='gray')
                plt.plot(ptWarped[0], ptWarped[1], color='red', marker='o')
                plt.title("Aligned IR")

                plt.subplot(2, 2, 3)
                plt.imshow(cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB))
                plt.plot(ptWarped[0], ptWarped[1], color='red', marker='o')
                plt.title("Orig RGB")

                plt.subplot(2, 2, 4)
                thumb = imgRef[thumb[1]:thumb[3], thumb[0]:thumb[2], :]
                plt.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
                plt.title("Thumb RGB")

                plt.show()
        else:

            if (displayResults):
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.title("Orig IR")

                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB))
                plt.title("Orig RGB")

                plt.show()

            print('alignment failed!')

        hotspots[i][7:11] = thumb

    with open(fileOut, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(headers)
        for i in range(0, len(hotspots)):
            writer.writerow(hotspots[i])

def register_images(hsm, showFigures=False, showImgs = False):
    for hs in hsm.hotspots:
        if not hs.rgb.load_image():
            print("Failed to load rgb image for hotspot" + hs.id)
            continue
        if not hs.ir.load_image():
            print("Failed to load ir image for hotspot" + hs.id)
            continue

        if not hs.thermal.load_image():
            print("Failed to load ir image for hotspot" + hs.id)
            continue

        img_ir = hs.ir.image[0]

        img_rgb = hs.rgb.image
        img_rgb = cv2.resize(img_rgb, (0,0), fx=0.2, fy=0.2)

        ## Draw circles on hotspots
        # cv2.circle(img_ir, hs.thermal_loc, 5, (0, 255, 0), 10)
        # cv2.circle(img_rgb, hs.getRGBCenterPt(), 50, (0, 255, 0), 50)

        # cv2.imshow('ir', img_ir)
        # cv2.imshow('rgb', img_rgb)
        # cv2.waitKey(0)

        # compute transform
        ret, transform, _ = computeTransform(img_rgb, img_ir, hs, showImgs = showImgs)
        if (not ret):
            print("failed!!!")
            continue



        # warp IR image
        imgWarped = cv2.warpPerspective(img_ir, transform, (img_rgb.shape[1], img_rgb.shape[0]))

        # sample hotspot
        pt = hs.thermal_loc

        # warp hotspot
        ptWarped = warpPoint(pt, transform)

        # write warped IR image
        cv2.imwrite("images/registration/warpedIr-" + str(hs.id) + ".JPG", imgWarped)

        # must write as .png to save with alpha channel, warning this will be a big file
        b_channel, g_channel, r_channel = cv2.split(img_rgb)
        imgBGRA = cv2.merge((b_channel, g_channel, r_channel, imgWarped))

        cv2.imwrite("images/registration/BGRA-" + str(hs.id) + ".PNG", imgBGRA)


        # display everything
        if showFigures:
            plt.figure()
            plt.imshow(img_ir, cmap='gray')
            plt.plot(pt[0], pt[1], color='red', marker='o')
            plt.title("Orig IR")

            plt.figure()
            plt.imshow(imgWarped, cmap='gray')
            plt.plot(ptWarped[0], ptWarped[1], color='red', marker='o')
            plt.title("Aligned IR")

            plt.figure()
            plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
            plt.plot(ptWarped[0], ptWarped[1], color='red', marker='o')
            plt.title("Orig RGB")

            plt.show()