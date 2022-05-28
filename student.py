import numpy as np
import cv2
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, gaussian
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import time


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    alpha = 0.06
    threshold = 0.00008  # need to normalize R value?
    step = 4   # 2
    sigma = 0.5
    rows = image.shape[0]
    cols = image.shape[1]
    xs = []
    ys = []

    # Compute x and y derivatives of the image
    Ix = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
    Iy = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)

    # smooth using gaussian filter
    Ix = gaussian(Ix, sigma)
    Iy = gaussian(Iy, sigma)

    # compute products of derivatives at every pixel
    Ixx = Ix * Ix
    Ixy = Iy * Ix
    Iyy = Iy * Iy

    hfw = int(feature_width / 2)  # half_feature_width

    # Compute the sums of the products of derivatives at each pixel
    for y in range(hfw, rows - hfw, step):
        for x in range(hfw, cols - hfw, step):
            Sxx = np.sum(Ixx[y - hfw:y + 1 + hfw, x - hfw:x + 1 + hfw])
            Syy = np.sum(Iyy[y - hfw:y + 1 + hfw, x - hfw:x + 1 + hfw])
            Sxy = np.sum(Ixy[y - hfw:y + 1 + hfw, x - hfw:x + 1 + hfw])
            # Compute the response of the detector at each pixel
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            R = det - alpha * (trace ** 2)
            # Threshold on value R
            if R > threshold:
                xs.append(x-1)
                ys.append(y-1)

    # xs = np.zeros(1)
    # ys = np.zeros(1)

    return np.asarray(xs), np.asarray(ys)


def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # This is a placeholder - replace this with your features!
    # Convert to integers for indexing
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    # Define window function to get 16 x 16 batches
    def window_16(y, x):
        # y indicates rows number
        # x indicates columns number
        hfw = feature_width / 2    # hfw : half feature width
        rows = (x - (hfw - 1), x + hfw)
        if rows[0] < 0:
            rows = (0, rows[1] - rows[0])
        if rows[1] >= image.shape[0]:
            rows = (rows[0] + (image.shape[0] - 1 - rows[1]), image.shape[0] - 1)
        cols = (y - (hfw - 1), y + hfw)
        if cols[0] < 0:
            cols = (0, cols[1] - cols[0])
        if cols[1] >= image.shape[1]:
            cols = (cols[0] - (cols[1] + 1 - image.shape[1]), image.shape[1] - 1)
        return int(rows[0]), int(rows[1]) + 1, int(cols[0]), int(cols[1]) + 1



    def apply_SIFT(row, col, magnitude, angle):
        features = np.zeros((len(x), l * l * 8))  # 2d array (interest points, no. features =128)
        n = 0
        for t in zip(row, col):
            rs, re, cs, ce = window_16(t[0], t[1])
            mag_window16 = magnitude[rs:re, cs:ce]
            ang_window16 = angle[rs:re, cs:ce]
            c = 0
            for i in range(l):
                for j in range(l):
                    mag_window4 = mag_window16[i * l:(i + 1) * l, j * l:(j + 1) * l]
                    ang_window4 = ang_window16[i * l:(i + 1) * l, j * l:(j + 1) * l]
                    var = np.histogram(ang_window4, bins=8,
                                       range=(0, 360), weights=mag_window4)[0]
                    features[n, c * 8:(c + 1) * 8] = var
                    c += 1

            features_norm = np.linalg.norm(features[n, :])

            if features_norm != 0:
                features[n, :] = features[n, :] / features_norm
            n += 1
        return features

    # Calculating the gradient magnitude and direction
    sigma = 0.8
    img = cv2.GaussianBlur(image, (3, 3), sigma)
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    l = int(feature_width / 4)
    normalized_batch = False




    if normalized_batch:
        n = 0
        features = np.zeros((len(x), 256))  # 2d array (interest points, no. features =128)
        for t in zip(x, y):

            rs, re, cs, ce = window_16(t[0], t[1])
            image_window16 = image[rs:re, cs:ce]
            image_window16 = image_window16.flatten()
            # Normalize the 16 x 16 batches
            features_norm = np.linalg.norm(image_window16)

            if features_norm != 0:
                features[n, :] = image_window16 / features_norm
            n += 1
    # SIFT :
    else:
        features = apply_SIFT(x, y, mag, ang)


    # raise each element of the final feature vector
    # to some power that is less than one.
    features = features ** 0.8

    return features


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    def apply_kd_tree(im1_features, im2_features):
        start_time = time.process_time()
        kdt = KDTree(im2_features, leaf_size=30, metric='euclidean')
        for i in range(im1_features.shape[0]):
            dist, sorted_index = kdt.query(im1_features[i, :].reshape(1, -1), k=2)  # get the nearst 2
            d1 = dist[0][0]
            d2 = dist[0][1]
            # Apply threshold to get best matches to increase overall accuracy
            threshold = 0.9
            assert d2 != 0, "d2 can't be zero"
            ratio = d1 / d2

            if ratio < threshold:
                matches.append([i, sorted_index[0][0]])
                # confidence used in evaluation is (- confidence)
                confidences.append(1 - ratio)
        end_time = time.process_time()
        full_time = end_time - start_time
        return confidences, full_time
    matches = []
    confidences = []

    print(im1_features.shape)
    pca_img = PCA(n_components=32)
    pca_img.fit(im1_features)
    im1_features = pca_img.transform(im1_features)
    im2_features = pca_img.transform(im2_features)

    print(im1_features.shape)
    KD_Tree = True

    # Condition to check if features < 2 to apply NNDR test
    if im2_features.shape[0] > 1:
        if KD_Tree:
            confidences, full_time = apply_kd_tree(im1_features, im2_features)

        else:
            start_time = time.process_time()
            for i in range(im1_features.shape[0]):
                # calculating the euclidean distance between one interest point of img1
                # to all interests points of img2
                distance = ((im1_features[i, :] - im2_features) ** 2)
                distance = distance.sum(axis=1)
                distance = np.sqrt(distance)

                sorted_index = np.argsort(distance)
                d1 = distance[sorted_index[0]]
                d2 = distance[sorted_index[1]]

                # Apply threshold to get best matches to increase overall accuracy
                threshold = 0.9
                assert d2 != 0, "d2 can't be zero"
                ratio = d1 / d2

                if ratio < threshold:
                    matches.append([i, sorted_index[0]])
                    confidences.append(1 - ratio)
            end_time = time.process_time()
            full_time = end_time - start_time

        if KD_Tree:
            print('time taken by KD Tree: ', full_time)
        else:
            print('time taken by tradition Method: ', full_time)

        confidences = np.asarray(confidences)
        matches = np.asarray(matches)

    # Condition to check if matches equal zero or ratio test
    # wasn't applied when image2 features elements < 2
    if len(matches) == 0:
        # better to raise exception (assert, note: msh hatshtghl)
            # These are placeholders - replace with your matches and confidences!
            matches = np.zeros((1, 2))
            confidences = np.zeros(1)

    return matches, confidences
