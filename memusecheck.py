import memory_profiler
from skimage import io, img_as_float32
from skimage.color import rgb2gray
from skimage.transform import rescale

import student
from helpers import evaluate_correspondence
import argparse
import numpy as np

def load_data(file_name):
    """
     1) Load stuff
     There are numerous other image sets in the supplementary data on the
     project web page. You can simply download images off the Internet, as
     well. However, the evaluation function at the bottom of this script will
     only work for three particular image pairs (unless you add ground truth
     annotations for other image pairs). It is suggested that you only work
     with the two Notre Dame images until you are satisfied with your
     implementation and ready to test on additional images. A single scale
     pipeline works fine for these two images (and will give you full credit
     for this project), but you will need local features at multiple scales to
     handle harder cases.

     If you want to add new images to test, create a new elif of the same format as those
     for notre_dame, mt_rushmore, etc. You do not need to set the eval_file variable unless
     you hand create a ground truth annotations. To run with your new images use
     python main.py -p <your file name>.

    :param file_name: string for which image pair to compute correspondence for

        The first three strings can be used as shortcuts to the
        data files we give you

        1. notre_dame
        2. mt_rushmore
        3. e_gaudi

    :return: a tuple of the format (image1, image2, eval file)
    """

    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"

    eval_file = "../data/NotreDame/NotreDameEval.mat"

    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2, eval_file

def memfunc():


    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pair", required=True,
                        help="Either notre_dame, mt_rushmore, or e_gaudi. Specifies which image pair to match")

    args = parser.parse_args()

    # (1) Load in the data
    image1_color, image2_color, eval_file = load_data(args.pair)

    image1 = rgb2gray(image1_color)
    image2 = rgb2gray(image2_color)
    scale_factor = 0.5
    feature_width = 16

    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    '''   # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"
    eval_file = "../data/NotreDame/NotreDameEval.mat"

    scale_factor = 0.5
    feature_width = 16

    image1 = img_as_float32(rescale(rgb2gray(io.imread(image1_file)), scale_factor))
    image2 = img_as_float32(rescale(rgb2gray(io.imread(image2_file)), scale_factor))'''

    (x1, y1) = student.get_interest_points(image1, feature_width)
    (x2, y2) = student.get_interest_points(image2, feature_width)

    image1_features = student.get_features(image1, x1, y1, feature_width)
    image2_features = student.get_features(image2, x2, y2, feature_width)

    matches, confidences = student.match_features(image1_features, image2_features)

    # evaluate_correspondence(image1, image2, eval_file, scale_factor,
    #                        x1, y1, x2, y2, matches, confidences, 0)

    ##############################(ADDED BY STUDENT ME)#########################################

    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pair", required=True,
                        help="Either notre_dame, mt_rushmore, or e_gaudi. Specifies which image pair to match")

    memuse = max(memory_profiler.memory_usage())
    print("Your program memory use: " + str(memuse) + " MiB")
    args = parser.parse_args()
    print("Matches: " + str(matches.shape[0]))
    num_pts_to_visualize = 50
    evaluate_correspondence(image1_color, image2_color, eval_file, scale_factor,
                            x1, y1, x2, y2, matches, confidences, num_pts_to_visualize, args.pair + '_matches.jpg')
    #############################################################################################


if __name__ == "__main__":
    memfunc()
    #memuse = max(memory_profiler.memory_usage(proc=memfunc))
    #print("Your program memory use: " + str(memuse) + " MiB")
