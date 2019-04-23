import sys
import os

sys.path.append(os.path.abspath('../'))

del sys, os

import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import base.classify_svhn as classify_svhn
from base.classify_svhn import Classifier
import numpy as np
import scipy as scipy

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    # Compute all the sample features
    concatenated_sample_features = np.concatenate(
        [np.reshape(sample_feature, (-1, 1)) for sample_feature in sample_feature_iterator],
        axis=1
    )

    # Compute the mean and covariance of samples
    mu_q = np.mean(concatenated_sample_features, axis=1)
    sigma_q = np.cov(concatenated_sample_features)

    # Compute all the test features
    concatenated_testset_features = np.concatenate(
        [np.reshape(testset_feature, (-1, 1)) for testset_feature in testset_feature_iterator],
        axis=1
    )

    # Compute the mean and covariance of samples
    mu_p = np.mean(concatenated_testset_features, axis=1)
    sigma_p = np.cov(concatenated_testset_features)

    return round(
        np.linalg.norm(mu_p - mu_q) ** 2 + np.trace(
            sigma_p + sigma_q - 2 * scipy.linalg.sqrtm(np.matmul(sigma_p, sigma_q) + np.eye(512, 512) * 10 ** (-7))
        ), 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
