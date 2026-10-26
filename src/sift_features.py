import cv2
import numpy as np

SIZE = 256
N_WORDS = 200
_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=300)
    return _sift


def prep(img):
    """Any image (BGR or gray) to a gray image with longest side 256 px."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    s = SIZE / max(h, w)
    return cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_CUBIC)


def descriptors(img):
    _, d = _get_sift().detectAndCompute(prep(img), None)
    return d  # None if no keypoints found


def histogram(d, kmeans):
    """SIFT descriptors to a normalized histogram of visual words."""
    n = kmeans.n_clusters
    if d is None or len(d) == 0:
        return np.zeros(n, np.float32)
    words = kmeans.predict(d.astype(np.float32))
    h = np.bincount(words, minlength=n).astype(np.float32)
    return h / h.sum()


def image_to_features(img, kmeans):
    return histogram(descriptors(img), kmeans)