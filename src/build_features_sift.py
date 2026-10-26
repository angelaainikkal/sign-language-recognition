import sys, glob, os
import cv2, numpy as np, joblib
from sklearn.cluster import MiniBatchKMeans
from sift_features import descriptors, histogram, N_WORDS

DATASET_DIR = sys.argv[1]          # folder containing images.npy and labels.npy
PER_LETTER = 100
LETTERS = [chr(c) for c in range(65, 91)]
EXTS = ("png", "jpg", "jpeg")

images = np.load(os.path.join(DATASET_DIR, "images.npy"), mmap_mode="r")
labels = np.load(os.path.join(DATASET_DIR, "labels.npy"))


def files_in(folder):
    out = []
    for e in EXTS:
        out += glob.glob(os.path.join(folder, f"*.{e}"))
        out += glob.glob(os.path.join(folder, f"*.{e.upper()}"))
    return sorted(set(out))


def load(p):
    if p.startswith("dataset:"):
        return np.array(images[int(p.split(":")[1])])
    return cv2.imread(p)


items = []  # (path, label, source, frame)

# 1) dataset: 100 evenly spaced frames per letter, keeping the frame index (0-999)
for L in LETTERS:
    idxs = np.where(labels == L)[0]
    if len(idxs) == 0:
        print("WARNING: no images for", L)
        continue
    pick = np.linspace(0, len(idxs) - 1, min(PER_LETTER, len(idxs))).astype(int)
    for i in pick:
        items.append((f"dataset:{idxs[i]}", L, "dataset", int(i)))

# 2) extra sign photos and drawings
for L in LETTERS:
    for f in files_in(os.path.join("data", "extra", "signs", L)):
        items.append((f, L, "extra", -1))

# 3) non-sign images
for f in files_in(os.path.join("data", "extra", "not_a_sign")):
    items.append((f, "not_a_sign", "not_a_sign", -1))

print("images:", len(items))

# pass 1: descriptors
descs, kept = [], []
for n, (p, lab, src, fr) in enumerate(items):
    img = load(p)
    if img is None:
        print("skip unreadable:", p)
        continue
    descs.append(descriptors(img))
    kept.append((p, lab, src, fr))
    if n % 200 == 0:
        print(n, "/", len(items))

# vocabulary from a sample of descriptors
pool = np.vstack([d for d in descs if d is not None]).astype(np.float32)
if len(pool) > 200000:
    pool = pool[np.random.RandomState(0).choice(len(pool), 200000, replace=False)]
kmeans = MiniBatchKMeans(n_clusters=N_WORDS, random_state=0, batch_size=2048, n_init=3)
kmeans.fit(pool)

# pass 2: histograms
X = np.array([histogram(d, kmeans) for d in descs], dtype=np.float32)
y = np.array([k[1] for k in kept])
source = np.array([k[2] for k in kept])
frame = np.array([k[3] for k in kept])
path = np.array([k[0] for k in kept])

os.makedirs("models", exist_ok=True)
joblib.dump(kmeans, "models/sift_kmeans.joblib")
np.savez("data/features_sift.npz", X=X, y=y, source=source, frame=frame, path=path)
print("saved", X.shape, "| empty histograms:", int((X.sum(1) == 0).sum()))