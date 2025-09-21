#!/usr/bin/env python3
import argparse, json, os, time, hashlib, math, glob, shutil, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
CLASS_MAP = {"no": 0, "yes": 1}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
#just a commit
def set_seed(seed=42):
    """Sets a global seed for reproducibility."""
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

def list_images(dir_path, classes=("no","yes")):
    """ Lists image files and their corresponding labels from a directory. Assumes a directory structure like: dir_path/class_name/image.jpg """
    files, labels = [], []
    for cls in classes:
        for p in glob.glob(os.path.join(dir_path, cls, "*")):
            if p.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(p); labels.append(CLASS_MAP[cls])
    return files, labels

def decode_and_resize(path):
    """Decodes an image file and resizes it, then applies EfficientNet preprocessing."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img

def augment(img):
    """Applies random data augmentation to an image tensor."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

def make_ds(files, labels, batch, shuffle=True, augment_on=False, sample_weight=1.0):
    """ Creates a TensorFlow Dataset from file paths and labels. Includes options for shuffling, augmentation, and sample weighting. """
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(files)))
    def _map(f, y):
        x = decode_and_resize(f)
        if augment_on:
            x = augment(x)
        w = tf.cast(sample_weight, tf.float32)
        return x, tf.cast(y, tf.float32), w
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(lr=1e-3, pretrained=True):
    """ Builds and compiles an EfficientNetB0 model for binary classification. Allows for optional use of ImageNet pretrained weights. """
    tf.keras.backend.set_image_data_format('channels_last')

    # 1) Force a 3-channel Input tensor
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="rgb_input")

    # 2) Build EfficientNet on that input
    base = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights=None, input_tensor=inp, pooling="avg"
    )

    # 3) If using pretrained, load the official notop weights onto this exact graph
    if pretrained:
        weights_path = tf.keras.utils.get_file(
            "efficientnetb0_notop.h5",
            "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
        )
        base.load_weights(weights_path, by_name=True, skip_mismatch=False)

    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt, loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )
    # Sanity assert: kernel should be (3,3,3,32)
    kshape = model.get_layer("stem_conv").weights[0].shape
    print("[DEBUG] stem_conv kernel shape:", kshape)  # expect (3, 3, 3, 32)
    return model

# ---

def best_threshold(y_true, y_prob):
    """Finds the optimal classification threshold for F1 score."""
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def entropy_binary(p):
    """Calculates binary cross-entropy for a probability."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(- (p*np.log2(p) + (1-p)*np.log2(1-p)))

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculates the Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; N = len(y_true)
    for i in range(n_bins):
        msk = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if not np.any(msk):
            continue
        conf = y_prob[msk].mean()
        acc = (y_true[msk] == (y_prob[msk] >= 0.5).astype(int)).mean()
        ece += (msk.sum()/N) * abs(acc - conf)
    return float(ece)

def write_txt(path, text):
    """Writes text to a file."""
    with open(path, "w") as f:
        f.write(str(text).strip()+"\n")

def main(argv=None):
    """ Main function for training the ML model. Handles data loading, model building, training, and evaluation. """
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--weak_dir", default="data/weak_feedback")
    p.add_argument("--out_dir", default="ml/artifacts/run")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weak_weight", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrained weights for EfficientNetB0")
    
    args = p.parse_args(argv)

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "saved_model").mkdir(parents=True, exist_ok=True)

    # Datasets
    tr_files, tr_labels = list_images(os.path.join(args.data_dir, "train"))
    val_files, val_labels = list_images(os.path.join(args.data_dir, "val"))
    te_files, te_labels = list_images(os.path.join(args.data_dir, "test"))
    weak_files, weak_labels = [], []

    if os.path.isdir(args.weak_dir):
        for cls in ("no", "yes"):
            files = glob.glob(os.path.join(args.weak_dir, cls, "*"))
            weak_files += [f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))]
            weak_labels += [CLASS_MAP[cls]]*len([f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))])

    ds_tr_main = make_ds(tr_files, tr_labels, args.batch_size, shuffle=True, augment_on=True, sample_weight=1.0)
    if weak_files:
        ds_tr_weak = make_ds(weak_files, weak_labels, args.batch_size, shuffle=True, augment_on=True, sample_weight=args.weak_weight)
        ds_train = tf.data.Dataset.sample_from_datasets([ds_tr_main, ds_tr_weak], weights=[0.8, 0.2])
    else:
        ds_train = ds_tr_main

    ds_val = make_ds(val_files, val_labels, args.batch_size, shuffle=False, augment_on=False, sample_weight=1.0)
    ds_test = make_ds(te_files, te_labels, args.batch_size, shuffle=False, augment_on=False, sample_weight=1.0)

    # Build & train
    model = build_model(lr=args.lr, pretrained=(not args.no_pretrained))
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(out_dir / "ckpt.keras"), monitor="val_auc", mode="max", save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True)
    history = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=[ckpt, early], verbose=2)

    # Save SavedModel
    model.save("artifacts/current/model.keras")           # single-file Keras format
    model.export(out_dir / "saved_model")         # TF SavedModel directory (for Serving/TFLite/etc.)

    # Evaluate on val/test
    def collect_probs(ds):
        y_true, y_prob = [], []
        for x, y, _w in ds:
            p = model.predict(x, verbose=0).ravel()
            y_prob.extend(p.tolist()); y_true.extend(y.numpy().astype(int).tolist())
        return np.array(y_true), np.array(y_prob)

    yv, pv = collect_probs(ds_val)
    yt, pt = collect_probs(ds_test)

    print("val size:", len(yv), "positives:", yv.sum(), "negatives:", (1-yv).sum())
    print("p>=0.5 ratio:", (pv>=0.5).mean(), "p mean:", pv.mean(), "p min/max:", pv.min(), pv.max())

    val_auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else 0.5
    thr, val_f1 = best_threshold(yv, pv)
    val_prec = float(precision_score(yv, (pv>=thr).astype(int), zero_division=0))
    val_rec  = float(recall_score(yv, (pv>=thr).astype(int), zero_division=0))
    val_acc  = float(((pv>=thr).astype(int) == yv).mean())
    ece = expected_calibration_error(yv, pv)
    tn, fp, fn, tp = confusion_matrix(yv, (pv>=thr).astype(int)).ravel()

    test_auc = float(roc_auc_score(yt, pt)) if len(np.unique(yt)) > 1 else 0.5
    test_f1 = float(f1_score(yt, (pt>=thr).astype(int), zero_division=0))

    metrics = {
        "val": {"auc": val_auc, "f1": float(val_f1), "acc": val_acc, "precision": val_prec, "recall": val_rec},
        "test": {"auc": test_auc, "f1": test_f1},
        "calibration": {"ece": ece},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": thr
    }
    with open(out_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(out_dir / "label_map.json", "w") as f: json.dump(INV_CLASS_MAP, f, indent=2)
    write_txt(out_dir / "threshold.txt", thr)

    # Version string: git sha if present, else timestamp
    gitsha = os.getenv("GIT_SHA", "")
    ver = f"{gitsha[:7]}-{int(time.time())}" if gitsha else f"local-{int(time.time())}"
    write_txt(out_dir / "model_version.txt", ver)

    print(f"[OK] Saved model to: {out_dir}/saved_model")
    print(f"[OK] Val AUC={val_auc:.3f} F1={val_f1:.3f}  Test AUC={test_auc:.3f} F1={test_f1:.3f}")
    print(f"[OK] Threshold={thr:.3f} ECE={ece:.3f}")

if __name__ == "__main__":
    main()