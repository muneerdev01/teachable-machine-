import os
import joblib
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_PATH = os.path.join(ROOT_DIR, "model", "model.pkl")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
_feature_extractor = None


def get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        mobilenet = models.mobilenet_v3_small(weights="DEFAULT")
        mobilenet.classifier = torch.nn.Identity()
        mobilenet.eval()
        _feature_extractor = mobilenet
    return _feature_extractor


def train_model():

    mobilenet = get_feature_extractor()

    X = []
    y = []

    class_names = sorted(os.listdir(DATASET_DIR))

    for class_name in class_names:

        class_path = os.path.join(DATASET_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            try:
                img = Image.open(image_path).convert("RGB")
                img = _transform(img).unsqueeze(0)

                with torch.no_grad():
                    features = mobilenet(img)

                X.append(features.squeeze().numpy())
                y.append(class_name)

            except Exception as e:
                print("Skipping:", image_path, e)

    X = np.array(X)

    # 🚨 SAFETY CHECK (IMPORTANT)
    if len(X) == 0:
        return {"error": "No images found in dataset"}

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    return {
        "classes": class_names,
        "samples": len(y)
    }


def load_trained_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_image(model, image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    img_tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        features = get_feature_extractor()(img_tensor)

    features = features.squeeze().numpy().reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    scored = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return scored