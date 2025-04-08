import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import io
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import jaccard_score  # IoU
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

class RescuenetDataset(Sequence):
    def __init__(self, image_ids, image_dir, mask_dir, batch_size=8, img_size=(256, 256)):
        self.image_ids = image_ids  # e.g. ['11078', '11079']
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for img_id in batch_ids:
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{img_id}_lab.png")

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to load image: {img_path}")
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Failed to load mask: {mask_path}")
                continue

            img = cv2.resize(img, self.img_size)
            mask = cv2.resize(mask, self.img_size)

            img = img / 255.0
            mask = np.expand_dims(mask / 255.0, axis=-1)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

# # Mock model with .predict (replace with your actual model)
# class DummyModel:
#     def predict(self, image_batch):
#         # Mock prediction: label "Cat" or "Dog" randomly
#         return ["Cat" if np.random.rand() > 0.5 else "Dog" for _ in image_batch]

# model = DummyModel()

# Image folder
IMAGE_FOLDER = "images/test-org-img"  # Change to your image folder path

def load_images(folder):
    images = []
    filenames = []
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            images.append(img_path)
            filenames.append(file)
    return images, filenames

def show_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize as needed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def preprocess_image(selected_files):
    test_ids = [i.split('.')[0] for i in selected_files]
    print(test_ids)
    test_gen = RescuenetDataset(test_ids, 'images/test-org-img', 'images/test-label-img')
    return test_gen

st.title("Disaster-Resilient Military Base Damage Assessment with Autonomous Object Tracking")

# Model selection
model_choice = st.radio(
    "Choose the model to use:",
    ('UNet', 'YOLO'),
    index=0,
    horizontal=True
)

st.markdown(f"### 📌 Selected Model: `{model_choice}`")

if model_choice == 'UNet':
    model = load_model(r'unet_rescuenet.h5')
elif model_choice == 'YOLO':
    model = load_model(r'unet_rescuenet.h5')

images, filenames = load_images(IMAGE_FOLDER)

selected_files = st.multiselect("Select images to run prediction on:", filenames)

if selected_files:
    st.subheader("Selected Images")

    images_to_predict = []
    cols = st.columns(len(selected_files))

    for idx, file in enumerate(selected_files):
        path = os.path.join(IMAGE_FOLDER, file)
        img = show_image(path)
        images_to_predict.append(img)
        cols[idx].image(path, caption=file, use_container_width=True)

    if st.button("Run Prediction"):

        start_time = time.time()
        predictions = model.predict(np.array(images_to_predict))
        elapsed_time = time.time() - start_time

        st.subheader("🔍 Prediction Results")

        # To accumulate metrics across samples
        all_gt = []
        all_pred = []
        # Class labels (0 to 11)
        NUM_CLASSES = 12
        CLASS_NAMES = ['Background', 'Debris', 'Water', 'Building_No_Damage', 'Building_Minor_Damage',
                    'Building_Major_Damage', 'Building_Total_Destruction', 'Vehicle', 'Road', 'Tree', 'Pool', 'Sand']
        for fname, pred in zip(selected_files, predictions):
            testname = f"{fname.split('.')[0]}_lab.png"
            mask_path = os.path.join("images/test-label-img", testname)
            img_path = os.path.join(IMAGE_FOLDER, fname)

            # Load original image
            orig_image = Image.open(img_path)

            # Load ground truth mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # Process predicted mask
            if isinstance(pred, np.ndarray):
                pred = pred.squeeze()
                if pred.max() <= 1.0:
                    pred = (pred * 255).astype(np.uint8)


            class_colors = [
                (0, 0, 0),             # 0 - background
                (7,3,252),             #debris
                (61, 230, 250),        # 1 - water
                (180, 120, 120),       # 2 - building-no-damage
                (235, 255, 7),         # 3 - building-minor-damage
                (255, 184, 6),         # 4 - building-major-damage
                (255, 0, 0),           # 5 - building-total-destruction
                (255, 0, 245),         # 6 - vehicle
                (140, 140, 140),       # 7 - road
                (4, 250, 7),           # 8 - tree
                (255, 235, 0),          # 9 - pool
                (160, 150, 20)        # 10 - sand
            ]

            # Normalize RGB values from 0–255 to 0–1 for matplotlib
            normalized_colors = np.array(class_colors) / 255.0
            cmap = ListedColormap(normalized_colors)
            norm = BoundaryNorm(np.arange(len(class_colors) + 1), cmap.N)

            # Now apply to your masks:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(orig_image)
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(mask, cmap=cmap, norm=norm)
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis('off')

            axs[2].imshow(pred, cmap=cmap, norm=norm)
            axs[2].set_title("Predicted Mask")
            axs[2].axis('off')

            plt.tight_layout()
        
            # # Plot using matplotlib with colormaps
            # fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            # axs[0].imshow(orig_image)
            # axs[0].set_title("Original Image")
            # axs[0].axis('off')

            # axs[1].imshow(mask, cmap='nipy_spectral')
            # axs[1].set_title("Ground Truth Mask")
            # axs[1].axis('off')

            # axs[2].imshow(pred, cmap='nipy_spectral')
            # axs[2].set_title("Predicted Mask")
            # axs[2].axis('off')

            # plt.tight_layout()

            # Convert matplotlib figure to image buffer for Streamlit
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf, caption=f"Results for {fname}", use_container_width=True)
            plt.close()
            
            #gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Resize prediction to GT size if needed
            if pred.shape != mask.shape:
                pred = cv2.resize(pred.squeeze(), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            all_gt.append(mask.flatten())
            all_pred.append(pred.flatten())

        # Flatten all masks
        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        # Calculate per-class and macro-averaged scores
        iou_scores = jaccard_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
        precision_scores = precision_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
        recall_scores = recall_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
        f1_scores = f1_score(all_gt, all_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)

        # Mean scores
        mean_iou = np.mean(iou_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)

        st.subheader("📈 Segmentation Metrics Per Class")

        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "IoU": iou_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1-Score": f1_scores
        }).round(4)

        st.dataframe(df)

        st.markdown("### 🔍 Macro-Averaged Scores")
        st.write(f"**Mean IoU:** {mean_iou:.4f}")
        st.write(f"**Mean Precision:** {mean_precision:.4f}")
        st.write(f"**Mean Recall:** {mean_recall:.4f}")
        st.write(f"**Mean F1-Score:** {mean_f1:.4f}")


        st.markdown("---")
        st.metric(label="🕒 Prediction Time", value=f"{elapsed_time:.2f} seconds")
        st.metric(label="📊 Number of Images", value=len(selected_files))

else:
    st.info("Please select at least one image to run prediction.")

