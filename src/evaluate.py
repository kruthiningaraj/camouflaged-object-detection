import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_image_mask_pairs
from sklearn.metrics import jaccard_score

def evaluate_model(model_path, test_img_dir, test_mask_dir):
    model = tf.keras.models.load_model(model_path)
    X_test, y_test = load_image_mask_pairs(test_img_dir, test_mask_dir)

    preds = model.predict(X_test)
    preds_bin = (preds > 0.5).astype(np.uint8)

    iou_scores = []
    for i in range(len(y_test)):
        iou = jaccard_score(y_test[i].flatten(), preds_bin[i].flatten())
        iou_scores.append(iou)

    print(f"Mean IoU: {np.mean(iou_scores):.4f}")

    # Visualize a few predictions
    for i in range(3):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(X_test[i]); plt.title("Image")
        plt.subplot(1,3,2); plt.imshow(y_test[i].squeeze(), cmap='gray'); plt.title("Ground Truth")
        plt.subplot(1,3,3); plt.imshow(preds_bin[i].squeeze(), cmap='gray'); plt.title("Prediction")
        plt.savefig(f"outputs/sample_{i}.png")
        plt.close()

if __name__ == "__main__":
    evaluate_model("models/unet_camo.h5", "data/CAMO/test/images", "data/CAMO/test/masks")
