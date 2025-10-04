import os
import json
import cv2
import numpy as np

def highlight_anomalies(original_image, segmentation_mask):
    mask = (segmentation_mask > 0).astype(np.uint8)
    blue_overlay = np.zeros_like(original_image)
    blue_overlay[:] = [255, 0, 0] 
    
    yellow_border = np.zeros_like(original_image)
    yellow_border[:] = [0, 255, 255]  
    
    alpha = 0.4
    highlighted_image = original_image.copy()
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)  
    eroded_mask = cv2.erode(mask, kernel, iterations=1)    
    border_mask = dilated_mask - eroded_mask  
    
    if original_image.ndim == 3 and mask.ndim == 2:
        mask_3d = mask[:, :, np.newaxis]
        border_3d = border_mask[:, :, np.newaxis]
    else:
        mask_3d = mask
        border_3d = border_mask
    
    highlighted_image = np.where(
        mask_3d > 0, 
        original_image * (1 - alpha) + blue_overlay * alpha, 
        highlighted_image
    )
    highlighted_image = np.where(
        border_3d > 0,
        yellow_border,
        highlighted_image
    )
    return highlighted_image.astype(np.uint8)

if __name__ == "__main__":
    DATASET_DIR = "dataset/mvtec_loco_anomaly_detection" 
    TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "{class_name}", "test")
    ALL_CLASSES_DIR = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) 
                if os.path.isdir(os.path.join(DATASET_DIR, d))]
    DATASET_MASKED_DIR = "dataset_masked/mvtec_loco_anomaly_detection"
    os.makedirs(DATASET_MASKED_DIR, exist_ok=True)
    
    for class_dir in ALL_CLASSES_DIR:
        print(f"Processing class: {os.path.basename(class_dir)}")
        ground_truth_path = os.path.join(class_dir, "ground_truth")
        test_path = os.path.join(class_dir, "test")
        ALL_ANOMALY_DIR = [os.path.join(ground_truth_path, d) for d in os.listdir(ground_truth_path)]
        for anomaly_dir in ALL_ANOMALY_DIR:
            ALL_IMAGES_DIR = [f for f in os.listdir(anomaly_dir)]
            test_anomaly_dir = os.path.join(test_path, os.path.basename(anomaly_dir))
            save_dir = os.path.join(DATASET_MASKED_DIR, os.path.basename(class_dir), os.path.basename(anomaly_dir))
            os.makedirs(os.path.join(save_dir, 'masked_pictures'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'mask'), exist_ok=True)
            for images_DIR in ALL_IMAGES_DIR:
                mask_list = []
                for mask_path in os.listdir(os.path.join(anomaly_dir, images_DIR)):
                    mask = cv2.imread(os.path.join(anomaly_dir, images_DIR, mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = np.where(mask>0, 1, 0).astype(np.uint8)
                    mask_list.append(mask)
                anomaly_image = cv2.imread(os.path.join(test_anomaly_dir, images_DIR+".png"))
                combined_mask = np.max(mask_list, axis=0)
                highlighted_image = highlight_anomalies(anomaly_image, combined_mask)
                cv2.imwrite(os.path.join(save_dir, 'masked_pictures', images_DIR+".png"), highlighted_image)
                cv2.imwrite(os.path.join(save_dir, 'mask', images_DIR+".png"), combined_mask*255)
        print(f"Finished processing class: {os.path.basename(class_dir)}")
    print("All classes processed!")

    


                
