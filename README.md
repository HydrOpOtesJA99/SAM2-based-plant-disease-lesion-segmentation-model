# Hugging Face Resources

## Fine-tuned Model
- [SAM2 Fine-tuned Model](https://huggingface.co/godoldol99/SAM2-based-plant-disease-lesion-segmentation-model)


## Dataset
- [Leaf Segmentation Dataset](https://huggingface.co/datasets/godoldol99/SAM2-based-plant-disease-lesion-segmentation-model-DATASET)


## Python cod
```python
###################################################
# REF : https://learnopencv.com/finetuning-sam2/
###################################################

# 1. Git clone
!git clone https://github.com/facebookresearch/segment-anything-2
%cd ./segment-anything-2
!pip install -e .

# 2. Install kaggle
pip install kaggle

# 3. get dataset from Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d ankanghosh651/leaf-sengmentation-dataset-sam2-format
!sudo apt-get install zip unzip
!unzip leaf-sengmentation-dataset-sam2-format.zip -d ./leaf-seg 

# 4. Load model parameter
!wget -O sam2_hiera_tiny.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
!wget -O sam2_hiera_small.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
!wget -O sam2_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
!wget -O sam2_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


$5. Import modules
import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
 
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


#5. Setting seeds
def set_seeds():
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
 
set_seeds()

#6. Load Images
data_dir = "./leaf-seg/leaf-seg"
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")
 
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
 
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
 
train_data = []
for index, row in train_df.iterrows():
   image_name = row['imageid']
   mask_name = row['maskid']
   train_data.append({
       "image": os.path.join(images_dir, image_name),
       "annotation": os.path.join(masks_dir, mask_name)
   })
 
test_data = []
for index, row in test_df.iterrows():
   image_name = row['imageid']
   mask_name = row['maskid']
   test_data.append({
       "image": os.path.join(images_dir, image_name),
       "annotation": os.path.join(masks_dir, mask_name)
   })


#7. Batch read preparation
def read_batch(data, visualize_data=True):
   ent = data[np.random.randint(len(data))]
   Img = cv2.imread(ent["image"])[..., ::-1]  
   ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
 
   if Img is None or ann_map is None:
       print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
       return None, None, None, 0
 
   r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
   Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
   ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                        interpolation=cv2.INTER_NEAREST)
 
   binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
   points = []
   inds = np.unique(ann_map)[1:]
   for ind in inds:
       mask = (ann_map == ind).astype(np.uint8)
       binary_mask = np.maximum(binary_mask, mask)
 
   eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
   coords = np.argwhere(eroded_mask > 0)
   if len(coords) > 0:
       for _ in inds:
           yx = np.array(coords[np.random.randint(len(coords))])
           points.append([yx[1], yx[0]])
   points = np.array(points)
 
   if visualize_data:
       plt.figure(figsize=(15, 5))
       plt.subplot(1, 3, 1)
       plt.title('Original Image')
       plt.imshow(Img)
       plt.axis('off')
 
       plt.subplot(1, 3, 2)
       plt.title('Binarized Mask')
       plt.imshow(binary_mask, cmap='gray')
       plt.axis('off')
 
       plt.subplot(1, 3, 3)
       plt.title('Binarized Mask with Points')
       plt.imshow(binary_mask, cmap='gray')
       colors = list(mcolors.TABLEAU_COLORS.values())
       for i, point in enumerate(points):
           plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100)
       plt.axis('off')
 
       plt.tight_layout()
       plt.show()
 
   binary_mask = np.expand_dims(binary_mask, axis=-1)
   binary_mask = binary_mask.transpose((2, 0, 1))
   points = np.expand_dims(points, axis=1)
   return Img, binary_mask, points, len(inds)
 
Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)


#8. Load checkpoint
sam2_checkpoint = "./sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
 
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
 
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)


#9. Set train(fine-tuning) scheduler
scaler = torch.amp.GradScaler()
NO_OF_STEPS = 6000
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"
 
optimizer = torch.optim.AdamW(params=predictor.model.parameters(),
                              lr=0.00005,
                              weight_decay=1e-4)
 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.6)
accumulation_steps = 8


#10. Train IoU
def train(predictor, train_data, step, mean_iou):    
    with torch.amp.autocast(device_type='cuda'):
        image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
         
        if image is None or mask is None or num_masks == 0:
            return
 
        input_label = np.ones((num_masks, 1))
         
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            return
 
        if input_point.size == 0 or input_label.size == 0:
            return
 
        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
         
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            return
 
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )
 
        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
 
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
 
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
         
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
         
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()
         
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
 
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05
         
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
         
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
         
        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            predictor.model.zero_grad()
 
        scheduler.step()
         
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
         
        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Step {step}: Current LR = {current_lr:.6f}, IoU = {mean_iou:.6f}, Seg Loss = {seg_loss:.6f}")
    return mean_iou

# 11. Valid IoU
def validate(predictor, test_data, step, mean_iou):
    predictor.model.eval()
    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            image, mask, input_point, num_masks = read_batch(test_data, visualize_data=False)
             
            if image is None or mask is None or num_masks == 0:
                return
     
            input_label = np.ones((num_masks, 1))
             
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                return
     
            if input_point.size == 0 or input_label.size == 0:
                return
     
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
             
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                return
     
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
 
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
 
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
 
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
 
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6)
                        - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()
 
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
 
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05
            loss = loss / accumulation_steps
 
            if step % 500 == 0:
                FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".pt"
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)
             
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
 
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {step}: Current LR = {current_lr:.6f}, Valid_IoU = {mean_iou:.6f}, Valid_Seg Loss = {seg_loss:.6f}")
    return mean_iou

# 12. Train code
train_mean_iou = 0
valid_mean_iou = 0
 
for step in range(1, NO_OF_STEPS + 1):
    train_mean_iou = train(predictor, train_data, step, train_mean_iou)
    valid_mean_iou = validate(predictor, test_data, step, valid_mean_iou)

# 13. read and resize image and mask
def read_image(image_path, mask_path):  # read and resize image and mask
   img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
   mask = cv2.imread(mask_path, 0)
   r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
   img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
   mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
   return img, mask
 
def get_points(mask, num_points):  # Sample points inside the input mask
   points = []
   coords = np.argwhere(mask > 0)
   for i in range(num_points):
       yx = np.array(coords[np.random.randint(len(coords))])
       points.append([[yx[1], yx[0]]])
   return np.array(points)

# 14. Print final results
# Metric 계산 함수
def compute_metrics(pred_mask, gt_mask):
    """
    pred_mask: 예측 segmentation map (0/1/2/...)
    gt_mask: GT mask (0/1), 즉 binary mask
    """
    # 예측을 binary mask로 변환 (0=배경, 1=object)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    union = np.logical_or(pred_binary == 1, gt_binary == 1).sum()
    iou = intersection / union if union != 0 else 0.0

    correct = (pred_binary == gt_binary).sum()
    total = gt_binary.size
    pixel_acc = correct / total

    return iou, pixel_acc


# 전체 결과 저장 리스트
mIoU_list = []
pixel_acc_list = []

# Loop through all test samples
for idx, selected_entry in enumerate(test_data):
    print(f"\n===== Test Sample {idx+1}/{len(test_data)} =====")
    print(selected_entry)

    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']

    # Load the selected image and mask
    image, target_mask = read_image(image_path, mask_path)

    # Generate random points for inference
    num_samples = 30
    input_points = get_points(target_mask, num_samples)

    # Perform inference
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Sort predicted masks by score
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    # Build final segmentation map
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue

        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False
        seg_map[mask_bool] = i + 1
        occupancy_mask[mask_bool] = True

    # ----- Compute Metrics -----
    iou, pixel_acc = compute_metrics(seg_map, target_mask)
    mIoU_list.append(iou)
    pixel_acc_list.append(pixel_acc)

    print(f"mIoU: {iou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")

    # Visualization
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Original Mask Overlay')
    plt.imshow(image)
    plt.imshow(target_mask, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation Overlay')
    plt.imshow(image)
    plt.imshow(seg_map, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ===== 전체 평균 metric 출력 =====
mean_mIoU = np.mean(mIoU_list)
mean_pixel_acc = np.mean(pixel_acc_list)

print("\n======================")
print(" Test Set Evaluation ")
print("======================")
print(f"Average mIoU: {mean_mIoU:.4f}")
print(f"Average Pixel Accuracy: {mean_pixel_acc:.4f}")
