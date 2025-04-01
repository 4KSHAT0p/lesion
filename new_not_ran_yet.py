import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score, average_precision_score
import random
from torchvision.transforms import functional as TF
import cv2
from skimage import exposure, filters


# Advanced preprocessing functions for medical images
def clahe_enhancement(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image"""
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Convert to LAB color space for better contrast enhancement
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # For RGB images
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back to LAB and convert to RGB
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image_np)
    
    return Image.fromarray(enhanced_image)


def adaptive_gamma_correction(image):
    """Apply adaptive gamma correction based on image histogram"""
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Calculate image histogram
    if len(image_np.shape) == 3:
        # For RGB images, convert to grayscale for histogram analysis
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    # Calculate mean brightness
    mean = np.mean(gray) / 255.0
    
    # Adaptively determine gamma value (lower for darker images, higher for brighter)
    if mean < 0.4:
        gamma = 0.8  # Brighten dark images
    elif mean > 0.7:
        gamma = 1.2  # Darken bright images
    else:
        gamma = 1.0  # Leave mid-range alone
        
    # Apply gamma correction
    if len(image_np.shape) == 3:
        # RGB image
        enhanced_image = np.power(image_np/255.0, gamma) * 255.0
    else:
        # Grayscale
        enhanced_image = np.power(image_np/255.0, gamma) * 255.0
        
    return Image.fromarray(enhanced_image.astype(np.uint8))


def unsharp_masking(image):
    """Apply unsharp masking to enhance edges and details"""
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Create gaussian blur of the image
    blur = cv2.GaussianBlur(image_np, (0, 0), 3)
    
    # Subtract blurred image from original to get mask
    mask = cv2.addWeighted(image_np, 1.5, blur, -0.5, 0)
    
    # Clip values to valid range
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    return Image.fromarray(mask)


def vessel_enhancement(image):
    """Enhance vessels using multi-scale line detection approach"""
    # Convert to numpy if needed
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Convert to grayscale if color
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    # Apply Frangi filter for vessel enhancement
    try:
        from skimage.filters import frangi
        enhanced = frangi(gray)
        # Normalize to 0-255 range
        enhanced = exposure.rescale_intensity(enhanced, out_range=(0, 1))
        enhanced = (enhanced * 255).astype(np.uint8)
    except:
        # Fallback if frangi filter is not available
        enhanced = cv2.medianBlur(gray, 5)
        # Apply Laplacian filter to enhance edges
        laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
        enhanced = np.uint8(np.absolute(laplacian))
        
    # Convert back to original format
    if len(image_np.shape) == 3:
        # Create 3-channel image
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        # Blend with original for better results
        enhanced_result = cv2.addWeighted(image_np, 0.7, enhanced_color, 0.3, 0)
    else:
        enhanced_result = enhanced
        
    return Image.fromarray(enhanced_result)


class EnhancedMedicalTransform:
    """Custom transform class for medical image preprocessing pipeline"""
    def __init__(self, output_size=(256, 256), is_mask=False):
        self.output_size = output_size
        self.is_mask = is_mask
        
    def __call__(self, img):
        if self.is_mask:
            # For masks, just resize and convert to tensor
            img_resized = img.resize(self.output_size, Image.NEAREST)
            return TF.to_tensor(img_resized)
        
        # For medical images, apply advanced preprocessing
        # 1. CLAHE enhancement
        img_clahe = clahe_enhancement(img)
        
        # 2. Unsharp masking for detail enhancement
        img_sharp = unsharp_masking(img_clahe)
        
        # 3. Gamma correction
        img_gamma = adaptive_gamma_correction(img_sharp)
        
        # 4. Resize to target size
        img_resized = img_gamma.resize(self.output_size, Image.BILINEAR)
        
        # 5. Convert to tensor (0-1 range)
        img_tensor = TF.to_tensor(img_resized)
        
        # 6. Normalize using medical imaging statistics (approximated for retinal images)
        img_norm = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],  # ImageNet stats as fallback
            std=[0.229, 0.224, 0.225]
        )
        
        return img_norm


class LesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))  # Ensure masks match images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert `.tif` mask to grayscale

        # Apply same random transformations to both image and mask
        if self.augment:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation (0, 90, 180, or 270 degrees)
            rotation_angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, rotation_angle)
            mask = TF.rotate(mask, rotation_angle)

            # Random brightness adjustment (image only)
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)

            # Random contrast adjustment (image only)
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)

        # Apply custom medical image preprocessing
        if self.transform:
            # Apply standard transform from the caller
            image = self.transform(image)
            # For mask, ensure it's a single channel
            mask = TF.to_tensor(mask)
        else:
            # Apply our enhanced medical transform
            medical_transform = EnhancedMedicalTransform(output_size=(256, 256), is_mask=False)
            mask_transform = EnhancedMedicalTransform(output_size=(256, 256), is_mask=True)
            
            image = medical_transform(image)
            mask = mask_transform(mask)

        return image, mask


# Define paths (Change 'EX' to HE, MA, or SE as needed)
image_path = "train/image/"
mask_path = "train/label/EX/"
test_image_path = "test/image/"
test_mask_path = "test/label/EX/"
val_image_path = "valid/image/"
val_mask_path = "valid/label/EX/"

# Define transformations - now using our medical preprocessing pipeline
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets with augmentation
train_dataset = LesionDataset(image_path, mask_path, transform=None, augment=True)  # Using our custom preprocessing
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Check one batch
images, masks = next(iter(train_dataloader))
print(images.shape, masks.shape)  # Should be (B, C, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final_conv(dec1))


def dice_loss(pred, target):
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (
        (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    )


# Add focal loss for better handling of class imbalance
def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction="none")

    # Calculate focal weights
    pt = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - pt) ** gamma

    # Apply alpha weighting for class imbalance
    alpha_weight = target * alpha + (1 - target) * (1 - alpha)

    loss = focal_weight * alpha_weight * bce
    return loss.mean()


# Combined loss function
def bce_dice_focal_loss(
    pred, target, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2
):
    bce = F.binary_cross_entropy(pred, target, reduction="mean")
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return bce_weight * bce + dice_weight * dice + focal_weight * focal


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    validation_loader=None,
):
    model.to(device)
    best_val_loss = float("inf")
    best_model_state = None

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        # Validation phase
        if validation_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_images, val_masks in validation_loader:
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)

                    val_outputs = model(val_images)
                    val_loss_batch = criterion(val_outputs, val_masks)
                    val_loss += val_loss_batch.item() * val_images.size(0)

            val_loss = val_loss / len(validation_loader.dataset)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Load best model if validation was used
    if validation_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    iou_scores = []
    dice_scores = []  # Also track Dice scores

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred_masks = (outputs > 0.5).float()

            # Calculate metrics for batch
            for pred, true in zip(pred_masks, masks):
                pred_np = pred.cpu().numpy().flatten()
                true_np = true.cpu().numpy().flatten()

                # Handle empty mask case
                if np.sum(true_np) == 0 and np.sum(pred_np) == 0:
                    iou = 1.0  # Perfect match when both are empty
                    dice = 1.0
                elif np.sum(true_np) == 0 or np.sum(pred_np) == 0:
                    iou = 0.0  # No overlap when one is empty
                    dice = 0.0
                else:
                    # Calculate IoU
                    iou = jaccard_score(true_np > 0.5, pred_np > 0.5, average="binary")

                    # Calculate Dice coefficient
                    intersection = np.sum(pred_np * true_np)
                    dice = (2.0 * intersection) / (
                        np.sum(pred_np) + np.sum(true_np) + 1e-8
                    )

                iou_scores.append(iou)
                dice_scores.append(dice)

    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    print(f"Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
    return mean_iou, mean_dice


def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    images, masks = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        preds = model(images)
        preds = (preds > 0.5).float()

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(min(num_samples, len(images))):
        # Original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        img_np = images[i].permute(1, 2, 0).numpy()
        
        # Denormalize for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks[i].squeeze(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        # Predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(preds[i].squeeze(), cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_visualization.png")
    plt.show()


def evaluate_on_test_data(model_path, test_image_dir, test_mask_dir, device, transform=None):
    # 1. Load the trained model
    model = UNet()  # Initialize with same architecture as training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # 2. Create test dataset and dataloader
    test_dataset = LesionDataset(test_image_dir, test_mask_dir, transform=transform, augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 3. Initialize metrics
    iou_scores = []
    dice_scores = []
    all_predictions = []
    all_targets = []

    # 4. Evaluation loop
    with torch.no_grad():  # No need to track gradients for evaluation
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()  # Threshold at 0.5

            # Store raw predictions and binary targets for AP calculation
            binary_masks = (masks > 0.5).float()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(binary_masks.cpu().numpy().flatten())

            # Calculate IoU and Dice for each image in batch
            for pred, true in zip(pred_masks, masks):
                pred_flat = pred.cpu().numpy().flatten()
                true_flat = true.cpu().numpy().flatten()

                # Handle empty masks
                if np.sum(true_flat) == 0 and np.sum(pred_flat) == 0:
                    iou = 1.0
                    dice = 1.0
                elif np.sum(true_flat) == 0 or np.sum(pred_flat) == 0:
                    iou = 0.0
                    dice = 0.0
                else:
                    # IoU (Jaccard index)
                    intersection = np.sum(pred_flat * true_flat)
                    union = np.sum(pred_flat) + np.sum(true_flat) - intersection
                    iou = intersection / (union + 1e-8)

                    # Dice coefficient
                    dice = (2.0 * intersection) / (
                        np.sum(pred_flat) + np.sum(true_flat) + 1e-8
                    )

                iou_scores.append(iou)
                dice_scores.append(dice)

    # Convert to binary arrays for AP calculation
    all_targets_binary = (np.array(all_targets) > 0.5).astype(np.int64)

    # Calculate AP
    ap = average_precision_score(all_targets_binary, np.array(all_predictions))

    # 5. Report results
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)

    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Mean IoU Score: {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")

    return {"ap": ap, "iou": mean_iou, "dice": mean_dice}


# Create validation dataset
val_dataset = LesionDataset(val_image_path, val_mask_path, transform=None, augment=False)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, optimizer, and loss function
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = bce_dice_focal_loss  # Use combined loss function

# Train the model with validation
trained_model = train_model(
    model,
    train_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    validation_loader=val_dataloader,
)

# Save the model
torch.save(trained_model.state_dict(), "lesion_segmentation_model_EX.pth")

# Run evaluation on testing data
print("RESULTS ON TESTING DATA")
resultsTEST = evaluate_on_test_data(
    "lesion_segmentation_model_EX.pth", test_image_path, test_mask_path, device, transform=None
)

# Fix the bug in the original code: val_image_path was used twice
print("RESULTS ON VALIDATION DATA")
resultsVAL = evaluate_on_test_data(
    "lesion_segmentation_model_EX.pth", val_image_path, val_mask_path, device, transform=None
)

# Print comparison summary
print("\nSUMMARY COMPARISON:")
print(f"TEST vs VALIDATION:")
print(f"AP:   {resultsTEST['ap']:.4f} vs {resultsVAL['ap']:.4f}")
print(f"IoU:  {resultsTEST['iou']:.4f} vs {resultsVAL['iou']:.4f}")
print(f"Dice: {resultsTEST['dice']:.4f} vs {resultsVAL['dice']:.4f}")

# Visualize predictions on test data
test_dataset_vis = LesionDataset(test_image_path, test_mask_path, transform=None, augment=False)
test_dataloader_vis = DataLoader(test_dataset_vis, batch_size=4, shuffle=True)
print("Visualizing predictions on test data...")
visualize_predictions(trained_model, test_dataloader_vis, device, num_samples=3)

# Additional Analysis: Generate precision-recall curve
def plot_precision_recall_curve(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            binary_masks = (masks > 0.5).float()
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(binary_masks.cpu().numpy().flatten())
    
    # Convert to binary arrays
    all_targets_binary = (np.array(all_targets) > 0.5).astype(np.int64)
    
    # Calculate precision-recall curve
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(all_targets_binary, np.array(all_predictions))
    pr_auc = auc(recall, precision)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.show()
    
    return pr_auc

# Generate PR curve for test data
print("Generating precision-recall curve...")
pr_auc = plot_precision_recall_curve(trained_model, test_dataloader, device)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Additional visualization: Show challenging cases
def visualize_challenging_cases(model, dataloader, device, num_samples=3):
    model.eval()
    images_list = []
    masks_list = []
    preds_list = []
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # Calculate Dice for each image
            for i in range(images.size(0)):
                pred_np = preds[i].cpu().numpy().flatten()
                true_np = masks[i].cpu().numpy().flatten()
                
                # Calculate Dice coefficient
                intersection = np.sum(pred_np * true_np)
                dice = (2.0 * intersection) / (np.sum(pred_np) + np.sum(true_np) + 1e-8)
                
                # Store data and score
                images_list.append(images[i].cpu())
                masks_list.append(masks[i].cpu())
                preds_list.append(preds[i].cpu())
                dice_scores.append(dice)
                
            if len(images_list) >= 20:  # Collect a reasonable number of samples
                break
    
    # Find indices of worst performing cases
    worst_indices = np.argsort(dice_scores)[:num_samples]
    
    plt.figure(figsize=(15, 5 * num_samples))
    for i, idx in enumerate(worst_indices):
        # Original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        img_np = images_list[idx].permute(1, 2, 0).numpy()
        
        # Denormalize for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        plt.imshow(img_np)
        plt.title(f"Challenging Image (Dice={dice_scores[idx]:.2f})")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks_list[idx].squeeze(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        # Predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(preds_list[idx].squeeze(), cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("challenging_cases.png")
    plt.show()

# Visualize challenging cases
print("Visualizing challenging cases...")
visualize_challenging_cases(trained_model, test_dataloader, device, num_samples=3)

print("Evaluation complete.")
