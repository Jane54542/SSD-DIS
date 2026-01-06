import os
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image

# ============================================================================
# CONFIGURATION PARAMETERS - USERS CAN MODIFY THESE PATHS AND PARAMETERS
# ============================================================================

# Shadow mask library folder (contains all available shadow mask images)
SHADOW_MASK_LIBRARY_DIR = 'D:\\shadow_document_datasets\\semi-synthetic dataset\\shadows'

# Shadow-free document images folder (users need to provide their own shadow-free document images)
# Note: Choose the appropriate folder based on your needs (train, val, or test)
SHADOW_FREE_IMAGES_DIR = 'D:\\shadow_document_datasets\\semi-synthetic dataset\\shadow_free_images\\test'  # Change to 'train' or 'val' as needed

# Output folder settings (it's recommended to create these directories before running)
OUTPUT_SHADOW_MASKS_DIR = 'D:\\shadow_document_datasets\\semi-synthetic dataset\\vers_shadow_masks\\val'
OUTPUT_SHADOW_IMAGES_DIR = 'D:\\shadow_document_datasets\\semi-synthetic dataset\\vers_shadow_images\\val'
OUTPUT_ADJUSTED_MASKS_DIR = 'D:\\shadow_document_datasets\\semi-synthetic dataset\\vers_shadow_masks_intensity_adjusted\\val'

# Shadow intensity range (calibrated based on human perception, recommended: 0.15-0.8)
# The intensity range used in the paper is (0.15, 0.8)
SHADOW_INTENSITY_RANGE = (0.15, 0.8)  # Can be adjusted to (0.8, 1.0) for stronger shadows

# ============================================================================
# MAIN PROGRAM STARTS HERE
# ============================================================================

def synthesize_shadow_dataset():
    """
    Main function: Batch synthesize document shadow dataset
    Functionality: Randomly select masks from the shadow mask library, resize them, 
                   and apply them to shadow-free document images to generate 
                   document images with synthetic shadows and corresponding shadow masks.
    """
    
    # 1. Check if input directories exist
    if not os.path.exists(SHADOW_FREE_IMAGES_DIR):
        print(f"Error: Shadow-free images directory does not exist - {SHADOW_FREE_IMAGES_DIR}")
        return
    
    if not os.path.exists(SHADOW_MASK_LIBRARY_DIR):
        print(f"Error: Shadow mask library directory does not exist - {SHADOW_MASK_LIBRARY_DIR}")
        return
    
    # 2. Create output directories if they don't exist
    for output_dir in [OUTPUT_SHADOW_MASKS_DIR, OUTPUT_SHADOW_IMAGES_DIR, OUTPUT_ADJUSTED_MASKS_DIR]:
        os.makedirs(output_dir, exist_ok=True)
    
    # 3. Get all shadow-free document image files
    shadow_free_files = [f for f in os.listdir(SHADOW_FREE_IMAGES_DIR) 
                         if os.path.isfile(os.path.join(SHADOW_FREE_IMAGES_DIR, f))]
    
    print(f"Starting to process {len(shadow_free_files)} shadow-free document images...")
    print(f"Shadow intensity range: {SHADOW_INTENSITY_RANGE[0]:.2f} - {SHADOW_INTENSITY_RANGE[1]:.2f}")
    print(f"Output directory: {OUTPUT_SHADOW_IMAGES_DIR}")
    
    # 4. Get all mask files from the shadow mask library
    shadow_mask_files = os.listdir(SHADOW_MASK_LIBRARY_DIR)
    if not shadow_mask_files:
        print("Error: No image files found in the shadow mask library")
        return
    
    # 5. Process each shadow-free document image
    for shadow_free_file in shadow_free_files:
        # 5.1 Randomly select a shadow mask from the library
        random_mask_file = random.choice(shadow_mask_files)
        
        # 5.2 Read the original shadow mask and resize it to match the document image
        mask_source_path = os.path.join(SHADOW_MASK_LIBRARY_DIR, random_mask_file)
        mask_target_path = os.path.join(OUTPUT_SHADOW_MASKS_DIR, shadow_free_file)
        
        # Open the shadow mask and resize it
        with Image.open(mask_source_path) as mask_img:
            # Read the corresponding shadow-free document image to get the target size
            shadow_free_path = os.path.join(SHADOW_FREE_IMAGES_DIR, shadow_free_file)
            with Image.open(shadow_free_path) as doc_img:
                # Resize the shadow mask to the same dimensions as the document image
                resized_mask = mask_img.resize(doc_img.size, Image.Resampling.LANCZOS)
                # Save the resized mask
                resized_mask.save(mask_target_path)
        
        print(f"Processing: {shadow_free_file} | Using mask: {random_mask_file} | Saved to: {mask_target_path}")
        
        # 5.3 Synthesize the shadowed document image
        synthesize_shadow_image(shadow_free_file)
    
    print("=" * 80)
    print("Data processing completed!")
    print(f"Number of generated shadowed images: {len(shadow_free_files)}")
    print(f"Output file locations:")
    print(f"  - Shadow masks: {OUTPUT_SHADOW_MASKS_DIR}")
    print(f"  - Synthetic shadow images: {OUTPUT_SHADOW_IMAGES_DIR}")
    print(f"  - Adjusted masks: {OUTPUT_ADJUSTED_MASKS_DIR}")

def synthesize_shadow_image(shadow_free_filename):
    """
    Synthesize a single shadowed document image
    
    Parameters:
        shadow_free_filename: Filename of the shadow-free document image
    """
    # 1. Read the shadow-free document image
    shadow_free_path = os.path.join(SHADOW_FREE_IMAGES_DIR, shadow_free_filename)
    shadow_free_image = read_image(shadow_free_path)
    
    # 2. Read the resized shadow mask
    shadow_mask_path = os.path.join(OUTPUT_SHADOW_MASKS_DIR, shadow_free_filename)
    shadow_mask = read_image(shadow_mask_path)
    
    # 3. Generate random shadow intensity for the current image
    # Note: Using the globally defined SHADOW_INTENSITY_RANGE
    shadow_intensity = random.uniform(SHADOW_INTENSITY_RANGE[0], SHADOW_INTENSITY_RANGE[1])
    
    # 4. Convert the shadow mask to a tensor and adjust intensity
    # Shadow masks are typically binary images (0-255), normalize to 0-1 range
    shadow_tensor = 1.0 - shadow_mask.float() / 255.0
    
    # Apply shadow intensity: shadowed area = shadow_intensity, non-shadowed area = 1.0
    shadow_tensor = shadow_intensity + (1.0 - shadow_intensity) * shadow_tensor
    
    # 5. Normalize the shadow-free document image to [0, 1] range
    shadow_free_tensor = shadow_free_image.float() / 255.0
    
    # 6. Expand the channel dimension of the shadow tensor to match the document image (single-channel -> three-channel)
    shadow_tensor = shadow_tensor.expand_as(shadow_free_tensor)
    
    # 7. Apply shadow: Multiply the document image with the shadow tensor
    # Shadowed areas become darker, non-shadowed areas remain unchanged
    shadowed_image = shadow_free_tensor * shadow_tensor
    
    # 8. Save the synthesized shadowed document image
    shadowed_image = (shadowed_image * 255.0).byte()  # Convert back to [0, 255] range
    output_image_path = os.path.join(OUTPUT_SHADOW_IMAGES_DIR, shadow_free_filename)
    transforms.ToPILImage()(shadowed_image).save(output_image_path)
    
    # 9. Save the intensity-adjusted shadow mask (inverted, for visualization)
    # Invert the shadow tensor: shadowed area = 0, non-shadowed area = 1
    adjusted_mask = 1.0 - shadow_tensor
    adjusted_mask = (adjusted_mask * 255.0).byte()
    adjusted_mask_path = os.path.join(OUTPUT_ADJUSTED_MASKS_DIR, shadow_free_filename)
    transforms.ToPILImage()(adjusted_mask).save(adjusted_mask_path)

# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Print program information
    print("=" * 80)
    print("SSD-DIS Dataset Generation Tool")
    print("Function: Synthesize shadowed document images from shadow-free images and shadow mask library")
    print("=" * 80)
    
    # Execute the synthesis process
    synthesize_shadow_dataset()