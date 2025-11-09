"""
Create GIF from demo results
"""

import cv2
import glob
import os
from pathlib import Path

def images_to_gif(output_dir="demo_output", gif_name="face_recognition_demo.gif", duration_per_frame=1000):
    """
    Convert PNG images to GIF
    
    Args:
        output_dir: Directory containing demo result images
        gif_name: Output GIF filename
        duration_per_frame: Duration to show each frame in ms
    """
    import imageio
    
    print(f"Creating GIF from demo results...")
    
    # Get all result images (excluding composite)
    image_files = sorted(glob.glob(os.path.join(output_dir, "result_*.png")))
    
    if not image_files:
        print("Error: No result images found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Read images
    images = []
    target_size = None
    
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is not None:
            # Set target size from first image
            if target_size is None:
                target_size = (400, 400)  # Standard demo size
            
            # Resize to target size
            img = cv2.resize(img, target_size)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            print(f"  ✓ Loaded {os.path.basename(img_file)}")
    
    if not images:
        print("Error: Could not load any images!")
        return
    
    # Create GIF
    try:
        imageio.mimsave(gif_name, images, duration=duration_per_frame/1000)
        print(f"\n✓ GIF saved: {gif_name}")
        
        # Get file size
        size_kb = os.path.getsize(gif_name) / 1024
        print(f"  File size: {size_kb:.1f} KB")
        
    except ImportError:
        print("\nError: imageio not installed!")
        print("Install with: pip install imageio pillow")
        print("\nAlternatively, use FFmpeg to create video:")
        print("  ffmpeg -framerate 1 -i demo_output/result_%d_*.png -c:v libx264 -pix_fmt yuv420p face_recognition_demo.mp4")
    except Exception as e:
        print(f"Error creating GIF: {e}")

def create_video_instructions():
    """Print instructions for creating video"""
    print("\n" + "="*60)
    print("Alternative: Create MP4 Video with FFmpeg")
    print("="*60)
    print("\nIf you have FFmpeg installed, run:")
    print("\n  ffmpeg -framerate 1 \\")
    print("    -pattern_type glob -i 'demo_output/result_*.png' \\")
    print("    -c:v libx264 -pix_fmt yuv420p \\")
    print("    face_recognition_demo.mp4")
    print("\nOr with higher frame rate:")
    print("\n  ffmpeg -framerate 2 \\")
    print("    -pattern_type glob -i 'demo_output/result_*.png' \\")
    print("    -c:v libx264 -pix_fmt yuv420p \\")
    print("    -vf 'fps=30' face_recognition_demo.mp4")
    print("="*60 + "\n")

if __name__ == "__main__":
    images_to_gif(duration_per_frame=2000)  # Show each image for 2 seconds
    create_video_instructions()
