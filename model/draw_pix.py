import os
import sys
from PIL import Image, ImageDraw

def main():
    image_path = "/home/vlak/data/demo_video/00001.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    # Load the image
    img = Image.open(image_path)
    width, height = img.size
    print(f"Loaded image: {image_path} ({width}x{height})")

    # Define base point p0 (default to center of the image)
    # You can customize these coordinates as needed
    p0_x = width // 2
    p0_y = height // 2
    
    # Define points p1, p2, p3 at 4 and 8 pixel offsets
    # p1: 4 pixels away horizontally
    # p2: 8 pixels away horizontally
    # p3: 4 pixels away vertically (or 8 pixels away depending on requirements)
    p0 = (p0_x, p0_y)
    p1 = (p0_x + 4, p0_y)
    p2 = (p0_x + 8, p0_y)
    p3 = (p0_x, p0_y + 4)
    p4 = (p0_x, p0_y + 8) # Extra point at 8 pixels vertically

    print(f"p0 (base): {p0}")
    print(f"p1 (4px away horizontally): {p1}")
    print(f"p2 (8px away horizontally): {p2}")
    print(f"p3 (4px away vertically): {p3}")
    print(f"p4 (8px away vertically): {p4}")

    # Draw red points
    draw = ImageDraw.Draw(img)
    
    # We use a radius of 1 (a 3x3 square/circle) so the points are visible on high resolution images
    r = 1 
    fill_color = (255, 0, 0) # Red

    for name, pt in [("p0", p0), ("p1", p1), ("p2", p2), ("p3", p3), ("p4", p4)]:
        x, y = pt
        # Draw target pixel
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill_color)
        # Also draw a single pixel exactly on target (in case radius is too large)
        draw.point((x, y), fill=fill_color)

    # Save output image
    output_path = os.path.join(os.path.dirname(__file__), "annotated_points.jpg")
    img.save(output_path)
    print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    main()
