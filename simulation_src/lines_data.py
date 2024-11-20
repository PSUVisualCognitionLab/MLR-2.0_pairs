import random
from PIL import Image, ImageDraw
import argparse
import math
import torch
import torchvision.transforms as transforms

# Define the colors
colors = ["white", "blue", "green", "yellow", "red", "pink", "orange", "purple", "cyan", "magenta", "brown"]

# Define the image size and line parameters
image_size = (500, 500)
line_length = 140
line_width = 20

def generate_line_grid(num_lines, grid_size, same_color, same_angle, generate_changes=False):
    # Create a black background image
    image = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(image)
    original_frames = []
    change_frames = []  # Will contain one set of frames with a single change

    # Calculate line positions based on grid size and center the grid
    if grid_size == 4:
        x_positions = [image_size[0] // 4, 3 * image_size[0] // 4]
        y_positions = [image_size[1] // 4, 3 * image_size[1] // 4]
    elif grid_size == 8:
        x_positions = [image_size[0] // 8, 3 * image_size[0] // 8, 5 * image_size[0] // 8, 7 * image_size[0] // 8]
        y_positions = [image_size[1] // 8, 3 * image_size[1] // 8, 5 * image_size[1] // 8, 7 * image_size[1] // 8]
    else:  # grid_size == 9
        x_positions = [image_size[0] // 6, 3 * image_size[0] // 6, 5 * image_size[0] // 6]
        y_positions = [image_size[1] // 6, 3 * image_size[1] // 6, 5 * image_size[1] // 6]

    # Choose colors and angles for original frames
    if same_color:
        selected_color = random.choice(colors)
        original_colors = [selected_color] * num_lines
    else:
        original_colors = [random.choice(colors) for _ in range(num_lines)]
    
    if same_angle:
        base_angle = random.randint(0, 179)
        original_angles = [base_angle] * num_lines
    else:
        original_angles = [random.randint(0, 179) for _ in range(num_lines)]

    positions = [(x, y) for x in x_positions for y in y_positions][:num_lines]

    # Generate original frames
    for i, (x, y) in enumerate(positions):
        frame = Image.new("RGB", image_size, "black")
        draw_frame = ImageDraw.Draw(frame)

        x1 = x - line_length // 2
        y1 = y - line_width // 2
        x2 = x + line_length // 2
        y2 = y + line_width // 2

        draw_frame.line([(x1, y1), (x2, y2)], fill=original_colors[i], width=line_width)
        
        if not same_angle:
            frame = frame.rotate(original_angles[i], expand=True, center=(x, y))

        original_frames.append(frame)

    # Generate change frames if requested
    if generate_changes:
        # Choose which frame to change
        change_idx = random.randint(0, num_lines - 1)
        
        # Create the changed set of frames
        for i, (x, y) in enumerate(positions):
            frame = Image.new("RGB", image_size, "black")
            draw_frame = ImageDraw.Draw(frame)

            x1 = x - line_length // 2
            y1 = y - line_width // 2
            x2 = x + line_length // 2
            y2 = y + line_width // 2

            if i == change_idx:  # This is the frame we want to change
                if same_angle:
                    # Change color only
                    available_colors = [c for c in colors if c != original_colors[i]]
                    new_color = random.choice(available_colors)
                    draw_frame.line([(x1, y1), (x2, y2)], fill=new_color, width=line_width)
                    if not same_angle:
                        frame = frame.rotate(original_angles[i], expand=True, center=(x, y))
                elif same_color:
                    # Change angle only
                    new_angle = (original_angles[i] + random.randint(30, 150)) % 180
                    draw_frame.line([(x1, y1), (x2, y2)], fill=original_colors[i], width=line_width)
                    frame = frame.rotate(new_angle, expand=True, center=(x, y))
                else:
                    # Change both color and angle
                    available_colors = [c for c in colors if c != original_colors[i]]
                    new_color = random.choice(available_colors)
                    new_angle = (original_angles[i] + random.randint(30, 150)) % 180
                    draw_frame.line([(x1, y1), (x2, y2)], fill=new_color, width=line_width)
                    frame = frame.rotate(new_angle, expand=True, center=(x, y))
            else:
                # Keep the original frame
                draw_frame.line([(x1, y1), (x2, y2)], fill=original_colors[i], width=line_width)
                if not same_angle:
                    frame = frame.rotate(original_angles[i], expand=True, center=(x, y))

            change_frames.append(frame)

    return image, original_frames, change_frames

def save_images(original_image, original_frames, change_frames, index):
    original_image.save(f"original_image_{index}.png")
    for i, frame in enumerate(original_frames):
        frame.save(f"original_frame{i}_{index}.png")
    for i, frame in enumerate(change_frames):
        frame.save(f"change_frame{i}_{index}.png")

def main(args):
    num_images = args.num_images
    num_lines = args.num_lines
    same_color = args.same_color
    same_angle = args.same_angle
    to_tensor = args.to_tensor
    
    if num_lines != 4:
        grid_size = 9
    else:
        grid_size = 4

    if same_angle and not same_color:
        name = '8c'
    elif not same_angle and same_color:
        name = '8o'
    else:
        name = '4c4o'

    if to_tensor:
        totensor = transforms.ToTensor()
        all_original_frames = []
        all_change_frames = []

        for _ in range(num_images):
            original_image, original_frames, change_frames = generate_line_grid(
                num_lines, grid_size, same_color=same_color, same_angle=same_angle, generate_changes=True
            )
            
            # Process original frames
            original_frames = [frame.resize((28, 28)) for frame in original_frames]
            original_frames = [totensor(frame).view(1, 3, 28, 28) for frame in original_frames]
            all_original_frames.append(torch.stack(original_frames))
            
            # Process change frames
            change_frames = [frame.resize((28, 28)) for frame in change_frames]
            change_frames = [totensor(frame).view(1, 3, 28, 28) for frame in change_frames]
            all_change_frames.append(torch.stack(change_frames))

        # Stack all frames
        all_original_frames = torch.stack(all_original_frames)
        all_change_frames = torch.stack(all_change_frames)
        
        print(f"Original frames size: {all_original_frames.size()}")
        print(f"Change frames size: {all_change_frames.size()}")
        
        # Save both original and change frames
        torch.save({
            'original_frames': all_original_frames,
            'change_frames': all_change_frames
        }, f'frames_{name}1.pth')

    else:
        for count in range(num_images):
            original_image, original_frames, change_frames = generate_line_grid(
                num_lines, grid_size, same_color=same_color, same_angle=same_angle, generate_changes=True
            )
            save_images(original_image, original_frames, change_frames, count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a grid of lines for change detection tasks.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--num_lines", type=int, default=4, help="Number of lines in the grid (4 or 8)")
    parser.add_argument("--same_color", type=bool, default=False, help="Whether all lines should have the same color")
    parser.add_argument("--same_angle", type=bool, default=False, help="Whether all lines should have the same rotation angle")
    parser.add_argument("--to_tensor", type=bool, default=False, help="Save as tensor")
    args = parser.parse_args()
    main(args)

    args.num_lines = args.num_lines * 2
    args.same_color = True
    main(args)

    args.same_color = False
    args.same_angle = True
    main(args)