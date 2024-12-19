from PIL import Image
import numpy as np

def generate_gif(frames, output_path, duration=200):
    # frames: 3D NumPy array of image frames
    # output_path: Path to save the generated GIF
    # duration: Duration (in milliseconds) for each frame

    pil_frames = []
    for frame in frames:
        # Convert each 2D frame to PIL Image
        pil_frame = Image.fromarray(frame)
        pil_frames.append(pil_frame)

    # Save the frames as GIF using the save method of the first frame
    pil_frames[0].save(
        output_path,
        format='GIF',
        append_images=pil_frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )

# Example usage:
# Assuming you have a 3D NumPy array of image frames called 'frames' and you want to save the GIF as 'output.gif'
# tmp_im = xHatAbs/np.amax(xHatAbs)*1.5 
# tmp_im[tmp_im > 1] = 1
# tmp_im = (tmp_im * 255).astype(np.uint8)
# generate_gif(tmp_im, 'saved_S_output_2e2_it'+str(ii+1)+'.gif', 80)
