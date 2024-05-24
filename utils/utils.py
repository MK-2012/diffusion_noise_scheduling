from os.path import join
from torch import einsum, mean

import matplotlib.pyplot as plt
from IPython.display import clear_output

from imageio import mimsave
from torchvision.io import write_video, write_png


# Special loss with time correlations
def video_loss(pred, true, gram_matrix):
	diff = pred - true
	diff *= einsum("abi...,ij->abj...", diff, gram_matrix)
	return mean(diff)

# Simple callback for drawing frames from the first video of the batch
def draw_single_vid_frames(videos):
	clear_output(wait=True)

	video = ((videos[0].detach().cpu().clamp(-2, 2) / 2 + 1) * 255 / 2).detach().cpu().to(uint8).permute(1, 2, 3, 0)

	step = video.shape[0] // 5

	fig, ax = plt.subplots(nrows=1, ncols=5, constrained_layout=True)
	fig.set_size_inches(12, 4)

	for i in range(5):
		ax[i].imshow(video[i * step], cmap="grey")
		ax[i].axis("off")

	plt.show()

def save_images(images_tensor, base_path, prefix="", format="png"):
	match format:
		case "png":
			for i, im in enumerate(images_tensor):
				write_png(im, join(base_path, f"{prefix}_{i}.png"))

def save_videos(video_tensor, base_path, prefix="", fps=7, format="mp4"):
	match format:
		case "mp4":
			for i, v in enumerate(video_tensor):
				write_video(join(base_path, f"{prefix}_{i}.mp4"), v.repeat(3, 1, 1, 1).permute(1, 2, 3, 0), fps=7)
		case "gif":
			for i, v in enumerate(video_tensor):
				mimsave(uri=join(base_path, f"{prefix}_{i}.gif"), ims=video_tensor, fps=fps)
