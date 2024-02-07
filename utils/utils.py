import matplotlib.pyplot as plt
from IPython.display import clear_output


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
