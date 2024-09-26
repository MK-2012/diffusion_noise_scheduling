from os import listdir
from os.path import join
from argparse import ArgumentParser
from json import dump
from numpy import load as np_load
from tqdm import tqdm

from metrics.common_metrics_on_video_quality.calculate_fvd import calculate_fvd
# from metrics.common_metrics_on_video_quality.calculate_psnr import calculate_psnr
# from metrics.common_metrics_on_video_quality.calculate_ssim import calculate_ssim
# from metrics.common_metrics_on_video_quality.calculate_lpips import calculate_lpips

from torch import load, as_tensor, stack
from torchvision.io import read_video


def _load_folder(path, device="cpu"):
	video_names = listdir(path)
	
	saved_videos = []
	for p in tqdm(video_names):
		v, _, _ = read_video(join(path, p), pts_unit="sec")
		saved_videos.append(v)
	ret_videos = stack(saved_videos).permute(0, 1, 4, 2, 3).to(device) # Делаем B T C H W

	return ret_videos

def _load_dataset(path, device="cpu"):
	try:
		dataset = load(path, map_location=device)
	except:
		try:
			dataset = as_tensor(np_load(path), device=device)
		except:
			raise Exception(f"Cannot load this dataset format: {path}")

	return dataset


def compare(base_path, compare_path, base_type="dataset", compare_type="dataset", device="cuda:0"):
	match base_type:
		case "dataset":
			base_dataset = _load_dataset(base_path, device=device)
		case "directory":
			base_dataset = _load_folder(base_path, device=device)
		case _:
			raise Exception("Not implemented yet")
	print("Loading base done")

	match compare_type:
		case "dataset":
			comp_dataset = _load_dataset(compare_path, device=device)
		case "directory":
			comp_dataset = _load_folder(compare_path, device=device)
		case _:
			raise Exception("Not implemented yet")
	print("Loading comp done")

	# Нужен формат B, T, C, H, W
	len_ = min(base_dataset.shape[0], comp_dataset.shape[0])
	print(f"Resulting length: {len_}")
	base_dataset = base_dataset[:len_].repeat(1, 1, 3, 1, 1)
	comp_dataset = comp_dataset[:len_]

	result = {}
	result['fvd'] = calculate_fvd(base_dataset, comp_dataset, device, method='styleganv')
	# result['fvd'] = calculate_fvd(base_dataset, comp_dataset, device, method='videogpt')
	# result['ssim'] = calculate_ssim(base_dataset, comp_dataset)
	# result['psnr'] = calculate_psnr(base_dataset, comp_dataset)
	# result['lpips'] = calculate_lpips(base_dataset, comp_dataset, device)

	return result

def save(result, save_path):
	with open(save_path, "w", encoding="UTF-8") as f:
		dump(result, f, indent=4)
	

if __name__ == "__main__":
	parser = ArgumentParser(
		prog='Metric calculator',
	)
	parser.add_argument("--base_path", required=True)
	parser.add_argument("--compare_path", required=True)
	parser.add_argument("--save_path", required=True)
	parser.add_argument("--base_type", default="directory", choices=["dataset", "model_weights", "directory"])
	parser.add_argument("--compare_type", default="directory", choices=["dataset", "model_weights", "directory"])
	parser.add_argument("--device", default="cuda:0")

	args = parser.parse_args()

	result = compare(
		base_path=args.base_path,
		compare_path=args.compare_path,
		base_type=args.base_type,
		compare_type=args.compare_type,
		device=args.device,
	)
	save(
		result=result,
		save_path=args.save_path,
	)
