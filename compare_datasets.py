from argparse import ArgumentParser
from json import dump
from numpy import load as np_load

from metrics.common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from metrics.common_metrics_on_video_quality.calculate_psnr import calculate_psnr
from metrics.common_metrics_on_video_quality.calculate_ssim import calculate_ssim
from metrics.common_metrics_on_video_quality.calculate_lpips import calculate_lpips

from torch import load, as_tensor


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
	if base_type != "dataset" or compare_type != "dataset":
		raise Exception("Not implemented yet")
	
	base_dataset = _load_dataset(base_path)
	comp_dataset = _load_dataset(compare_path)
	len = min(base_dataset.shape[0], comp_dataset.shape[0])
	base_dataset = base_dataset[:len]
	comp_dataset = comp_dataset[:len].permute(0, 2, 1, 3, 4)

	result = {}
	result['fvd'] = calculate_fvd(base_dataset, comp_dataset, device, method='styleganv')
	# result['fvd'] = calculate_fvd(base_dataset, comp_dataset, device, method='videogpt')
	result['ssim'] = calculate_ssim(base_dataset, comp_dataset)
	result['psnr'] = calculate_psnr(base_dataset, comp_dataset)
	result['lpips'] = calculate_lpips(base_dataset, comp_dataset, device)

	return result

def save(result, save_path):
	with open(save_path, "w", encoding="UTF-8") as f:
		dump(result, f, indent=4)
	

if __name__ == "__main__":
	parser = ArgumentParser(
		prog='Metric calculator',
	)
	parser.add_argument("--base_path")
	parser.add_argument("--compare_path")
	parser.add_argument("--save_path")
	parser.add_argument("--base_type", default="dataset", choices=["dataset", "model_weights"])
	parser.add_argument("--compare_type", default="dataset", choices=["dataset", "model_weights"])
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
