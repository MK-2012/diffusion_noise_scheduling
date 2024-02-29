from argparse import ArgumentParser
from json import dump
from numpy import load as np.load

from metrics.common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from metrics.common_metrics_on_video_quality.calculate_psnr import calculate_psnr
from metrics.common_metrics_on_video_quality.calculate_ssim import calculate_ssim
from metrics.common_metrics_on_video_quality.calculate_lpips import calculate_lpips

from torch import load, as_tensor


def _load_dataset(path, device="cpu"):
	try:
		dataset = load(base_path, map_location=device)
	except:
		try:
			base_dataset = as_tensor(np.load(base_path), device=device)
		except:
			raise Exception("Cannot load this dataset format")

def compare(base_path, compare_path, base_type="dataset", compare_type="dataset", device="cuda:0"):
	if base_type != "dataset" or compare_type != "dataset":
		raise Exception("Not implemented yet")
	
	base_dataset = _load_dataset(base_path)
	comp_dataset = _load_dataset(compare_path)

	result = {}
	result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
	# result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt')
	result['ssim'] = calculate_ssim(videos1, videos2)
	result['psnr'] = calculate_psnr(videos1, videos2)
	result['lpips'] = calculate_lpips(videos1, videos2, device)

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
	parser.add_argument("--comparetype", default="dataset", choices=["dataset", "model_weights"])
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
