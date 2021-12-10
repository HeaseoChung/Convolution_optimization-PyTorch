import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
from models import EDSR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EDSR(scale_factor=args.scale, num_channels=args.num_channels).to(device)
    inputs = torch.randn(1, 3, 1080, 1920, dtype=torch.float32).cuda()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of params : {pytorch_total_params}")
