"""Evaluate a RAF-DB checkpoint without starting training."""

import os
from datetime import datetime
import torch
from torch import nn
import clip
from dataloader.data_utils import get_labelname, set_up_datasets
from engine import test
from models.bis_clip import CLIPVIT
from train import build_parser
from utils.misc import convert_models_to_fp32, init_log, setup_seed


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA-enabled PyTorch installation.")
    torch.cuda.set_device(args.gpu)
    args.device = "cuda:{}".format(args.gpu)
    setup_seed(args.seed)
    set_up_datasets(args)
    get_labelname(args)
    dataset = args.Dataset.RafDB(
        root=args.data_path, train=False, index=range(args.classes), args=args
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    backbone, _ = clip.load(args.clip_path, device=args.device, jit=False)
    model = CLIPVIT(args, backbone)
    convert_models_to_fp32(model)
    # Load only checkpoints from a trusted source.
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    args.record_path = os.path.join(
        args.record_path, "eval_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    )
    os.makedirs(args.record_path, exist_ok=False)
    logger = init_log(args, args.record_path)
    prompts = [
        "a photo of a {} facial expression, a type of facial expression.".format(name)
        for name in args.label_nms
    ]
    accuracy, _ = test(
        model, args, nn.CrossEntropyLoss(), dataloader, logger, clip.tokenize(prompts)
    )
    print("RAF-DB accuracy: {:.4f}%".format(accuracy * 100))


if __name__ == "__main__":
    main()
