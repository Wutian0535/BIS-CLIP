import os
from torch import nn
import argparse
import torch.backends.cudnn as cudnn
import clip
from utils.misc import *
from dataloader.data_utils import *
from utils.optimizer import build_optimizer
from utils.LossFunctions import *
from models.bis_clip import CLIPVIT
from engine import train, test


def main(args):
    setup_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    # Init Recoder
    record_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "-" + args.dataset + "-loss_" + args.loss_function + \
                  "-ep_" + str(args.epochs) + "-lr_" + str(args.lr) + "-bs_" + str(args.batch_size) + \
                  "-lam_" + str(args.lamda) + "-fix_" + str(args.fix_layer)
    args.record_path = os.path.join("outputs", "use_pgs", record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logger = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)

    # Init DataLoader

    set_up_datasets(args)
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataloader(args)
    get_labelname(args)

    # FER
    label_prompts = [f"a photo of a {label_nm} facial expression, a type of facial expression." for label_nm in args.label_nms]
    logger.info("First stage label prompts:{}".format(label_prompts))
    print("First stage label prompts:{}".format(label_prompts))

    label_token = clip.tokenize(label_prompts)

    # Build Model: CLIPVIT
    clip_model, _ = clip.load(args.clip_path, jit=False)
    print("loading clip from {}".format(args.clip_path))
    model = CLIPVIT(args, clip_model)
    convert_models_to_fp32(model)
    model = model.to(args.device)

    # Load CLIP
    clip_model, _ = clip.load(args.clip_path, jit=False)
    clip_model.train()
    clip_model = clip_model.to(args.device)

    # Build Optimizer
    optimizer = build_optimizer(args, model)

    if args.loss_function == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'focal':
        criterion = FocalLoss(class_num=args.classes, device=args.device)
    elif args.loss_function == 'balanced':
        criterion = BalancedLoss(class_num=args.classes, device=args.device)
    elif args.loss_function == 'cosine':
        criterion = CosineLoss()
    elif args.loss_function == 'gce':
        criterion = GCELoss(args.classes, device=args.device)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train(model, clip_model, args, optimizer, criterion, train_dataloader, logger, label_token, epoch)
        model.eval()
        val_acc, _ = test(model, args, criterion, val_dataloader, logger, label_token, epoch)

        if val_acc > args.save_acc_threshold:
            save_name = "model_epoch_{}_acc_{:.4f}.pth".format(epoch, float(val_acc))
            torch.save(model.state_dict(), os.path.join(args.record_path, save_name))
            logger.info("Saved checkpoint: {} (val_acc={:.4f})".format(save_name, float(val_acc)))
            print("Saved checkpoint: {} (val_acc={:.4f})".format(save_name, float(val_acc)))

#norm_pred=None
def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",                   type=int,   default=42  )
    parser.add_argument("--record_path",            type=str,   default='/home/CEPrompt/record')

    parser.add_argument('--classes', type=int, default=7)
    parser.add_argument('--dataset', type=str, default='rafdb', choices=['rafdb', 'affectnet', 'affectnet_8'])
    parser.add_argument('--data-path', type=str, default='./data/RAF-DB/basic/', choices=['/data/RAFDB/basic/', '/data/AffectNet/'])

    parser.add_argument("--clip-path", type=str, default='./data/CEPrompt_ckpt/pre-trained_model/ViT-B-16.pt',
                        choices=['/data/CEPrompt_ckpt/pre-trained_model/ViT-B-16.pt', '/data/CEPrompt_ckpt/pre-trained_model/ViT-B-32.pt', '/data/CEPrompt_ckpt/pre-trained_model/ViT-L-14.pt'])
    
    parser.add_argument("--batch-size",             type=int,   default=64,    )
    parser.add_argument("--test-batch-size",        type=int,   default=50,     )
    parser.add_argument("--epochs",                 type=int,   default=150,     )
    parser.add_argument("--warmup_epochs",          type=int,   default=5,      )
    parser.add_argument("--loss_function",          type=str,   default='ce', choices=['ce', 'focal', 'balanced', 'cosine', 'gce'])
    parser.add_argument("--lr",                     type=float, default=5e-6,   )
    parser.add_argument("--min_lr",                 type=float, default=1e-8,   )
    parser.add_argument("--weight_decay",           type=float, default=0.0005, )
    parser.add_argument("--workers",                type=int,   default=0,      )
    parser.add_argument("--momentum",               type=float, default=0.90,   )
    parser.add_argument("--lamda",                  type=float, default=1.5,      )
    parser.add_argument("--input_size",             type=int,   default=224,    )
    parser.add_argument("--gpu",                    type=int,   default=0      )
    parser.add_argument("--layer_decay",            type=float, default=0.65    )
    parser.add_argument("--fix_layer",              type=int,   default=2      )

    parser.add_argument("--save_acc_threshold", type=float, default=0.924)


    # 中间层蒸馏参数
    parser.add_argument("--use_teacher_distill", default=False, help='是否使用预训练模型的中间层进行蒸馏')
    parser.add_argument("--teacher_layer", type=int, default=4,
                      help='预训练模型中用于蒸馏的中间层索引（0-based），默认第6层')
    parser.add_argument("--distill_weight", type=float, default=1.5,
                      help='蒸馏损失的权重，默认0.5')

    parser.add_argument("--use_expr_restitution", default=True, action='store_true',
                      help='是否启用 Expression-Referenced OT Calibration 去身份风格并恢复表情区域残差')
    parser.add_argument("--expr_restitution_p", type=float, default=0.7,
                      help='ERIN training-time activation probability')
    parser.add_argument("--expr_style_strength", type=float, default=0.7,
                      help='ER-OTC expression-referenced OT calibration strength')
    parser.add_argument("--expr_restitution_strength", type=float, default=0,
                      help='表情关键区域 residual restitution 强度')
    parser.add_argument("--expr_restitution_center_sigma", type=float, default=0.5,
                      help='表情区域中心先验宽度')
    parser.add_argument("--use_identity_decorr", default=True, action='store_true',
                      help='enable identity-expression covariance decorrelation loss')
    parser.add_argument("--identity_decorr_weight", type=float, default=0.01,
                      help='weight for identity-expression decorrelation loss')
    # ===== PGS (Patch Generation-to-Selection) 参数 =====
    parser.add_argument("--use_pgs", default=True, action='store_true',
                      help='是否使用 Patch Generation-to-Selection 策略进行训练增强')
    parser.add_argument("--force_patch_dropout", type=float, default=0.05,
                      help='PGS 初始候选丢弃比例，默认0.05')
    parser.add_argument("--target_mask_ratio", type=float, default=0.3,
                      help='PGS 相似度阈值，控制哪些 patch 对不能同时被掩码，默认0.3')
    parser.add_argument("--min_mask_ratio", type=float, default=0.1,
                      help='PGS 最小掩码比例，默认0.1（FER任务建议低于CLIP-PGS的0.3-0.5）')
    parser.add_argument("--use_ed", default=True, action='store_true',
                      help='PGS 是否使用 Scharr 边缘检测保护面部关键区域')
    parser.add_argument("--use_otn", default=False, action='store_true',
                      help='PGS 是否使用最优传输归一化（Sinkhorn）平衡相似度矩阵')
    parser.add_argument("--use_dynamic_mask", default=True, action='store_true',
                      help='是否启用 Scharr + 流形距离的逐图动态 mask ratio（启用后不使用 OTN）')
    parser.add_argument("--dynamic_mask_min", type=float, default=0.05,
                      help='逐图动态 mask ratio 下界')
    parser.add_argument("--dynamic_mask_max", type=float, default=0.25,
                      help='逐图动态 mask ratio 上界')
    parser.add_argument("--dynamic_mask_lambda", type=float, default=1.0,
                      help='背景比例映射到动态 mask ratio 的强度')
    parser.add_argument("--protect_ratio", type=float, default=0.3,
                      help='Scharr 面部保护 patch 比例，保护的 patch 不会被 mask')
    parser.add_argument("--manifold_topk", type=int, default=6,
                      help='构建 patch 流形图时每个 patch 连接的近邻数量')
    parser.add_argument("--face_threshold", type=float, default=0.35,
                      help='估计背景比例时的面部保护分数阈值')
    # =====================================================

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # 打印蒸馏配置信息
    if hasattr(args, 'use_teacher_distill') and args.use_teacher_distill:
        print(f"Using teacher distillation with teacher layer: {args.teacher_layer}, distill weight: {args.distill_weight}")

    # 打印 PGS 配置信息
    if hasattr(args, 'use_pgs') and args.use_pgs:
        print(f"[PGS] Patch Generation-to-Selection enabled: "
              f"target_mask_ratio={args.target_mask_ratio}, "
              f"min_mask_ratio={args.min_mask_ratio}, "
              f"use_ed={args.use_ed}, use_otn={args.use_otn}, "
              f"dynamic_mask={args.use_dynamic_mask}, "
              f"dynamic_range=[{args.dynamic_mask_min}, {args.dynamic_mask_max}], "
              f"protect_ratio={args.protect_ratio}, manifold_topk={args.manifold_topk}, "
              f"expr_p={args.expr_restitution_p}, expr_ot_strength={args.expr_style_strength}, "
              f"expr_residual_strength={args.expr_restitution_strength}, "
              f"identity_decorr={args.use_identity_decorr}, decorr_weight={args.identity_decorr_weight}, "
              f"patch_dropout={args.force_patch_dropout}")

    main(args)
    

