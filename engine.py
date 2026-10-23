"""BIS-CLIP training and evaluation loops."""

import os
import json
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils.lr_sched as lrs
from utils.misc import compute_ACC, Averager
from torch.cuda.amp import autocast


def covariance_decorrelation_loss(z_expr, z_id, eps=1e-6):
    if z_expr is None or z_id is None or z_expr.size(0) <= 1:
        return None

    z_expr = z_expr.float()
    z_id = z_id.float()
    z_expr = (z_expr - z_expr.mean(dim=0, keepdim=True)) / (z_expr.std(dim=0, keepdim=True, unbiased=False) + eps)
    z_id = (z_id - z_id.mean(dim=0, keepdim=True)) / (z_id.std(dim=0, keepdim=True, unbiased=False) + eps)
    cov = z_expr.t().matmul(z_id) / max(z_expr.size(0) - 1, 1)
    return cov.pow(2).mean()


def train(model, clip_model, args, optimizer, criterion, dataloader, logger, label_token, epoch):
    logger.info("=======================TRAINING MODE, Epoch: {}/{}=======================".format(epoch, args.epochs))
    print("=======================TRAINING MODE, Epoch: {}/{}=======================".format(epoch, args.epochs))

    mean_dist_loss = 0
    mean_rank_loss = 0
    mean_decorr_loss = 0
    tl = Averager()
    ta = Averager()

    for i, (train_inputs, train_labels) in enumerate(tqdm(dataloader)):  # train_inputs: (bs,3,224,224), train_labels: (bs, 7)

        lrs.adjust_learning_rate(optimizer, i / len(dataloader) + epoch, args)

        # import pdb; pdb.set_trace() print(torch.nonzero(1+train_labels[2]))
        optimizer.zero_grad()

        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        label_token = label_token.cuda()


        logits, _, dist_feat = model(train_inputs, label_token, epoch=epoch)  # logits:(bs,7), _:(bs,196,512), dist_feat:(bs,512)

        rank_loss = criterion(logits, train_labels)

        # 使用预训练模型的中间层特征进行蒸馏
        if hasattr(args, 'use_teacher_distill') and args.use_teacher_distill:
            with torch.no_grad():
                # 从预训练模型的指定中间层提取特征
                teacher_feat = model.extract_teacher_features(
                    train_inputs, 
                    layer_idx=args.teacher_layer
                )

            # 计算蒸馏损失（学生模型的特征与教师模型中间层特征的L1距离）
            distill_loss = F.l1_loss(dist_feat, teacher_feat)

            # 总损失 = 分类损失 + 蒸馏损失
            loss = rank_loss + args.distill_weight * distill_loss

            # 用于日志记录
            dist_loss = distill_loss
        else:
            # 用CLIP的visual encoder提取local和global作为teacher, 不参加梯度更新
            with torch.no_grad():
                _, tea_dist_feat = clip_model.encode_image(train_inputs)

            dist_loss = F.l1_loss(dist_feat, tea_dist_feat.float())
            loss = rank_loss + args.lamda * dist_loss
        decorr_loss = logits.new_tensor(0.0)
        if getattr(args, 'use_identity_decorr', False):
            identity_style = model.get_identity_style() if hasattr(model, 'get_identity_style') else None
            loss_value = covariance_decorrelation_loss(logits, identity_style)
            if loss_value is not None:
                decorr_loss = loss_value
                loss = loss + args.identity_decorr_weight * decorr_loss

        acc = compute_ACC(logits, train_labels)

        tl.add(loss.item())
        ta.add(acc)
        mean_dist_loss += dist_loss.item()
        mean_rank_loss += rank_loss.item()
        mean_decorr_loss += decorr_loss.item()

        loss.backward()
        optimizer.step()

    mean_dist_loss /= len(dataloader)
    mean_rank_loss /= len(dataloader)
    mean_decorr_loss /= len(dataloader)

    learning_rate = optimizer.param_groups[-1]['lr']

    tl = tl.item()
    ta = ta.item()

    logger.info("FINETUNING Epoch: {}/{} \tLoss: {:.4f} \tRankLoss: {:.4f} \tDistLoss: {:.4f} \tDecorrLoss: {:.6f} \tLearningRate {:.6f} \tTrain Acc: {:.4f} ".format(epoch, args.epochs, tl, mean_rank_loss, mean_dist_loss, mean_decorr_loss, learning_rate, ta))
    print("FINETUNING Epoch: {}/{} \tLoss: {:.4f} \tRankLoss: {:.4f} \tDistLoss: {:.4f} \tDecorrLoss: {:.6f} \tLearningRate {:.6f} \tTrain Acc: {:.4f} ".format(epoch, args.epochs, tl, mean_rank_loss, mean_dist_loss, mean_decorr_loss, learning_rate, ta))

    return ta, tl


########### TEST FUNC ###########
def test(model, args, criterion, dataloader, logger, label_token, epoch=-1):
    logger.info("-----------------------EVALUATION MODE-----------------------")
    print("-----------------------EVALUATION MODE-----------------------")

    predRST = []
    labelRET = []
    legacy_batch_acc = Averager()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(tqdm(dataloader)):

            _, dist_feat = model.encode_img(features.cuda())

            label_emb = model.forward_text(label_token.cuda())

            logits = dist_feat @ label_emb.t()
            logits = logits / logits.norm(dim=-1, keepdim=True)

            labels_cuda = labels.cuda()
            loss = criterion(logits, labels_cuda)
            batch_size = int(labels.numel())
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

            acc = compute_ACC(logits, labels_cuda)
            legacy_batch_acc.add(acc)

            preds_np = np.array(torch.argmax(logits, dim=1).cpu())
            labels_np = np.array(labels)
            predRST.append(preds_np)
            labelRET.append(labels_np)
            # 转换成一维数组
        predRST = np.concatenate(predRST, axis=0)
        labelRET = np.concatenate(labelRET, axis=0)
        # 保存成mat文件, predRST为a列, labelRET为b列
        # scipy.io.savemat('predRST_first.mat', {'a': predRST, 'b': labelRET})

        logger.info("completed calculating predictions over all images")

    if sample_count == 0:
        raise ValueError("Evaluation dataloader produced no samples")

    num_classes = int(args.classes)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labelRET, predRST), 1)
    support = confusion.sum(axis=1)
    per_class_recall = np.divide(
        np.diag(confusion).astype(np.float64),
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support != 0,
    )
    sample_acc = float((predRST == labelRET).sum() / sample_count)
    macro_recall = float(per_class_recall.mean())
    mean_loss = float(loss_sum / sample_count)
    legacy_acc = float(legacy_batch_acc.item())
    label_names = list(getattr(args, "label_nms", [str(i) for i in range(num_classes)]))

    eval_metrics = {
        "epoch_zero_based": int(epoch),
        "num_samples": int(sample_count),
        "sample_val_acc": sample_acc,
        "legacy_val_acc": legacy_acc,
        "val_loss": mean_loss,
        "macro_recall": macro_recall,
        "per_class_recall": {
            name: float(value) for name, value in zip(label_names, per_class_recall)
        },
        "confusion_matrix": confusion.tolist(),
    }
    args.last_eval_metrics = eval_metrics

    metrics_text = json.dumps(eval_metrics, ensure_ascii=False, sort_keys=True)
    logger.info("Evaluation metrics: %s", metrics_text)
    print("Evaluation metrics: {}".format(metrics_text))
    logger.info(
        "Test Loss: {:.4f} \t Sample Test ACC: {:.4f} \t Legacy Batch ACC: {:.4f} \t Macro Recall: {:.4f}".format(
            mean_loss, sample_acc, legacy_acc, macro_recall
        )
    )
    print(
        "Test Loss: {:.4f} \t Sample Test ACC: {:.4f} \t Legacy Batch ACC: {:.4f} \t Macro Recall: {:.4f}".format(
            mean_loss, sample_acc, legacy_acc, macro_recall
        )
    )

    record_path = getattr(args, "record_path", None)
    if record_path:
        with open(os.path.join(record_path, "latest_eval_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(eval_metrics, handle, indent=2, ensure_ascii=False)
        with open(os.path.join(record_path, "eval_history.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    return sample_acc, mean_loss

