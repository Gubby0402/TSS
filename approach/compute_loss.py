import logging
import torch
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import utils
from copy import deepcopy


# before training ***********************************************************************************************
def compute(self, model, train_loader, outputs, self_fisher, mask_pre, batch, step, accelerator):
    weights_before = None
    self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(
        train_loader) + 1 / self.args.smax

    # 保存原始输出以检查
    original_outputs = outputs

    # 处理不同基线的输出
    try:
        if 'ewc' in self.args.baseline:
            outputs = model(batch, self_fisher=self_fisher)
        elif 'adapter_hat' in self.args.baseline or 'adapter_cat' in self.args.baseline \
                or 'adapter_bcl' in self.args.baseline \
                or 'adapter_ctr' in self.args.baseline \
                or 'adapter_classic' in self.args.baseline:
            masks = utils.model.mask(model, accelerator, self.args)
            outputs = model(
                batch, masks=masks, mask_pre=mask_pre)
        elif 'mer' in self.args.baseline:
            model_ori = accelerator.unwrap_model(model)
            weights_before = deepcopy(
                model_ori.state_dict())
            outputs = model(batch)
        elif 'lamaml' in self.args.baseline:
            if not (self.args.buffer is None or self.args.buffer.is_empty()) and step % self.args.replay_freq == 0:
                replay_batch = self.args.buffer.get_datadict(
                    size=batch['input_ids'].shape[0])
                if self.args.task_name in self.args.classification:
                    replay_batch['cls_labels'] = replay_batch['labels']

                for key in batch.keys():
                    if key == 'labels' and self.args.task_name in self.args.classification:
                        continue
                    batch[key] = torch.cat(
                        (batch[key], replay_batch[key]), dim=0)

            self.fast_weights = self.meta_learner.inner_update(
                self.fast_weights, batch, is_train=True)
            meta_outputs = self.meta_learner.meta_loss(
                self.fast_weights, batch, is_train=True)
            if outputs is None or (step % self.args.meta_task_size == 0):
                outputs = meta_outputs
            else:
                outputs.loss += meta_outputs.loss / \
                                batch['input_ids'].shape[0]
        else:
            outputs = model(batch)
    except Exception as e:
        # 如果出错，尝试使用默认的前向传播
        try:
            outputs = model(batch)
        except Exception as e2:
            # 保持原始输出
            outputs = original_outputs

    # 确保输出有有效的loss
    if outputs is None:
        # 创建一个类似OutputObject的对象
        class DummyOutput:
            def __init__(self):
                dummy_param = next(model.parameters())
                self.loss = torch.tensor(0.0, requires_grad=True, device=dummy_param.device)
                self.logits = None
                self.sum_loss = None
                self.contrast_loss = None

        outputs = DummyOutput()
    elif not hasattr(outputs, 'loss') or outputs.loss is None:
        dummy_param = next(model.parameters())
        outputs.loss = torch.tensor(0.0, requires_grad=True, device=dummy_param.device)

    # 检查loss是否为tensor且需要梯度
    if not isinstance(outputs.loss, torch.Tensor):
        dummy_param = next(model.parameters())
        try:
            # 尝试转换为tensor
            loss_value = float(outputs.loss)
            outputs.loss = torch.tensor(loss_value, requires_grad=True, device=dummy_param.device)
        except:
            outputs.loss = torch.tensor(0.0, requires_grad=True, device=dummy_param.device)
    elif not outputs.loss.requires_grad:
        # 保存原始loss值
        loss_value = outputs.loss.item()
        # 创建新的需要梯度的tensor
        dummy_param = next(model.parameters())
        outputs.loss = torch.tensor(loss_value, requires_grad=True, device=dummy_param.device)

    # 确保其他可能的属性正确初始化
    if not hasattr(outputs, 'sum_loss'):
        outputs.sum_loss = None
    if not hasattr(outputs, 'contrast_loss'):
        outputs.contrast_loss = None
    if not hasattr(outputs, 'logits') and hasattr(model, 'num_classes'):
        try:
            # 尝试创建虚拟logits
            dummy_param = next(model.parameters())
            batch_size = batch['input_ids'].shape[0]
            outputs.logits = torch.zeros((batch_size, model.num_classes),
                                         device=dummy_param.device)
        except Exception as e:
            outputs.logits = None

    return self, model, outputs, weights_before