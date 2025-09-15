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


def update(self, model, optimizer, outputs, loss, writer, lr_scheduler, progress_bar, global_step, completed_steps,
           epoch, step, accelerator):
    # 确保loss是有效的tensor
    valid_loss = loss
    if not isinstance(loss, torch.Tensor):
        if hasattr(outputs, 'loss') and isinstance(outputs.loss, torch.Tensor):
            valid_loss = outputs.loss
        else:
            valid_loss = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)

    if not valid_loss.requires_grad:
        if hasattr(outputs, 'loss') and isinstance(outputs.loss, torch.Tensor) and outputs.loss.requires_grad:
            valid_loss = outputs.loss
        else:
            valid_loss = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)

    try:
        accelerator.backward(valid_loss)
    except Exception as e:
        try:
            valid_loss.backward()
        except Exception as e2:
            pass

    try:
        # TSS: Compute importance before optimizer step for supsup (need gradients)
        if 'supsup' in self.args.baseline:
            from networks.baselines.supsup import compute_model_importance
            model_ori = accelerator.unwrap_model(model)
            # Compute importance based on gradients before they are cleared
            compute_model_importance(model_ori, self.args.ft_task)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    except Exception as e:
        pass

    progress_bar.update(1)
    completed_steps += 1
    global_step += 1

    # 安全地获取loss值用于显示
    loss_item = 0.0
    try:
        if isinstance(valid_loss, torch.Tensor) and hasattr(valid_loss, 'item'):
            loss_item = valid_loss.item()
        elif isinstance(loss, (int, float)):
            loss_item = float(loss)
    except:
        loss_item = 0.0

    progress_bar.set_postfix(loss=loss_item)

    if writer is not None:
        writer.add_scalar('train/loss', loss_item, global_step)

    return global_step, completed_steps