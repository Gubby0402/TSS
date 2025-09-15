from networks.baselines import supsup
from networks.baselines import ewc, hat
import torch


def run_forward(input_ids, attention_mask, task, cls_labels, my_model, self_fisher, masks=None, mask_pre=None,
                inputs_embeds=None,
                head_mask=None,
                only_return_output=False,
                ):
    hidden_states = None
    loss = None
    logits = None

    if 'supsup' in my_model.args.baseline:
        if 'mtl' in my_model.args.baseline:  # these are only useful for supsup
            supsup.set_model_sim(my_model.model, 'both')  # if nothing
            supsup.set_model_specific_task(my_model, task)  # in case nothing is used
            supsup.set_model_share_task(my_model, 0)  # alwasy use the same, as shared knwoeldeg accorss all
        elif 'ncl' in my_model.args.baseline:
            supsup.set_model_sim(my_model.model, 'specific')  # if nothing
            supsup.set_model_specific_task(my_model, 0)  # alwasys use the same

        else:
            supsup.set_model_sim(my_model.model, 'specific')  # if nothing

            if 'forward' in my_model.args.baseline:
                task_dup = task.repeat(2)
                supsup.set_model_specific_task(my_model, task_dup)  # in case nothing is used
            else:
                supsup.set_model_specific_task(my_model, task)  # in case nothing is used

        # 重要：在supsup分支中也需要调用模型的前向传播！
        if my_model.args.is_reference:
            outputs = my_model.teacher(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=cls_labels,
                                       attention_mask=attention_mask,
                                       head_mask=head_mask,
                                       output_hidden_states=True, task=task, only_return_output=only_return_output)
        else:
            outputs = my_model.model(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=cls_labels,
                                     attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     output_hidden_states=True, task=task, only_return_output=only_return_output
                                     )

        if only_return_output:
            hidden_states = outputs.hidden_states
        else:
            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.hidden_states

    else:
        if my_model.args.is_reference:
            outputs = my_model.teacher(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=cls_labels,
                                       attention_mask=attention_mask,
                                       head_mask=head_mask,
                                       output_hidden_states=True, task=task, only_return_output=only_return_output)
        else:
            outputs = my_model.model(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=cls_labels,
                                     attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     output_hidden_states=True, task=task, only_return_output=only_return_output
                                     )

        if only_return_output:
            hidden_states = outputs.hidden_states
        else:
            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.hidden_states

    # 确保loss是一个需要梯度的张量
    if loss is None:
        dummy_param = next(my_model.model.parameters())
        loss = torch.tensor(0.0, requires_grad=True, device=dummy_param.device)
    elif isinstance(loss, (int, float)) and loss == 0:
        dummy_param = next(my_model.model.parameters())
        loss = torch.tensor(float(loss), requires_grad=True, device=dummy_param.device)
    elif isinstance(loss, torch.Tensor) and not loss.requires_grad:
        loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
        dummy_param = next(my_model.model.parameters())
        loss = torch.tensor(loss_value, requires_grad=True, device=dummy_param.device)

    if 'ewc' in my_model.args.baseline and my_model.training and self_fisher is not None:  # only if we are training
        ewc_loss = ewc.loss_compute(my_model, self_fisher)
        if loss is None:
            loss = ewc_loss
        else:
            loss += ewc_loss

    elif ('adapter_hat' in my_model.args.baseline or 'adapter_cat' in my_model.args.baseline
          or 'adapter_bcl' in my_model.args.baseline
          or 'adapter_ctr' in my_model.args.baseline
          or 'adapter_classic' in my_model.args.baseline) and my_model.training and not my_model.args.is_cat:  # no need for testing
        hat_loss = hat.loss_compute(masks, mask_pre, my_model.args)
        if loss is None:
            loss = hat_loss
        else:
            loss += hat_loss

    return loss, logits, hidden_states

