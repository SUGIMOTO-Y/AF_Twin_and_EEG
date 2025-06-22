import torch
from tqdm import tqdm
from utilities.CostumScheduler import *

def Finetuning(args, dataset, model):
    t_total = len(dataset) // args.Train.gradient_accumulation_steps *args.Train.EPOCHS
    num_warmup_steps = int(0.20 * t_total)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.Train.lr, weight_decay= args.Train.weight_decay, eps=args.Train.eps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)
    model = model.to(torch.device(args.device))
    for epoch in range(int(args.Train.EPOCHS+1)):
        SetLog = '[Epoch{:3d}] Train | loss:{:.7f} | TS:{:.4f}'\
            .format(epoch + 1, loss_/len(dataset.dataset), logit_scale) \
                if epoch != 0 else '[Epoch{:3d}]'.format(epoch+1) 
        loss_ = 0
        with tqdm(dataset, leave=True) as pbar_ite:
            pbar_ite.set_description(SetLog)
            for i, (sig, img, lab) in enumerate(pbar_ite):
                model.train()
                IMG_features, EEG_features= model(img, sig, is_step = 1)
                IMG_features = IMG_features / IMG_features.norm(dim=-1, keepdim=True)
                EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * IMG_features @ EEG_features.t()
                logits_per_text = logit_scale * EEG_features @ IMG_features.t()
                labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
                image_loss = torch.functional.F.cross_entropy(logits_per_image, labels)
                eeg_loss  = torch.functional.F.cross_entropy(logits_per_text, labels)
                loss = (image_loss + eeg_loss) / 2
                loss.backward()
                if (i + 1) % args.Train.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                model.zero_grad()
                loss_ += loss.item()
                pbar_ite.update(1)
    return model
            