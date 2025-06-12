import torch

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def CalculateAcc(args, dataset, model, img):
    MaxEpoch = 999999
    img = torch.Tensor(img)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (sig, lab) in enumerate(dataset):
            if i < MaxEpoch:
                model.eval()
                sig = sig.to(torch.device(args.device))
                lab = lab.to(torch.device(args.device))
                EEG_features = model.EEGEncoder(sig).float()
                EEG_features /= EEG_features.norm(dim=-1, keepdim=True)
                img = img.to(torch.device(args.device))[:200,:]
                IMG_features = model.IMGEncoder(img).float()
                IMG_features /= IMG_features.norm(dim=-1, keepdim=True)
                IMG_probs = 100.0 * EEG_features @ IMG_features.T
                acc1, acc5 = accuracy(IMG_probs, lab, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += sig.size(0)
                pred = IMG_probs.data.max(1, keepdim=True)[1]
                pred = pred.cpu().detach().numpy()
                lab = lab.cpu().detach().numpy()
            else: break
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return (top1, top5)
