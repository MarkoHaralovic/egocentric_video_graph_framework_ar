from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.utils import logits_to_probs
from .evaluate import evaluate, evaluation_metrics

def train(
    net,
    optimizer,
    data_loader,
    device,
    global_step,
    num_classes,
    text_features = False
):
    net.train()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    for batch_id, data_dict in tqdm(enumerate(data_loader), desc="Processing train dataloader"):
        image_embeddings = data_dict["visual_features"].to(device)
        
        if text_features and "text_features" in data_dict:
            text_embeddings = data_dict["text_features"].to(device)
        else:
            text_embeddings = None
            
        targets = data_dict["activity_label"].to(device)
    
        optimizer.zero_grad()
        output = net(x_img=image_embeddings, x_text=text_embeddings)
        loss = F.cross_entropy(output, targets, weight=None)
            
        logits = output.data
        pred = logits.max(1)[1]
        
        y_pred_np = pred.cpu().numpy()
        y_true_np = targets.cpu().numpy()
        
        all_preds.extend(y_pred_np)
        all_targets.extend(y_true_np)

        loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += loss.item()
    
    y_pred_all = np.array(all_preds)
    y_true_all = np.array(all_targets)
    
    eval_metrics, conf_mat = evaluation_metrics(y_pred_all, y_true_all, num_classes)
    epoch_result = {}
    epoch_result["eval_metrics"] = eval_metrics
    epoch_result["conf_mat"] = conf_mat
    
    avg_loss = total_loss / len(data_loader)
    print(f"Training average Loss: {avg_loss:.4f}")
    print(f"Train accuracy : {eval_metrics['acc']*100:.2f}%")
    print(f"Train f1 : {eval_metrics['f1']*100:.2f}%")
    
    return global_step, epoch_result
 
def do_epoch(
    device,
    net,
    opt,
    train_loader,
    validate_loader,
    global_step,
    num_classes,
    text_features=False
):
    global_step, train_epoch_result = train(
        net,
        opt,
        train_loader,
        device,
        global_step=global_step,
        num_classes=num_classes,
        text_features=text_features
    )

    opt.zero_grad()

    val_epoch_result = evaluate(
        net, 
        validate_loader, 
        device,
        num_classes=num_classes,
        text_features=text_features
    )
    
    epoch_result = {}
    epoch_result["train"] = train_epoch_result
    epoch_result["val"] = val_epoch_result
    return epoch_result, global_step