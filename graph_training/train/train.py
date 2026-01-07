from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
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
    
    for batch_id, data_dict in tqdm(enumerate(data_loader), desc="Processing train dataloader", total = len(data_loader)): 
        targets = data_dict["activity_label"].to(device)
        graphs = data_dict["full_action_graphs"]
        optimizer.zero_grad()
        output = net(graphs)
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
    num_classes_train,
    num_classes_val,
):
    global_step, train_epoch_result = train(
        net,
        opt,
        train_loader,
        device,
        global_step=global_step,
        num_classes=num_classes_train,
    )

    opt.zero_grad()

    val_epoch_result = evaluate(
        net, 
        validate_loader, 
        device,
        num_classes=num_classes_val,
    )
    
    epoch_result = {}
    epoch_result["train"] = train_epoch_result
    epoch_result["val"] = val_epoch_result
    return epoch_result, global_step