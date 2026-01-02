import os
import numpy as np
from sklearn import metrics
import torch
from tqdm import tqdm

def evaluate(net, data_loader, device, num_classes, text_features = False):
    net.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for _, data_dict in tqdm(enumerate(data_loader), total = len(data_loader), desc = "Evaluating"):
            image_embeddings = data_dict["visual_features"].to(device)
            if image_embeddings.dim() == 3:
                # issue with timesformere where data is [8,1,600] instead of [8,600]
                image_embeddings = image_embeddings.squeeze(1)
            if text_features and "text_features" in data_dict:
                text_embeddings = data_dict["text_features"].to(device)
            else:
                text_embeddings = None
                
            targets = data_dict["activity_label"].to(device)
            output = net(x_img=image_embeddings, x_text=text_embeddings)

            logits = output.data
            pred = logits.max(1)[1]
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    y_pred_np = np.array(all_preds)
    y_true_np = np.array(all_targets)

    eval_metrics, conf_mat = evaluation_metrics(y_pred_np, y_true_np, num_classes)
    epoch_result = {}
    epoch_result["eval_metrics"] = eval_metrics
    epoch_result["conf_mat"] = conf_mat
    
    return epoch_result

def evaluation_metrics(y_pred, y_true, num_classes):

   confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=tuple(range(num_classes)))

   results = {
      'acc': metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
      'f1': metrics.f1_score(y_true=y_true, y_pred=y_pred,average="macro"),
   }

   return results, confusion_matrix

def store_model(
    net, opt, epoch, save_path, metric="f1"
):
    for f in os.listdir(save_path):
        if f.startswith(f"best_train_model_{metric}_") and f.endswith(".pt"):
            os.remove(os.path.join(save_path, f))
    
    file_name = f"best_model_{metric}_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "metric": metric,
        },
        os.path.join(save_path, file_name),
    )
    