import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score, precision_recall_curve,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def classify_task_metric(probabilities_tensor,groud_truth_tensor,cfg,dataset_tasks,test_set_indices,smiles_list):
    pd_data = {'Sample ID': test_set_indices, 'SMILES': smiles_list}

    if cfg.train.Ablation.is_ablation == False:
        model_name = "{}_{}".format(cfg.Expert.model,cfg.LLMs.model)
    elif cfg.train.Ablation.is_ablation == True:
        model_name = "{}".format(cfg.train.Ablation.experiment_model)
    
    metric_save_root_path = './tmp/{}/{}'.format(cfg.train.dataset,model_name)
    if os.path.exists(metric_save_root_path) == False:
        os.makedirs(metric_save_root_path)

            
    for task_index,task_name in enumerate(dataset_tasks):
        pred_probabilities = []
        pred_label = []
        target = []
        for pred_t, target_t in zip(probabilities_tensor,groud_truth_tensor):
        
            pred_probabilities.append(round(pred_t[task_index].item(),4))
            pred_label.append(1 if pred_t[task_index] > 0.5 else 0)
            target.append(int(target_t[task_index]))

        cm = confusion_matrix(target, pred_label)
        print("{}_{}_cm is \n".format(model_name,task_name),cm)

        #acc precision recall auc
        accuracy, precision, recall, f1_score = cal_metric(target, pred_label)
        print("{}_{}_accuracy is ".format(model_name,task_name),accuracy)
        print("{}_{}_precision is ".format(model_name,task_name),precision)
        print("{}_{}_recall is ".format(model_name,task_name),recall)
        print("{}_{}_f1_score is ".format(model_name,task_name),f1_score)
        
        


        #保存混淆矩阵
        plot_confusion_matrix(target, pred_label, ["0", "1"], "{} confusion matrix".format(task_name),"./tmp/{}/{}/{}_{}_cm.png".format(cfg.train.dataset,model_name,model_name,task_name))
        
        # 绘制ROC曲线
        auroc = plot_roc(target,pred_probabilities, './tmp/{}/{}/{}_{}_roc.png'.format(cfg.train.dataset,model_name,model_name,task_name))
        print("{}_{}_auroc is ".format(model_name,task_name),round(auroc,4))
        
        # PRC曲线
        auprc = plot_prc(target, pred_probabilities, './tmp/{}/{}/{}_{}_prc.png'.format(cfg.train.dataset,model_name,model_name,task_name))
        print("{}_{}_auprc is ".format(model_name,task_name),round(auprc,4))

        pd_data.update({"{} Ground Truth".format(task_name):target, "{} Predicted Probability".format(task_name):pred_probabilities, "{} Predicted Label".format(task_name):pred_label})
    
    df = pd.DataFrame(pd_data)
    df.to_csv("./tmp/{}/{}/{}_{}_result.csv".format(cfg.train.dataset,model_name,model_name,task_name), index=False)

def regression_task_metric(target, pred_label):
    rmse = np.sqrt(mean_squared_error(target, pred_label))
    mae = mean_absolute_error(target, pred_label)
    r2 = r2_score(target, pred_label)
    print("Mean Absolute Error (MAE):", round(mae,4))
    print("Root Mean Squared Error (RMSE):", round(rmse,4))
    print("R-squared (R2):", round(r2,4))
    

def cal_metric(target, pred_label):
    accuracy = accuracy_score(target, pred_label)
    
    precision = precision_score(target, pred_label)
    
    recall = recall_score(target, pred_label)

    f1 = f1_score(target, pred_label)
    return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)
    

def plot_confusion_matrix(target, pred_label, labels_name, title,save_png_path, colorbar=False, cmap='Greens'):
    cm = confusion_matrix(target, pred_label)
    plt.figure()  # 创建一个新的图形对象
    plt.imshow(cm, interpolation='nearest', cmap=cmap)    # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.title(title)    # 图像标题
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')

    plt.savefig("{}".format(save_png_path))
    plt.show()

def plot_roc(target,pred_probabilities, save_png_path):
    fpr, tpr, _ = roc_curve(target, pred_probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("{}".format(save_png_path))
    plt.show()
    return roc_auc

def plot_prc(target, pred_probabilities, save_png_path):
    precision, recall, _ = precision_recall_curve(target, pred_probabilities)
    prc_auc = auc(recall, precision)
    # 绘制PRC曲线
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (AUC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig("{}".format(save_png_path))
    plt.show()
    return prc_auc


     
