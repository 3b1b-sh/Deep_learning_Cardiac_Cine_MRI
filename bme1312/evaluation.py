
import torch
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,jaccard_score


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    # SR = SR > threshold
    # GT = GT == torch.max(GT)
    # corr = torch.sum(SR == GT)
    # tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    # acc = float(corr) / float(tensor_size)
    acc=accuracy_score(GT.cpu().flatten().numpy(),SR.cpu().flatten().numpy()>threshold)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # # TP : True Positive
    # # FN : False Negative
    # TP = ((SR == 1) & (GT == 1))
    # FN = ((SR == 0) & (GT == 1))

    # print("torch.sum(TP)",torch.sum(TP))
    # print("torch.sum(TP + FN)",torch.sum(TP + FN))

    # SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    SE=precision_recall_fscore_support(GT.cpu().flatten().numpy(),SR.cpu().flatten().numpy()>threshold)[1][1]
    return SE


def get_specificity(SR, GT, threshold=0.5):
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # # TN : True Negative
    # # FP : False Positive
    # TN = ((SR == 0) & (GT == 0))
    # FP = ((SR == 1) & (GT == 0))

    # SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    # print("------------------specificity---------------")
    # print("TN:",torch.sum(TN))
    # print("FP:",torch.sum(FP))
    # print("SP:",SP)
    SP=precision_recall_fscore_support(GT.cpu().flatten().numpy(),SR.cpu().flatten().numpy()>threshold)[1][0]
    return SP


def get_precision(SR, GT, threshold=0.5):
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # # TP : True Positive
    # # FP : False Positive
    # TP = ((SR == 1) & (GT == 1))
    # FP = ((SR == 1) & (GT == 0))

    # PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    # print("------------------precision---------------")
    # print("TP:", torch.sum(TP))
    # print("FP:", torch.sum(FP))
    # print("PC:", PC)
    PC=precision_recall_fscore_support(GT.cpu().flatten().numpy(),SR.cpu().flatten().numpy()>threshold)[0][1]
    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    # SE = get_sensitivity(SR, GT, threshold=threshold)
    # PC = get_precision(SR, GT, threshold=threshold)

    # F1 = 2 * SE * PC / (SE + PC + 1e-6)
    F1 = precision_recall_fscore_support(GT.cpu().flatten().numpy(),SR.cpu().flatten().numpy()>threshold)[2][1]
    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    # SR = SR > threshold
    # GT = GT == torch.max(GT)

    # Inter = torch.sum(((SR == 1) & (GT == 1)))
    # Union = torch.sum((SR + GT) >= 1)

    # JS = float(Inter) / (float(Union) + 1e-6)
    JS = jaccard_score(GT.cpu().flatten().numpy(),(SR.cpu().flatten().numpy()>threshold))
    return JS


# def get_DC(SR, GT, threshold=0.5):
#     # DC : Dice Coefficient
#     # SR = SR > threshold
#     # GT = GT == torch.max(GT)


#     # Inter = torch.sum(((SR==1) & (GT==1)))
#     # DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)
#     DC = precision_recall_fscore_support(GT.cpu().flatten().numpy(),SR.cpu().flatten().numpy()>threshold)[2][1]
#     return DC

def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    # SR = SR > threshold
    # GT = GT == torch.max(GT)
    # Inter = torch.sum(((SR==1) & (GT==1)))
    # DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)
    DC = precision_recall_fscore_support(GT.cpu().detach().numpy().flatten(), SR.cpu().detach().numpy().flatten() > threshold)[2][1]
    return DC