from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np

def evaluate_model(model, val_gen, class_names):
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    
    acc = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # multi-class AUC
    try:
        auc = roc_auc_score(val_gen.classes, preds, multi_class='ovr')
    except:
        auc = None  # if AUC cannot be calculated

    report = classification_report(y_true, y_pred, target_names=class_names)

    return acc, auc, precision, recall, f1, report
