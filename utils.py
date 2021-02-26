import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from model import get_y_init_given_threshold,ZeroerModel

DEL = 1e-300


def get_results(true_labels, predicted_labels):
    p = precision_score(true_labels, predicted_labels)
    r = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return p, r, f1


def run_zeroer(similarity_features_df, similarity_features_lr,id_dfs,true_labels,LR_dup_free,run_trans):
    similarity_matrix = similarity_features_df.values
    y_init = get_y_init_given_threshold(similarity_features_df)
    similarity_matrixs = [similarity_matrix,None,None]
    y_inits = [y_init,None,None]
    if similarity_features_lr[0] is not None:
        similarity_matrixs[1] = similarity_features_lr[0].values
        similarity_matrixs[2] = similarity_features_lr[1].values
        y_inits[1] = get_y_init_given_threshold(similarity_features_lr[0])
        y_inits[2] = get_y_init_given_threshold(similarity_features_lr[1])
    feature_names = similarity_features_df.columns

    c_bay = 0.1
    model, y_pred = ZeroerModel.run_em(similarity_matrixs, feature_names, y_inits,id_dfs,LR_dup_free,run_trans, y_true=true_labels,
                                       hard=False, c_bay=c_bay)
    if true_labels is not None:
        p, r, f1 = get_results(true_labels, np.round(np.clip(y_pred + DEL, 0., 1.)).astype(int))
        print("Results after EM:")
        print("F1: {:0.2f}, Precision: {:0.2f}, Recall: {:0.2f}".format(f1, p, r))
    return y_pred
