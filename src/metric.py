import numpy as np


def probabilistic_f1(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-10)
    c_recall = ctp / (y_true_count + 1e-10)
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def probabilistic_f1_oversample(labels, predictions):
    true = np.array(labels)
    pred = np.array(predictions)
    _, counts = np.unique(true, return_counts=True)

    diff = counts[0] - counts[1]
    quotient = diff // counts[1]
    remainder = diff % counts[1]

    temp = pred[np.where(true==1.)]
    new_true = true.tolist() + [1] * diff
    new_pred = pred.tolist() + temp.tolist() * quotient + temp.tolist()[:remainder]

    new_true = np.array(new_true)
    new_pred = np.array(new_pred)

    return probabilistic_f1(labels=new_true, predictions=new_pred)


if __name__ == '__main__':
    preds = [1, 0, 0]
    labels = [0, 0, 1]

    print(probabilistic_f1_oversample(labels, preds))

