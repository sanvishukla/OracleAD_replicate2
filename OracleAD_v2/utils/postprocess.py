import numpy as np

def postprocess_predictions(scores, threshold,
                            min_len=5,
                            gap=3,
                            dilation=3):
    preds = (scores >= threshold).astype(int)

    # light dilation
    for i in range(len(preds)):
        if preds[i] == 1:
            preds[max(0, i-dilation):min(len(preds), i+dilation+1)] = 1

    # merge small gaps
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            k = j
            while k < len(preds) and preds[k] == 0 and k-j <= gap:
                k += 1
            if k < len(preds) and preds[k] == 1:
                preds[j:k] = 1
            i = k
        else:
            i += 1

    # remove very short segments
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            if j - i < min_len:
                preds[i:j] = 0
            i = j
        else:
            i += 1

    return preds
