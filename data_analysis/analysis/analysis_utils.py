import numpy as np


def percent_diff(v1, v2):
    return np.round(np.abs(v1-v2)/((v1+v2)/2), 3)


def fix_losses_bug(df, acc_series_name, loss_series_name):
    acc_series = df[acc_series_name]
    loss_series = df[loss_series_name]
    for i, loss_val in enumerate(loss_series):
        if loss_val > 10:#(a bug in loss caused weird loss values (higher than usual - so adjusting the value according to the accuracy improvement)
            if i == 0: # if this is the first cell and needs to be fixed, we will compare to next gen
                if acc_series.values[i+1] >= acc_series.values[i]:# we will improve next gen we are sucks now
                    loss_series.values[i] = loss_series.values[i+1] + loss_series.values[i+1] * percent_diff(acc_series.values[i+1], acc_series.values[i])
                else:
                     loss_series.values[i] = loss_series.values[i+1] - loss_series.values[i+1] * percent_diff(acc_series.values[i+1], acc_series.values[i])
            else:# is this is not the first cell, we can compare to prev gen
                if acc_series.values[i] >= acc_series.values[i-1]: # we have improved in acc
                    loss_series.values[i] =  loss_series.values[i-1] - loss_series.values[i-1] * percent_diff(acc_series.values[i], acc_series.values[i-1]) # so our loss is better too
                else:# we don't have an improvement in acc
                    loss_series.values[i] =  loss_series.values[i-1] + loss_series.values[i-1] * percent_diff(acc_series.values[i], acc_series.values[i-1])# so out loss is sucks too
    return df