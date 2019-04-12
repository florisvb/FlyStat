from random import shuffle
import copy
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def get_resampled_pairs(data1, data2, shuffle_labels=False, randomize=True):
    assert len(data1) == len(data2)

    resampled_data1 = []
    resampled_data2 = []

    for d in range(len(data1)):
        if randomize:
            idx = np.random.randint(0, len(data1[d]), len(data1[d]))
        else:
            idx = np.arange(0, len(data1[d]))
        d1 = [data1[d][i] for i in idx]

        if randomize:
            idx = np.random.randint(0, len(data2[d]), len(data2[d]))
        else:
            idx = np.arange(0, len(data2[d]))
        d2 = [data2[d][i] for i in idx]

        if shuffle_labels:
            merge = copy.copy(d1)
            merge.extend(d2)
            shuffle(merge)
            d1 = merge[0:len(d1)]
            d2 = merge[len(d1):]
            #for i in range(len(d1)):
            #    if np.random.random() > 0.5: # swap labels
            #        d1i = d1[i]
            #        d2i = d2[i]
            #        d2[i] = d1i
            #        d1[i] = d2i

        resampled_data1.append(d1)
        resampled_data2.append(d2)

    return resampled_data1, resampled_data2

def get_resampled_pair_difference(data1, data2, shuffle_labels=False, randomize=True):
    resampled_data1, resampled_data2 = get_resampled_pairs(data1, data2, shuffle_labels, randomize)
    differences = []
    for d in range(len(resampled_data1)):
        differences.append( np.mean( np.array(resampled_data1[d]) ) - np.mean( np.array(resampled_data2[d]) ) )





    return np.mean(differences)

def get_pvalue_for_resampled_pair_difference(data1, data2, iterations=10000):
    shuffled = Parallel(n_jobs=num_cores)(delayed(get_resampled_pair_difference)(data1, data2, shuffle_labels=True) for i in range(iterations) )
    actual = get_resampled_pair_difference(data1, data2, shuffle_labels=False, randomize=False)

    if actual < 0:
        idx = np.where(np.array(shuffled)<actual)[0]
        return len(idx) / float(iterations)
    else:
        idx = np.where(np.array(shuffled)>actual)[0]
        return len(idx) / float(iterations)

#def get_resampled_paired_pval(data1, data2):

