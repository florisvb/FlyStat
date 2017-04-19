import numpy as np
import networkx # sudo apt-get install python-networkx
import community # https://bitbucket.org/taynaud/python-louvain
from collections import defaultdict
from matplotlib import pyplot, patches
import scipy.stats
import copy
import fly_plot_lib.plot as fpl
import matplotlib.pyplot as plt

def bootstrap_confidence_for_lines(lines, use='median', iterations=1000):
    if use == 'median':
        np.use = np.median
    elif use == 'mean':
        np.use = np.mean
    
    test_lines = []
    for n in range(iterations):
        indices = np.random.randint(0,len(lines),len(lines))
        test_lines.append(np.use(lines[indices,:], axis=0))
        
    test_lines = np.vstack(test_lines)
    test_lines.sort(axis=0)
    
    confidence_lo=0.025
    confidence_hi=0.975
    n = iterations
    hi_index = np.min([int(confidence_hi*n), n-1])
    lo_index = int(confidence_lo*n)
    
    line_lo = test_lines[lo_index,:]
    line_hi = test_lines[hi_index,:]

    return line_lo, line_hi

def bootstrap_medians_from_data(data, use='median', iterations=1000):
    medians = []
    for n in range(iterations):
        indices = np.random.randint(0,len(data),len(data))
        if use == 'median':
            medians.append(np.median(data[indices]))
        elif use == 'mean':
            medians.append(np.mean(data[indices]))
    medians.sort()
    return np.array(medians)
    
def bootstrap_medians_and_std_from_data(data, use='median', iterations=1000):
    medians = []
    stds = []
    for n in range(iterations):
        indices = np.random.randint(0,len(data),len(data))
        if use == 'median':
            medians.append(np.median(data[indices]))
        elif use == 'mean':
            medians.append(np.mean(data[indices]))
        stds.append(np.std(data[indices]))
    isort = np.array(medians).argsort()
    stds = np.array(stds)[isort]
    medians = np.array(medians)[isort]
    return medians, stds
    
def bootstrap_std_confidence_intervals(data, iterations=1000):
    medians, stds = bootstrap_medians_and_std_from_data(data, use='median', iterations=iterations)
    n = len(medians)
    confidence_lo=0.025
    confidence_hi=0.975
    hi_index = np.min([int(confidence_hi*n), n-1])
    lo_index = int(confidence_lo*n)
    return stds[lo_index], stds[hi_index]
    
    
def bootstrap_confidence_intervals_for_medians(medians, confidence_lo=0.025, confidence_hi=0.975):
    confidence_intervals = []
    medians.sort()
    n = len(medians)
    hi_index = np.min([int(confidence_hi*n), n-1])
    lo_index = int(confidence_lo*n)
    conf_interval = [medians[lo_index], medians[hi_index]]
    return conf_interval
    
def bootstrap_confidence_intervals_from_data(data, iterations=1000, use='median', confidence_lo=0.025, confidence_hi=0.975):
    medians = bootstrap_medians_from_data(data, use=use)
    conf_interval = bootstrap_confidence_intervals_for_medians(medians, confidence_lo, confidence_hi)
    return conf_interval
    
def probability_we_assign_correct_distribution_to_random_data_point_grouping(data):
    '''data is a list of datasets'''
    medians = np.array([np.median(d) for d in data])
    def assign_distribution_based_on_closest_distance_to_median(d):
        return np.argmin( np.abs(d-medians) )
    
    assignments = {}
    assignments_probabilities = np.zeros([len(data), len(data)])
    
    for dataset_number, dataset in enumerate(data):
        assignments.setdefault(dataset_number, [])
        for datapoint in dataset:
            assigned_distribution = assign_distribution_based_on_closest_distance_to_median(datapoint)
            assignments[dataset_number].append(assigned_distribution)
        assignments[dataset_number].sort()
        assignments[dataset_number] = np.array(assignments[dataset_number])
        assignment_probabilities = []
        for a in range(len(data)):
            na = len(np.where(assignments[dataset_number]==a)[0])
            p = na / float(len(assignments[dataset_number]))
            assignment_probabilities.append(p)
        assignments_probabilities[dataset_number, : ] = np.array(assignment_probabilities)
    return assignments_probabilities

def calculate_louvain_communities(assignment_matrix, node_order=None):
    # Calculate louvain communities
    G = networkx.to_networkx_graph(assignment_matrix, create_using=networkx.Graph())  
    louvain_community_dict = community.best_partition(G)
    # Convert community assignmet dict into list of communities
    louvain_comms = {}
    for node_index, comm_id in louvain_community_dict.iteritems():
        if comm_id in louvain_comms.keys():
            louvain_comms[comm_id].append(node_index)
        else:
            louvain_comms.setdefault(comm_id, [node_index])
    nodes_louvain_ordered = [node for comm in louvain_comms.values() for node in comm]
    # reorder original matrix according to node order
    adjacency_matrix = np.zeros_like(assignment_matrix)
    for i in range(assignment_matrix.shape[0]):
        for j in range(assignment_matrix.shape[0]):
            r = nodes_louvain_ordered.index(i)
            c = nodes_louvain_ordered.index(j)
            adjacency_matrix[r,c] = assignment_matrix[i,j] 
    return louvain_community_dict, nodes_louvain_ordered, adjacency_matrix
    
def recalculate_probabilities_after_merging_groups(data, louvain_community_dict):
    # Convert community assignmet dict into list of communities
    louvain_comms = defaultdict(list)
    for node_index, comm_id in louvain_community_dict.iteritems():
        louvain_comms[comm_id].append(node_index)
    louvain_comms = louvain_comms.values()
    new_data = []
    for community in louvain_comms:
        dataset = []
        for c in community:
            dataset.extend(data[c])
        new_data.append(dataset)
    assignments = probability_we_assign_correct_distribution_to_random_data_point_grouping(new_data)
    return assignments
    
def bootstrap_confidence_intervals_for_ratios(ratios, confidence_lo=0.025, confidence_hi=0.975):
    confidence_intervals = []
    medians = []
    for data in ratios:
        data.sort()
        n = len(data)
        medians.append(np.median(data))
        hi_index = int(confidence_hi*n)
        lo_index = int(confidence_lo*n)
        conf_interval = [data[lo_index], data[hi_index]]
        confidence_intervals.append(conf_interval)
    return medians, confidence_intervals      

def bootstrap_50_95_confidence_intervals(ratios):
    medians, confidence_intervals_95 = bootstrap_confidence_intervals_for_ratios(ratios, confidence_lo=0.025, confidence_hi=0.975)
    medians, confidence_intervals_50 = bootstrap_confidence_intervals_for_ratios(ratios, confidence_lo=0.25, confidence_hi=0.75)
    return medians, confidence_intervals_50, confidence_intervals_95
              
def calc_statistical_significance_through_resampling_no_power(data1, data2, iterations=1000, analysis='mean'):
    '''
    two tailed only
    '''
    if type(data1) is not list:
        data1 = data1.tolist()
    if type(data2) is not list:
        data2 = data2.tolist()

    if analysis == 'median':
        measured_difference = np.median(data1) - np.median(data2)
    elif analysis == 'mean':
        measured_difference = np.mean(data1) - np.mean(data2)
    data_mixture = []
    data_mixture.extend(data1)
    data_mixture.extend(data2)
    data_mixture = np.array(data_mixture)
    
    resampled_differences = []
    for i in range(iterations):
        indices1 = np.random.randint(0,len(data_mixture),len(data1))
        indices2 = np.random.randint(0,len(data_mixture),len(data2))
        
        sample_data1 = data_mixture[indices1]
        sample_data2 = data_mixture[indices2]
    
        if analysis == 'median':
            median1 = np.median(sample_data1)
            median2 = np.median(sample_data2)
            resampled_differences.append(median1-median2)
            
        elif analysis == 'mean':
            mean1 = np.mean(sample_data1)
            mean2 = np.mean(sample_data2)
            resampled_differences.append(mean1-mean2)
            
        
    resampled_differences.sort()
    index = np.argmin(np.abs(resampled_differences - measured_difference))
    raw_pval = index / float(len(resampled_differences))
    pval = 1 - 2*np.abs(raw_pval-.5)
    return pval
                
def calc_statistical_significance_through_resampling(data1, data2, iterations=1000, analysis='median', pval_for_power_calc=0.05, ax=None):
    '''
    two tailed only
    '''
    if type(data1) is not list:
        data1 = data1.tolist()
    if type(data2) is not list:
        data2 = data2.tolist()

    if analysis == 'median':
        measured_difference = np.median(data1) - np.median(data2)
    elif analysis == 'mean':
        measured_difference = np.mean(data1) - np.mean(data2)
    data_mixture = []
    data_mixture.extend(data1)
    data_mixture.extend(data2)
    data_mixture = np.array(data_mixture)
    
    resampled_differences = []
    for i in range(iterations):
        indices1 = np.random.randint(0,len(data_mixture),len(data1))
        indices2 = np.random.randint(0,len(data_mixture),len(data2))
        
        sample_data1 = data_mixture[indices1]
        sample_data2 = data_mixture[indices2]
    
        if analysis == 'median':
            median1 = np.median(sample_data1)
            median2 = np.median(sample_data2)
            resampled_differences.append(median1-median2)
            
        elif analysis == 'mean':
            mean1 = np.mean(sample_data1)
            mean2 = np.mean(sample_data2)
            resampled_differences.append(mean1-mean2)
            
        
    resampled_differences.sort()
    index = np.argmin(np.abs(resampled_differences - measured_difference))
    raw_pval = index / float(len(resampled_differences))
    pval = 1 - 2*np.abs(raw_pval-.5)
    
    
    
    resampled_data1_differences = []
    for i in range(iterations):
        indices1 = np.random.randint(0,len(data_mixture),len(data_mixture))
        indices2 = np.random.randint(0,len(data1),len(data1))
        
        sample_data1 = data_mixture[indices1]
        sample_data2 = np.array(data1)[indices2]
    
        if analysis == 'median':
            median1 = np.median(sample_data1)
            median2 = np.median(sample_data2)
            resampled_data1_differences.append(median1-median2)
            
        elif analysis == 'mean':
            mean1 = np.mean(sample_data1)
            mean2 = np.mean(sample_data2)
            resampled_data1_differences.append(mean1-mean2)
    
    resampled_data2_differences = []
    for i in range(iterations):
        indices1 = np.random.randint(0,len(data_mixture),len(data_mixture))
        indices2 = np.random.randint(0,len(data2),len(data2))
        
        sample_data1 = data_mixture[indices1]
        sample_data2 = np.array(data2)[indices2]
    
        if analysis == 'median':
            median1 = np.median(sample_data1)
            median2 = np.median(sample_data2)
            resampled_data2_differences.append(median1-median2)
            
        elif analysis == 'mean':
            mean1 = np.mean(sample_data1)
            mean2 = np.mean(sample_data2)
            resampled_data2_differences.append(mean1-mean2)
    
    
    i_p_lo = resampled_differences[ int(pval_for_power_calc/2.*len(resampled_differences)) ]
    i_p_hi = resampled_differences[ 1 - int(pval_for_power_calc/2.*len(resampled_differences)) ]
    
    data1_indices_above_p_hi = len(np.where(resampled_data1_differences > i_p_hi)[0])
    data1_indices_below_p_lo = len(np.where(resampled_data1_differences < i_p_lo)[0])
    data1_power = (data1_indices_above_p_hi + data1_indices_below_p_lo) / float(len( resampled_data1_differences ) )
    
    data2_indices_above_p_hi = len(np.where(resampled_data2_differences > i_p_hi)[0])
    data2_indices_below_p_lo = len(np.where(resampled_data2_differences < i_p_lo)[0])
    data2_power = (data2_indices_above_p_hi + data2_indices_below_p_lo) / float(len( resampled_data2_differences ) )
    
    if ax is not None:
        fpl.histogram(ax, [resampled_differences, resampled_data1_differences, resampled_data2_differences], bins=20, bin_width_ratio=0.6, colors=['black', 'red', 'blue'], edgecolor='none', bar_alpha=0.7, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3, 0.3], return_vals=False, show_smoothed=False, normed=True, peak_trace_alpha=0, show_peak_curve=True)
    
    return pval, data1_power, data2_power
                
                
def get_meandian_from_nested_data(data, levels=2, analysis='median'):
    assert levels == 2, "levels must be 2"
    if analysis == 'median':
        mdata = [np.median(l) for l in data]
        return np.median(mdata)
    elif analysis == 'mean':
        mdata = [np.mean(l) for l in data]
        return np.mean(mdata)

def get_nested_resampling(data_mixture, n, levels=2):
    assert levels == 2, "levels must be 2"
    
    indices_levels_0 = np.random.randint(0,len(data_mixture),n)
    sample_data = np.array(data_mixture)[indices_levels_0].tolist()
    
    indices_for_resampling = np.random.randint(0,len(sample_data),n)
    
    resampled = []
    for i in indices_for_resampling:
        d = sample_data[i]
        indices = np.random.randint(0,len(d),len(d))
        resampled.append(np.array(d)[indices])
        
    return resampled
    
def get_resampled_nested_distribution(data, iterations=1000):
    
    ms = []
    for i in range(iterations):
        resampled = get_nested_resampling(data, len(data))
        m = np.mean([np.mean(r) for r in resampled])
        ms.append(m)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(-3,3,30)
    ax.hist(ms, bins=bins)
    
    ax.set_xlim(-3,3)
    
    return ms
        
def calc_statistical_significance_through_resampling_nested(data1, data2, iterations=1000, analysis='median', pval_for_power_calc=0.05, levels=2):
    '''
    two tailed only
    '''
    assert levels == 2, "levels must be 2"
    

    measured_difference = get_meandian_from_nested_data(data1, levels, analysis) - get_meandian_from_nested_data(data2, levels, analysis)
    
    data_mixture = []
    data_mixture.extend(data1)
    data_mixture.extend(data2)
    
    resampled_differences = []
    for i in range(iterations):
        sample_data1 = get_nested_resampling(data_mixture, len(data1) )
        sample_data2 = get_nested_resampling(data_mixture, len(data2) )
        d = get_meandian_from_nested_data(sample_data1, levels, analysis) - get_meandian_from_nested_data(sample_data2, levels, analysis)
        resampled_differences.append(d)
    resampled_differences.sort()
    
    index = np.argmin(np.abs(resampled_differences - measured_difference))
    raw_pval = index / float(len(resampled_differences))
    pval = 1 - 2*np.abs(raw_pval-.5)
    
    resampled_data1_differences = []
    for i in range(iterations):
        sample_data1 = get_nested_resampling(data_mixture, len(data_mixture) )  
        sample_data2 = get_nested_resampling(data1, len(data1) )  
        d = get_meandian_from_nested_data(sample_data1, levels, analysis) - get_meandian_from_nested_data(sample_data2, levels, analysis)
        resampled_data1_differences.append(d)
    resampled_data1_differences.sort()
    
    resampled_data2_differences = []
    for i in range(iterations):
        sample_data1 = get_nested_resampling(data_mixture, len(data_mixture) )  
        sample_data2 = get_nested_resampling(data2, len(data2) )  
        d = get_meandian_from_nested_data(sample_data1, levels, analysis) - get_meandian_from_nested_data(sample_data2, levels, analysis)
        resampled_data2_differences.append(d)
    resampled_data2_differences.sort()
    
    i_p_lo = resampled_differences[ int(pval_for_power_calc/2.*len(resampled_differences)) ]
    i_p_hi = resampled_differences[ 1 - int(pval_for_power_calc/2.*len(resampled_differences)) ]
    
    data1_indices_above_p_hi = len(np.where(resampled_data1_differences > i_p_hi)[0])
    data1_indices_below_p_lo = len(np.where(resampled_data1_differences < i_p_lo)[0])
    data1_power = (data1_indices_above_p_hi + data1_indices_below_p_lo) / float(len( resampled_data1_differences ) )
    
    data2_indices_above_p_hi = len(np.where(resampled_data2_differences > i_p_hi)[0])
    data2_indices_below_p_lo = len(np.where(resampled_data2_differences < i_p_lo)[0])
    data2_power = (data2_indices_above_p_hi + data2_indices_below_p_lo) / float(len( resampled_data2_differences ) )
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fpl.histogram(ax, [resampled_differences, resampled_data1_differences, resampled_data2_differences], bins=20, bin_width_ratio=0.6, colors=['black', 'red', 'green'], edgecolor='none', bar_alpha=0.7, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3, 0.3], return_vals=False, show_smoothed=False, normed=True, peak_trace_alpha=0, show_peak_curve=True)
    
    return pval, data1_power, data2_power
                
def number_of_samples_required_to_pass_significance_with_mannwhitneyu(datalist, significance_level=0.05, sampling_resolution=20, iterations=100, bonferoni_correction=True):
    number_datapoints = [len(data) for data in datalist]
    max_samples = np.min(number_datapoints)
    nsamples = np.arange(5, max_samples, sampling_resolution)
    
    if bonferoni_correction:
        significance_level = significance_level / float(len(datalist)-1)
    
    results = -1*np.ones([len(datalist), len(datalist)])    
    
    
    for d1, data1 in enumerate(datalist):
        for d2, data2 in enumerate(datalist):
            for n in nsamples:
                if d1 == d2:
                    continue
                
                pval_tmp = []
                for i in range(iterations):
                    ind1 = np.random.randint(0,len(data1),n)
                    ind2 = np.random.randint(0,len(data2),n)
                    p = scipy.stats.mannwhitneyu(np.array(data1)[ind1], np.array(data2)[ind2])[1]
                    pval_tmp.append(p)
                if np.mean(pval_tmp) < significance_level:
                    if results[d1,d2] == -1:
                        results[d1,d2] = n
                        break
                    
    return results      
                
                
def pval_array_mannwhitneyu(datalist, bonferoni_correction=True):
    results = 0*np.ones([len(datalist), len(datalist)])    
    for d1, data1 in enumerate(datalist):
        for d2, data2 in enumerate(datalist):
            if d1 == d2:
                continue
            p = scipy.stats.mannwhitneyu(data1, data2)[1]
            if bonferoni_correction:
                p *= (len(datalist)-1)
            results[d1,d2] = p
    return results    
    
                
                
                
                
                
                
                
                
                
                
                
