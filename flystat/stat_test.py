import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages
import fly_plot_lib.plot as fpl

import numpy as np
import scipy.stats
import resampling
import matplotlib.pyplot as plt


def test():
    data1, data2 = get_two_datasets(100)
    plot_data(data1, data2)
    print( resampling.calc_statistical_significance_through_resampling(data1, data2, analysis='median'))
def plot_data(data1, data2):
    m1 = np.median(data1)
    m2 = np.median(data2)

    conf1 = resampling.bootstrap_confidence_intervals_from_data(data1, confidence_lo=0.025, confidence_hi=0.975, use='median', iterations=1000)
    conf2 = resampling.bootstrap_confidence_intervals_from_data(data2, confidence_lo=0.025, confidence_hi=0.975, use='median', iterations=1000)
    
    if m1 < m2:
        dconf = conf2[0] - m1#conf1[1]
    else:
        dconf = conf1[0] - m2#conf2[1]
        
    return dconf
    #std1 = resampling.bootstrap_std_confidence_intervals(data1, iterations=1000)
    #std2 = resampling.bootstrap_std_confidence_intervals(data2, iterations=1000)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    
    #fpl.plot_confidence_interval(ax, 1, m1, conf1, confidence_interval_50=None, width=0.3, color='blue', linewidth=3, alpha95=0.3, alpha50=0.5)
    #fpl.plot_confidence_interval(ax, 2, m2, conf2, confidence_interval_50=None, width=0.3, color='blue', linewidth=3, alpha95=0.3, alpha50=0.5)
    
    #fpl.scatter_box(ax, 1, data1, xwidth=0.3, ywidth=0.1, color='black', flipxy=False, shading='95conf')
    #fpl.scatter_box(ax, 2, data2, xwidth=0.3, ywidth=0.1, color='black', flipxy=False, shading='95conf')
    
    #print std1
    #ax.vlines(1, conf1[0]-std1[0], conf1[1]+std1[1])
    #ax.vlines(2, conf2[0]-std2[0], conf2[1]+std2[1])
    
def iterate_conf_check(n=100):
    
    dconfs = []
    pvals = []
    power1s = []
    power2s = []
    for i in range(n):
        print (i)
        data1, data2 = get_two_datasets(50)
        dconf = plot_data(data1, data2)
        pval, power1, power2 = resampling.plot_calc_statistical_significance_through_resampling(data1, data2, analysis='median')
        power1s.append(power1)
        power2s.append(power2)
        dconfs.append(dconf)
        pvals.append(pval)
    
    dconfs = np.array(dconfs)
    pvals = np.array(pvals)
    
    isort = np.argsort(pvals)
    
    dconfs = dconfs[isort]
    pvals = pvals[isort]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(pvals, dconfs)
    
    if 0:
        p05 = np.where(pvals<0.05)[0]
        p95 = np.where(pvals>0.05)[0]
        
        dconf_incorrectly_predicts_not_signficant = len( np.where( dconfs[p05] < 0 )[0] ) / float(len(dconfs[p05])) # type 1 error
        dconf_incorrectly_predicts_signficant = len( np.where( dconfs[p95] > 0 )[0] ) / float(len(dconfs[p95])) # type 2 error
        
        print (dconf_incorrectly_predicts_not_signficant) # fails to reject null hypothesis
        print (dconf_incorrectly_predicts_signficant) # fails to accept null hypothesis

    if 1:
        p05 = np.where(pvals<0.05)[0]
        p95 = np.where(pvals>0.05)[0]
        
        dconf_incorrectly_predicts_not_significant = len( np.where( dconfs[p05] < 0 )[0] ) 
        dconf_incorrectly_predicts_significant = len( np.where( dconfs[p95] > 0 )[0] ) 
        
        print ('Probability conf interval disagrees with resampling stat: ', (dconf_incorrectly_predicts_not_significant + dconf_incorrectly_predicts_significant) / float(len(dconfs)))
        print (dconf_incorrectly_predicts_not_significant) # fails to reject null hypothesis
        print (dconf_incorrectly_predicts_significant) # fails to accept null hypothesis
    
    #return pvals, dconfs

def get_data(n, mean, std):
    normal_distribution = scipy.stats.norm(mean, std)
    data = np.array([normal_distribution.rvs() for i in range(n)])
    return data
    
def get_two_datasets(n):
    data1 = get_data(n, 10, 5)
    data2 = get_data(n, 11, 5)
    return data1, data2
    


def plot_standard_errors(ax, x, data1, data2, width=0.6):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    std_err1 = np.std(data1) / np.sqrt(len(data1))
    std_err2 = np.std(data2) / np.sqrt(len(data2))
    
    #plt.fill_between([x-width/2.,x], [0, 0], [median1, median1])
    
    
    
def get_statistical_measures(data1, data2):
    ttest = scipy.stats.ttest_ind(data1, data2)[1]
    resampling_pval = resampling.calc_statistical_significance(data1, data2, iterations=1000)
    probability_of_correct_assignment = resampling.probability_we_assign_correct_distribution_to_random_data_point([data1, data2])
    
    return [ttest, resampling_pval, probability_of_correct_assignment]
    
    
def explore_statistical_measures_across_repetitions(iterations=5, ndatapoints=1000):
    stats = []
    for i in range(iterations):
        data1, data2 = get_two_datasets(ndatapoints)
        s = get_statistical_measures(data1, data2)
        stats.append(s)
    return stats
    
def analyze_statistical_exploration():
    ndatapoints = [5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    
    medians = []
    stdevs  = []
    for n in ndatapoints:
        stats = explore_statistical_measures_across_repetitions(iterations=20, ndatapoints=n)
        medians.append(np.mean(stats, axis=0))
        stdevs.append(np.std(stats, axis=0))
    medians = np.array(medians)
    stdevs = np.array(stdevs)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx()
    
    ax.plot(ndatapoints, medians[:,0], color='black', linewidth=4)
    ax.vlines(ndatapoints, medians[:,0]-stdevs[:,0], medians[:,0]+stdevs[:,0], color='black', linewidth=2)
    
    ax.plot(np.array(ndatapoints)+0.1*np.array(ndatapoints), medians[:,1], color='red', linewidth=4)
    ax.vlines(np.array(ndatapoints)+0.1*np.array(ndatapoints), medians[:,1]-stdevs[:,1], medians[:,1]+stdevs[:,1], color='red', linewidth=2)
    
    ax.plot(np.array(ndatapoints)-0.1*np.array(ndatapoints), medians[:,2], color='green', linewidth=4)
    ax.vlines(np.array(ndatapoints)-0.1*np.array(ndatapoints), medians[:,2]-stdevs[:,2], medians[:,2]+stdevs[:,2], color='green', linewidth=2)
    
    ax.set_xlim(0,1000)
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlabel('number of data points')
    ax.set_ylabel('statistical measure')
    fig.savefig('stats.pdf', format='pdf')
    
    fig = plt.figure(figsize=(10,8))
    bins = np.linspace(0,30,30)
    for i, n in enumerate(ndatapoints):
        ax = fig.add_subplot(1,len(ndatapoints),i+1)
        data1, data2 = get_two_datasets(n)
        fpl.histogram(ax, [data1, data2], bins=bins, colors=['blue', 'orange'], show_smoothed=False, normed=True,bin_width_ratio=0.9)
        ax.set_xlim(bins[0], bins[-1])
        fpl.adjust_spines(ax, ['bottom'], xticks=[0,30])
    
    fig.savefig('distributions.pdf', format='pdf')
