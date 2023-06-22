import numpy as np
import scipy.io
import pandas as pd

import os
from pathlib import Path

import scipy.stats as stats
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.integrate import trapz

import warnings
warnings.filterwarnings('ignore')

############################ STAT FEATURES ##############################

def ecg_stat(array):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A pd.DataFrame
    
    x = np.array(array)
    df = pd.DataFrame(data = [x.max(), x.min(), x.mean(), x.std(),
                              stats.kurtosis(x), stats.skew(x), np.quantile(x,0.5),
                              np.quantile(x, 0.25), np.quantile(x,0.75),
                              np.quantile(x, 0.05), np.quantile(x, 0.95)]).T
    df.columns = ['max_ecg', 'min_ecg', 'mean_ecg', 'sd_ecg', 'ku_ecg', 'sk_ecg', 'median_ecg',
                  'q1_ecg', 'q3_ecg', 'q05_ecg', 'q95_ecg']
    
    return df
 
 
############################ HRV FEATURES ##############################    

def ecg_peaks(array, sampling_rate=1000):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array
    
    x = np.array(array)
    
    # distance : minimal horizontal distance (>= 1) in samples between neighbouring peaks. By default = None
    # height : required height of peaks. By default = None
    
    distance = sampling_rate / 3
    #height = x.max() / 2
    height = np.quantile(x, 0.99)/2 #use 99th quantile instead of max (to ignore outliers)
    r_peaks, _ = signal.find_peaks(x, distance= distance, height=height)
    
    return r_peaks
    
    

def heart_rate(r_peaks, sampling_rate=1000, upsample_rate=4):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A pd.DataFrame
    
    x = np.array(r_peaks)
    rri = np.diff(x)
    rri = 1000 * rri / sampling_rate #convert to ms
    hr = 1000 * 60 / rri
    
    hr_time = np.cumsum(rri) / 1000
    hr_time -= hr_time[0] 
    
    interpolation_f = interp1d(hr_time, hr, kind='cubic')
    
    x = np.arange(1, hr_time.max(), 1/upsample_rate)
    hr_interpolated = interpolation_f(x)
    return hr_interpolated, x
    
    
def ecg_time(array, sampling_rate=1000):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A pd.DataFrame
    
    x = np.array(array)
    r_peaks = ecg_peaks(x, sampling_rate=sampling_rate)
    
    # RR intervals are expressed in number of samples and need to be converted into ms. By default, sampling_rate=1000
    # HR is given by: 60/RR intervals in seconds. 
    
    rri = np.diff(r_peaks) #RR intervals
    rri = 1000 * rri / sampling_rate #Convert to ms
    drri = np.diff(rri) #Difference between successive RR intervals
    hr = 1000*60 / rri #Heart rate
    
    meanHR = hr.mean()
    minHR = hr.min()
    maxHR = hr.max()
    sdHR = hr.std()
    modeHR =  maxHR - minHR
    nNN = rri.shape[0] / (x.shape[0]/sampling_rate/60) #to get the number of NN intervals per minute
    meanNN = rri.mean()
    SDSD = drri.std()
    CVNN = rri.std()
    SDNN = CVNN / meanNN
    pNN50 = np.sum(np.abs(drri)>50) / nNN * 100
    pNN20 = np.sum(np.abs(drri)>20) / nNN * 100
    RMSSD = np.sqrt(np.mean(drri**2))
    medianNN = np.quantile(rri, 0.5)
    q20NN = np.quantile(rri, 0.2)
    q80NN = np.quantile(rri, 0.8)
    minNN = rri.min()
    maxNN = rri.max()
    
    # HRV triagular index (HTI): The density distribution D of the RR intervals is estimated. The most frequent
    # RR interval lenght X is established. Y= D(X) is the maximum of the sample density distribution.
    # The HTI is then obtained as : (the number of total NN intervals)/Y

    bins = np.arange(meanNN - 3*CVNN , meanNN + 3*CVNN, 10)
    d,_ = np.histogram(rri, bins=bins)
    y = d.argmax()
    triHRV = nNN / y
    
    df = pd.DataFrame(data = [meanHR, minHR, maxHR, sdHR, modeHR, nNN, meanNN, 
                              SDSD, CVNN, SDNN, pNN50, pNN20, RMSSD, medianNN,
                              q20NN, q80NN, minNN, maxNN, triHRV]).T
    
    df.columns = ['meanHR', 'minHR', 'maxHR', 'sdHR', 'modeHR', 'nNN','meanNN', 'SDSD', 'CVNN', 
                  'SDNN', 'pNN50', 'pNN20', 'RMSSD', 'medianNN', 'q20NN','q80NN','minNN', 'maxNN', 'triHRV']
    return df
    
    

############################ FREQ FEATURES ##############################

def interpolate_Rpeaks(peaks, sampling_rate=1000, upsample_rate=4):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # The RR intervals are aranged over time, and the values are summed up to find the time points.
    # An interpolation function is defined, to use to sample from with any upsampling resolution. 
    # By default upsample_rate = 4 : 4 evenly spaced data points per seconds are added. 
    
    rr = np.diff(peaks)
    rr = 1000 * rr / sampling_rate # convert to ms
    rr_time = np.cumsum(rr) / 1000 # convert to s
    rr_time -= rr_time[0] 
    
    interpolation_f = interp1d(rr_time, rr, kind='cubic')
    
    x = np.arange(1, rr_time.max(), 1/upsample_rate)
    rr_interpolated = interpolation_f(x)
    
    return rr_interpolated, x
    
    
    
def ecg_freq(array, sampling_rate=1000, upsample_rate=4, freqband_limits=(.0, .0033,.04,.15,.4, .5)):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # FFT needs evenly sampled data, so the RR-interval can't be used directly and need to
    # be interpolated. Then the spectral density of the signal is computed using Welch method.
    
    x = np.array(array)
    r_peaks = ecg_peaks(x, sampling_rate=sampling_rate)
    rri, _ = interpolate_Rpeaks(r_peaks, upsample_rate=upsample_rate)
    freq, power = signal.welch(x=rri, fs=upsample_rate)
    
    lim_ulf= (freq >= freqband_limits[0]) & (freq < freqband_limits[1])
    lim_vlf = (freq >= freqband_limits[1]) & (freq < freqband_limits[2])
    lim_lf = (freq >= freqband_limits[2]) & (freq < freqband_limits[3])
    lim_hf = (freq >= freqband_limits[3]) & (freq < freqband_limits[4])
    lim_vhf = (freq >= freqband_limits[4]) & (freq < freqband_limits[5])
    
    # The power (PSD) of each frequency band is obtained by integrating the spectral density 
    # by trapezoidal rule, using the scipy.integrate.trapz function.
    
    ulf = trapz(power[lim_ulf], freq[lim_ulf])
    vlf = trapz(power[lim_vlf], freq[lim_vlf])
    lf = trapz(power[lim_lf], freq[lim_lf])
    hf = trapz(power[lim_hf], freq[lim_hf])
    vhf = trapz(power[lim_vhf], freq[lim_vhf])
    totalpower = ulf + vlf + lf + hf + vhf
    lfhf = lf / hf
    rlf = lf / (lf + hf) * 100
    rhf = hf / (lf + hf) * 100
    peaklf = freq[lim_lf][np.argmax(power[lim_lf])]
    peakhf = freq[lim_hf][np.argmax(power[lim_hf])]
    
    df = pd.DataFrame(data = [totalpower, lf, hf, ulf, vlf, vhf, lfhf,
                             rlf, rhf, peaklf, peakhf]).T
    
    df.columns = ['totalpower', 'LF', 'HF', 'ULF', 'VLF', 'VHF', 'LF/HF',
                 'rLF', 'rHF', 'peakLF', 'peakHF']
    return df
    
    
    
    
############################ NONLIN FEATURES ##############################  


def apEntropy(array, m=2, r=None):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A Float
    
    # m : a positive integer representing the length of each compared run of data (a window).
    # By default m = 2
    # r : a positive real number specifying a filtering level. By default r = 0.2 * sd.
    
    x = np.array(array)
    N = len(x)
    r = 0.2 * x.std() if r == None else r == r
    
    # A sequence of vectors z(1),..., z(N-m+1) is formed from a time series of N equally 
    # spaced raw data values x(1),…,x(N), such that z(i) = x(1),...,x(i+m-1).
    
    # For each i in {1,..., N-m+1}, C = [number of z(j) such that d(x(i),x(j)) < r]/[N-m+1]
    # is computed, with d(z(i),z(j)) = max|x(i)-x(j)|
    
    # phi_m(r) = (N-m+1)^-1 x sum log(Ci) is computed, and the ap Entropy is given by: 
    # phi_m(r) - phi_m+1(r)

    def _maxdist(zi, zj):
        return max([abs(xi - xj) for xi, xj in zip(zi, zj)])
    
    def _phi(m):
        z = [[x[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for zj in z if _maxdist(zi, zj) <= r]) / (N - m + 1.0)
        for zi in z]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
    apEn = abs(_phi(m + 1) - _phi(m))
    return apEn
    
    
def sampEntropy(array, m=2, r=None):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A Float
    
    # m: embedding dimension
    # r: tolerance distance to consider two data points as similar. By default r = 0.2 * sd.
    
    # SampEn is the negative logarithm of the probability that if two sets of simultaneous 
    # data points of length m have distance < r then two sets of simultaneous data points of 
    # length m + 1 also have distance < r. 
    
    x = np.array(array)
    N = len(x)
    r = 0.2 * x.std() if r == None else r == r
    
    # All templates vector of length m are defined. Distances d(xmi, xmj) are computed
    # and all matches such that d < r are saved. Same for the distances d(xm+1i, xm+1j).
    
    xmi = np.array([x[i : i + m] for i in range(N - m)])
    xmj = np.array([x[i : i + m] for i in range(N - m + 1)])
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    m += 1
    xm = np.array([x[i : i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    
    return -np.log(A / B)
    
    
def ecg_nonlinear(array, sampling_rate=1000, m=2, r=None):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # The Poincaré ellipse plot is diagram in which each RR intervals are plotted as a function 
    # of the previous RR interval value. SD1 is the standard deviation spread orthogonally 
    # to the identity line (y=x) and is the ellipse width. SD2 is the standard deviation spread
    # along the identity line and specifies the length of the ellipse.
    
    x = np.array(array)
    r_peaks = ecg_peaks(x, sampling_rate=sampling_rate)
    rri = np.diff(r_peaks)
    rr1 = rri[:-1]
    rr2 = rri[1:]
    SD1 =  np.std(rr2 - rr1) / np.sqrt(2)
    SD2 =  np.std(rr2 + rr1) / np.sqrt(2)
    SD1SD2 = SD1/SD2
    apEn = apEntropy(r_peaks, m=2)
    sampEn = sampEntropy(r_peaks, m=2)
    
    df = pd.DataFrame(data = [SD1, SD2, SD1SD2, apEn, sampEn]).T
    
    df.columns = ['SD1', 'SD2', 'SD1SD2', 'apEn', 'sampEn']
    return df
    



############################ DATABASE ##############################

def ecg_stat_features(dictionnary):
    data = dictionnary.copy()
    df_stat = ecg_stat( list(data.values())[0] )
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_stat = pd.concat([df_stat, ecg_stat(v)], axis=0) 
        names.append(k)
    
    df_stat.set_axis(names, inplace=True)
    return df_stat

def ecg_time_features(dictionnary, sampling_freq=500):
    data = dictionnary.copy()
    df_time = ecg_time(list(data.values())[0] , sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_time = pd.concat([df_time, ecg_time(v, sampling_freq)], axis=0)
        names.append(k)
        
    df_time.set_axis(names, inplace=True)
    return df_time


def ecg_freq_features(dictionnary, sampling_freq =500):
    data = dictionnary.copy()
    df_freq = ecg_freq( list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_freq = pd.concat([df_freq, ecg_freq(v, sampling_freq)], axis=0)
        names.append(k)
        
    df_freq.set_axis(names, inplace=True)
    return df_freq

def ecg_nonlinear_features(dictionnary, sampling_freq =500):
    data = dictionnary.copy()
    df_nonlinear = ecg_nonlinear( list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_nonlinear = pd.concat([df_nonlinear, ecg_nonlinear(v, sampling_freq)], axis=0)
        names.append(k)
        
    df_nonlinear.set_axis(names, inplace=True)
    return df_nonlinear


def get_ecg_features(dictionnary, sampling_freq =500):
    df_stat = ecg_stat_features(dictionnary)
    df_time = ecg_time_features(dictionnary, sampling_freq)
    df_freq = ecg_freq_features(dictionnary, sampling_freq)
    df_nonlinear = ecg_nonlinear_features(dictionnary, sampling_freq)
    
    df = pd.concat([df_time, df_stat, df_freq, df_nonlinear], axis=1)
    #df.set_axis(names, inplace=True)
    return df