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


def rsp_stat(array):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A pd.DataFrame
    
    x = np.array(array)
    df = pd.DataFrame(data = [x.max(), x.min(), x.mean(), x.std(),
                              stats.kurtosis(x), stats.skew(x), np.quantile(x,0.5),
                              np.quantile(x, 0.25), np.quantile(x,0.75),
                              np.quantile(x, 0.05), np.quantile(x, 0.95)]).T
    df.columns = ['max_rsp', 'min_rsp', 'mean_rsp', 'sd_rsp', 'ku_rsp', 'sk_rsp', 'median_rsp',
                  'q1_rsp', 'q3_rsp', 'q05_rsp', 'q95_rsp']
    
    return df
  
  
############################ TIME FEATURES ##############################  
    
def rsp_peaks(array, sampling_rate=1000):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array
    
    x = np.array(array)
    
    # distance : minimal horizontal distance (>= 1) in samples between neighbouring peaks. By default = None
    # height : required height of peaks. By default = None
    
    distance = sampling_rate * 3 #If we suppose a healthy person takes up to 20 breaths per minute
    height = x.std() / 4
    r_peaks, _ = signal.find_peaks(x, distance= distance, height=height)
    
    return r_peaks
    
    
def rsp_troughs(array, sampling_rate=1000):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array
    
    x = np.array(array)
    
    # distance : minimal horizontal distance (>= 1) in samples between neighbouring peaks. By default = None
    # height : required height of peaks. By default = None
    
    distance = sampling_rate * 3 #If we suppose a healthy person takes up to 20 breaths per minute
    height = x.std() / 4
    r_troughs, _ = signal.find_peaks(-x, distance= distance, height=height)
    
    return r_troughs
    
    
def rsp_time(array, sampling_rate=1000):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A pd.DataFrame
    
    x = np.array(array)
    r_peaks = rsp_peaks(x, sampling_rate=sampling_rate)
    
    # BB intervals are expressed in number of samples and need to be converted into ms. By default, sampling_rate=1000
    # RR is given by: 60/BB intervals in seconds. 
    
    bbi = np.diff(r_peaks) #RR intervals
    bbi = 1000 * bbi / sampling_rate #Convert to ms
    dbbi = np.diff(bbi) #Difference between successive RR intervals
    rr = 1000*60 / bbi #Respiration rate
    
    meanRR = rr.mean()
    minRR = rr.min()
    maxRR = rr.max()
    sdRR = rr.std()
    modeRR =  maxRR - minRR
    nBB = bbi.shape[0] / (x.shape[0]/sampling_rate/60) #to get the number of BB intervals per minute
    meanBB = bbi.mean()
    SDSDb = dbbi.std()
    CVNNb = bbi.std()
    SDNNb = CVNNb / meanBB
    RMSSDb = np.sqrt(np.mean(dbbi**2))
    medianBB = np.quantile(bbi, 0.5)
    q20BB = np.quantile(bbi, 0.2)
    q80BB = np.quantile(bbi, 0.8)
    minBB = bbi.min()
    maxBB = bbi.max()
    
    # TT intervals are expressed in number of samples and need to be converted into ms. By default, sampling_rate=1000
    # TT intervals correspond to time intervals between successives troughs.
    
    r_troughs = rsp_troughs(x, sampling_rate=sampling_rate)
    tti = np.diff(r_troughs)
    if tti.shape[0] != 0:
        meanTT = tti.mean()
        SDTT = tti.std()
        medianTT = np.quantile(tti, 0.5)
        q20TT = np.quantile(tti, 0.2)
        q80TT = np.quantile(tti, 0.8)
        minTT = tti.min()
        maxTT = tti.max()

    
        # Breath amplitudes coresponds to the differences between succesive peaks and trough. 
        # They are expressed in mV. 

        ba = np.subtract(x[r_peaks], np.mean(x[r_troughs]))

        meanBA = ba.mean()
        SDBA = ba.std()
        medianBA = np.quantile(ba, 0.5)
        q20BA = np.quantile(ba, 0.2)
        q80BA = np.quantile(ba, 0.8)
        minBA = ba.min()
        maxBA = ba.max()

        # Breath widths are the differences between two successive peak and trough, expressed in ms. 
        bw = np.sort(np.concatenate((r_peaks, r_troughs)))

        meanBW = bw.mean()
        SDBW = bw.std()
        medianBW = np.quantile(bw, 0.5)
        q20BW = np.quantile(bw, 0.2)
        q80BW = np.quantile(bw, 0.8)
        minBW = bw.min()
        maxBW = bw.max()
    
    else:
        tti = np.nan
        meanTT = np.nan
        SDTT = np.nan
        medianTT = np.nan
        q20TT = np.nan
        q80TT = np.nan
        minTT = np.nan
        maxTT = np.nan
        ba = np.nan

        meanBA = np.nan
        SDBA = np.nan
        medianBA = np.nan
        q20BA = np.nan
        q80BA = np.nan
        minBA = np.nan
        maxBA = np.nan

        # Breath widths are the differences between two successive peak and trough, expressed in ms. 
        bw = np.nan

        meanBW = np.nan
        SDBW = np.nan
        medianBW = np.nan
        q20BW = np.nan
        q80BW = np.nan
        minBW = np.nan
        maxBW = np.nan
        
    df = pd.DataFrame(data = [meanRR, minRR, maxRR, sdRR, modeRR, nBB, meanBB, 
                              SDSDb, CVNNb, SDNNb, RMSSDb, medianBB,
                              q20BB, q80BB, minBB, maxBB, meanTT, SDTT, medianTT,
                              q20TT, q80TT, minTT, maxTT, meanBA, SDBA, medianBA,
                              q20BA, q80BA, minBA, maxBA, meanBW, SDBW, medianBW,
                              q20BW, q80BW, minBW, maxBW]).T
    
    df.columns = ['meanRR', 'minRR', 'maxRR', 'sdRR', 'modeRR', 'nBB','meanBB', 'SDSDb', 'CVNNb', 
                  'SDNNb', 'RMSSDb', 'medianBB', 'q20BB','q80BB','minBB', 'maxBB', 'meanTT', 'SDTT', 'medianTT',
                  'q20TT', 'q80TT', 'minTT', 'maxTT', 'meanBA', 'SDBA', 'medianBA', 'q20BA', 'q80BA', 'minBA',
                  'maxBA', 'meanBW', 'SDBW', 'medianBW', 'q20BW', 'q80BW', 'minBW','maxBW']
    return df
    
    
############################ FREQ FEATURES ##############################

def interpolate_rsp_peaks(peaks, sampling_rate=1000, upsample_rate=4):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # The RR intervals are aranged over time, and the values are summed up to find the time points.
    # An interpolation function is defined, to use to sample from with any upsampling resolution. 
    # By default upsample_rate = 4 : 4 evenly spaced data points per seconds are added. 
    
    bb = np.diff(peaks)
    bb = 1000 * bb / sampling_rate # convert to ms
    bb_time = np.cumsum(bb) / 1000 # convert to s
    bb_time -= bb_time[0] 
    
    interpolation_f = interp1d(bb_time, bb, kind='cubic')
    
    x = np.arange(1, bb_time.max(), 1/upsample_rate)
    bb_interpolated = interpolation_f(x)
    
    return bb_interpolated, x
    
    
    
    
def rsp_freq(array, sampling_rate=1000, upsample_rate=6, freqband_limits=(.1,.05,.15,.5,.8)):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # FFT needs evenly sampled data, so the RR-interval can't be used directly and need to
    # be interpolated. Then the spectral density of the signal is computed using Welch method.
    
    x = np.array(array)
    r_peaks = rsp_peaks(x, sampling_rate=sampling_rate)
    try:
        rri, _ = interpolate_rsp_peaks(r_peaks, upsample_rate=upsample_rate)
        freq, power = signal.welch(x=rri, fs=upsample_rate)

        lim_vlf = (freq >= freqband_limits[0]) & (freq < freqband_limits[1])
        lim_lf = (freq >= freqband_limits[1]) & (freq < freqband_limits[2])
        lim_hf = (freq >= freqband_limits[2]) & (freq < freqband_limits[3])
        lim_vhf = (freq >= freqband_limits[3]) & (freq < freqband_limits[4])

        # The power (PSD) of each frequency band is obtained by integrating the spectral density 
        # by trapezoidal rule, using the scipy.integrate.trapz function.

        vlf = trapz(power[lim_vlf], freq[lim_vlf])
        lf = trapz(power[lim_lf], freq[lim_lf])
        hf = trapz(power[lim_hf], freq[lim_hf])
        vhf = trapz(power[lim_vhf], freq[lim_vhf])

        totalpower = vlf + lf + hf + vhf 
        lfhf = lf / hf
        rlf = lf / (lf + hf) * 100
        rhf = hf / (lf + hf) * 100
        peaklf = freq[lim_lf][np.argmax(power[lim_lf])]
        peakhf = freq[lim_hf][np.argmax(power[lim_hf])]

        df = pd.DataFrame(data = [totalpower, lf, hf, vlf, vhf, lfhf,
                                  rlf, rhf, peaklf, peakhf]).T
    except ValueError:
        df = pd.DataFrame(data = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
                                  np.NaN, np.NaN, np.NaN, np.NaN]).T

    
    df.columns = ['totalpower_rsp', 'LF_rsp', 'HF_rsp', 'VLF_rsp', 'VHF_rsp', 'LF/HF_rsp',
                 'rLF_rsp', 'rHF_rsp', 'peakLF_rsp', 'peakHF_rsp']
    
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
        try:
            p = (N - m + 1.0) ** (-1) * sum(np.log(C))
        except ZeroDivisionError:
            p = np.nan
        return p
    
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
    
    
def rsp_nonlinear(array, sampling_rate=1000, m=2, r=None):
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # The Poincaré ellipse plot is diagram in which each RR intervals are plotted as a function 
    # of the previous RR interval value. SD1 is the standard deviation spread orthogonally 
    # to the identity line (y=x) and is the ellipse width. SD2 is the standard deviation spread
    # along the identity line and specifies the length of the ellipse.
    
    x = np.array(array)
    r_peaks = rsp_peaks(x, sampling_rate=sampling_rate)
    rri = np.diff(r_peaks)
    rr1 = rri[:-1]
    rr2 = rri[1:]
    SD1 =  np.std(rr2 - rr1) / np.sqrt(2)
    SD2 =  np.std(rr2 + rr1) / np.sqrt(2)
    SD1SD2 = SD1/SD2
    apEn = apEntropy(r_peaks, m=2)
    #sampEn = sampEntropy(r_peaks, m=2)
    
    df = pd.DataFrame(data = [SD1, SD2, SD1SD2, apEn]).T
    #, sampEn]).T
    
    df.columns = ['SD1_rrv', 'SD2_rrv', 'SD1SD2_rrv', 'apEn_rrv']
    #, 'sampEn_rrv']
    return df
    


############################ DATABASE ##############################
    
    
def resp_stat_features(dictionnary):
    data = dictionnary.copy()
    df_stat = rsp_stat(list(data.values())[0])
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_stat = pd.concat([df_stat, rsp_stat(v)], axis=0)
        names.append(k)
        
    df_stat.set_axis(names, inplace=True)
    return df_stat


def resp_time_features(dictionnary, sampling_freq =500):
    data = dictionnary.copy()
    df_time = rsp_time(list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_time = pd.concat([df_time, rsp_time(v, sampling_freq)], axis=0)
        names.append(k)
        
    df_time.set_axis(names, inplace=True)
    return df_time


def resp_freq_features(dictionnary, sampling_freq =500):
    data = dictionnary.copy()
    df_freq = rsp_freq(list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_freq = pd.concat([df_freq, rsp_freq(v, sampling_freq)], axis=0)
        names.append(k)
        
    df_freq.set_axis(names, inplace=True)
    return df_freq



def resp_nonlinear_features(dictionnary, sampling_freq =500):
    data = dictionnary.copy()
    df_nonlinear = rsp_nonlinear(list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_nonlinear = pd.concat([df_nonlinear, rsp_nonlinear(v, sampling_freq)], axis=0)
        names.append(k)
        
    df_nonlinear.set_axis(names, inplace=True)
    return df_nonlinear


def get_resp_features(dictionnary, sampling_freq =500):
    df_stat = resp_stat_features(dictionnary)
    df_time = resp_time_features(dictionnary, sampling_freq)
    df_freq = resp_freq_features(dictionnary, sampling_freq)
    df_nonlinear = resp_nonlinear_features(dictionnary, sampling_freq)
    
    df = pd.concat([df_time, df_stat, df_freq, df_nonlinear], axis=1)
    return df