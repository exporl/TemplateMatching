from scipy.signal import argrelextrema
import numpy as np


def get_rms(trf):
    rms_over_channels = []
    for time_idx in np.arange(0, trf.shape[0]):
        x = abs(trf[time_idx,:])
        c_rms = np.sqrt(np.vdot(x, x) / x.size)
        rms_over_channels.append(c_rms)
    return np.array(rms_over_channels)

def shuffle_array(data): 
    flat_data = data.flatten()
    # do permuations
    shuffled_data = np.random.permutation(flat_data)
    # reshape to original size
    shuffled_data.resize(data.shape)
    return shuffled_data

def find_latency_template_matching(trf, trf_times, template, time_bounds, n_shuffles):
    """ Function to find the latency of a peak based upon a specified time region and given template (template can be
    be derived from the average across the whole sample. The template matching algorithm is based upon the local maxima
    of the RMS value. At a peak, there i a high variation across the different channels in the TRF (negative and positive
    polarities, i.e. the RMS will be high. At the local maxima of the RMS, the topography is investigated. The topography
    with the highest correlation with the template is chosen as corresponding peak. An addition criteria is opposed that
    at least 30 channels must have the same polarity as the template.

    INPUTS
    ----------
    trf         trf array (time x channels)
    trf_time    array with time samples (time x 1)
    template    array with the topography value for each channel (channels x 1)
    time_bounds two dimensional array in which the peak is searched
    n_shuffles  the number of shuffles to determine the significance of the found peak

    OUTPUTS
    ----------
    corresponding_latency   latency of found peak
    topography              found topography associated with the peak (channels x 1)

    NOTE: the unit of the time dimension can be chosen arbitarily (milliseconds, seconds, etc). It just needs to be
    consistent across the inputs

    ----------
    written by Marlies (04/07/2022)
    Questions: marlies.gillis@kuleuven.be
    """

    # take subset of trf
    subset_trf = np.array([trf_element for trf_element, trf_time in zip(trf, trf_times) if trf_time>time_bounds[0] and trf_time<time_bounds[1]])
    subset_trf_times = np.array([trf_time for trf_time in trf_times if trf_time>time_bounds[0] and trf_time<time_bounds[1]])

    # get the rms
    rms = get_rms(subset_trf)

    # find local maxima in the rms
    local_maxima = argrelextrema(rms, np.greater)
    if len(local_maxima[0]) == 0:
        corresponding_latency = np.nan
        topography = np.empty(template.shape)
        topography[:] = np.nan
        return corresponding_latency, topography

    # get times of local maxima
    correlations = np.array([np.corrcoef(subset_trf[local_maximum], template)[0,1] for local_maximum in local_maxima[0]])

    # find latency with highest correlation
    maximal_correlation_idx = correlations.argmax()
    indice_maximal_correlation = local_maxima[0][maximal_correlation_idx]
    corresponding_latency = subset_trf_times[indice_maximal_correlation]
    topography = subset_trf[indice_maximal_correlation, :]

    # get significance of the correlation
    maximal_correlations = []
    maximal_correlation_same_signs = []
    random.seed(0)
    for shuffle_idx in range(0, n_shuffles): 
        
        # shuffle the TRF in the time window
        data_trf = trf.x[0, :, :]
        shuffled_TRF = shuffle_array(data_trf)

        # check the rms 
        rms_values = get_rms(data_trf)
        local_maxima_shuffled_rms = argrelextrema(rms_values, np.greater)
        if len(local_maxima[0]) == 0:
            continue
        
        # check the correlations at the RMS
        correlations_shuffled_with_template = [scipy.stats.pearsonr(template, shuffled_TRF[:, idx])[0] for idx in local_maxima_shuffled_rms[0]]
        maximal_shuffled_correlation_idx = np.argmax(correlations_shuffled_with_template)
        
        maximal_correlations.append(correlations_shuffled_with_template[maximal_shuffled_correlation_idx])
        
        # find index of sample
        sample_idx = local_maxima_shuffled_rms[0][maximal_shuffled_correlation_idx]
        similar_signs = sum(np.sign(shuffled_TRF[:, sample_idx]) == np.sign(template))
        if similar_signs >= 30:
            maximal_correlation_same_signs.append(correlations_shuffled_with_template[maximal_shuffled_correlation_idx])


    # check how many channels have the same sign as the template
    similar_signs = sum(np.sign(topography) == np.sign(template))
    if similar_signs < 30:
        corresponding_latency = np.nan
        topography = np.empty(template.shape)
        topography[:] = np.nan
        
    # check whether the correlation of the peak is above the 95th percentile
    if not correlations[maximal_correlation_idx] > np.percentile(maximal_correlation_same_signs, 95):
        corresponding_latency = np.nan
        topography = np.empty(template.shape)
        topography[:] = np.nan

    return corresponding_latency, topography
