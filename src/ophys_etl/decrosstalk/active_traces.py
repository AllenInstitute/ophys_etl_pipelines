import numpy as np
import scipy.stats
import sys

__all__ = ['get_trace_events',
           'evaluate_components',
           'mode_robust',
           'find_event_gaps']


def mode_robust(input_data, axis=None, d_type=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        def fnc(x): return mode_robust(x, d_type=d_type)
        data_mode = np.apply_along_axis(fnc, axis, input_data)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(dt):
            if dt.size == 1:
                return dt[0]
            elif dt.size == 2:
                return dt.mean()
            elif dt.size == 3:
                i1 = dt[1] - dt[0]
                i2 = dt[2] - dt[1]
                if i1 < i2:
                    return dt[:2].mean()
                elif i2 > i1:
                    return dt[1:].mean()
                else:
                    return dt[1]
            else:

                w_min = np.inf
                n = dt.size / 2 + dt.size % 2
                n = int(n)
                for i in range(0, n):
                    w = dt[i + n - 1] - dt[i]
                    if w < w_min:
                        w_min = w
                        j = i

                return _hsm(dt[j:j + n])

        data = input_data.ravel()  # flatten all dimensions
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if d_type is not None:
            data = data.astype(d_type)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        data_mode = _hsm(data)

    return data_mode


def evaluate_components(traces, event_kernel=5, robust_std=False):
    """
    Define a metric and order components according to the
    probability if some "exceptional events" (like a spike).
    Such probability is defined as the likelihood of
    observing the actual trace value over N samples given
    an estimated noise distribution. The function first
    estimates the noise distribution by considering the
    dispersion around the mode. This is done only using values
    lower than the mode. The estimation of the noise std is
    made robust by using the approximation std=iqr/1.349. Then,
    the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Created on Tue Aug 23 09:40:37 2016
    @author: Andrea G with small modifications from farznaj

    :param traces: numpy.array of size NxM where
                          N : number of neurons (rois),
                          M: number of timestamps (Fluorescence traces)

    :param event_kernel: int, number of consecutive events

    :param robust_std: boolean indicating whether or not
                       to use the interquartile
                       range to calcluate standard deviation

    :return

        idx_components: numpy.array;
                        the components ordered according to the fitness

        fitness: numpy.array;

        erfc: numpy.array;
              probability at each time step of observing the
              N consecutive actual trace values given the distribution
              of noise. For each event, this is the sum of the log
              probabilities of finding a series of {event_kernel} events
              at least as extreme as the {event_kernel} sequence of
              events starting with designated event.

    """

    n_timesteps = np.shape(traces)[-1]
    mode = mode_robust(traces, axis=1)
    ff1 = traces - mode[:, None]

    # only consider values under the mode
    # to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)

    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        ns = np.round(np.sum(ff1 > 0, 1) * .5).astype(int)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1 ** 2, 1) / ns)

    # compute z value
    z = (traces - mode[:, None]) / (3 * sd_r[:, None])

    # probability of observing values larger or equal to z given normal
    # distribution with mean=mode and std sd_r
    erf = 1 - scipy.stats.norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    # np.errstate is to suppress "Divide by zero" warning from np.log
    with np.errstate(divide='ignore'):
        erf = np.log(erf)
        # ToDo: Debug "RuntimeWarning: divide by zero encountered in log"

    # build kernel for event detection
    filt = np.ones(event_kernel)

    # convolve probability with kernel
    erfc = np.apply_along_axis(lambda m: np.convolve(
        m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc[:, :n_timesteps]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components

    #    fitness = fitness[idx_components] % FN
    #    commented bc we want the indexing to match C and YrA.

    #    erfc = erfc[idx_components] % FN
    #    commented bc we want the indexing to match C and YrA.

    return idx_components, fitness, erfc


def find_event_gaps(evs):
    """
    function to find gaps between events
    :param evs: boolean; neurons x frames, indicates if there was an event.
                         (it can be 1 for several continuous frames too)
    :return:
        gap_evs_all: includes the gap before the 1st event and the gap before
                     the last event too, in addition to inter-event gaps
        begs_evs: index of event onsets, excluding the 1st events.
                  (in fact these are 1 index before the actual event onset;
                  since they are computed on the difference trace ('d'))
        ends_evs: index of event offsets, excluding the last event
        gap_evs: includes only gap between events
        bgap_evs: number of frames before the first event
        egap_evs: number of frames after the last event
    """
    d = np.diff(evs.astype(int), axis=1)
    begs = np.array(np.nonzero(d == 1))
    ends = np.array(np.nonzero(d == -1))
    gap_evs_all = []
    gap_evs = []
    begs_evs = []
    ends_evs = []
    bgap_evs = []
    egap_evs = []
    for iu in range(evs.shape[0]):
        # make sure there are events in the trace of unit iu
        if sum(evs[iu]) > 0:
            # indeces belong to "d" (the difference trace)
            begs_this_n = begs[1, begs[0] == iu]
            ends_this_n = ends[1, ends[0] == iu]

            # gap between event onsets will be
            # begs(next event) - ends(current event)
            if not evs[iu, 0] and not evs[iu, -1]:  # normal case
                b = begs_this_n[1:]
                e = ends_this_n[:-1]
                # after how many frames the first event happened
                bgap = [begs_this_n[0] + 1]
                # how many frames with no event exist after the last event
                egap = [evs.shape[1] - ends_this_n[-1] - 1]

            # first event was already going when the recording started.
            elif evs[iu, 0] and not evs[iu, -1]:
                # len(begs_this_n)+1 == len(ends_this_n):
                b = begs_this_n
                e = ends_this_n[:-1]

                bgap = []
                egap = [evs.shape[1] - ends_this_n[-1] - 1]

            # last event was still going on when the recording ended.
            elif not evs[iu, 0] and evs[iu, -1]:
                # len(begs_this_n) == len(ends_this_n)+1:
                b = begs_this_n[1:]
                e = ends_this_n

                bgap = [begs_this_n[0] + 1]
                egap = []

            # first event and last event were happening when the
            # recording started and ended.
            elif evs[iu, 0] and evs[iu, -1]:
                b = begs_this_n
                e = ends_this_n

                bgap = []
                egap = []

            else:
                sys.exit('doesnt make sense! plot d to debug')

            gap_this_n = b - e
            # includes all gaps, before the 1st event, between events,
            # and after the last event.
            gap_this = np.concatenate((bgap, gap_this_n, egap)).astype(int)

        else:  # there are no events in this neuron; set everything to nan
            gap_this = np.nan
            gap_this_n = np.nan
            b = np.nan
            e = np.nan
            bgap = np.nan
            egap = np.nan

        # includes the gap before the 1st event and
        # the gap before the last event too.
        gap_evs_all.append(gap_this)

        # only includes gaps between events: b - e
        gap_evs.append(gap_this_n)

        begs_evs.append(b)
        ends_evs.append(e)
        bgap_evs.append(bgap)
        egap_evs.append(egap)

    # size: number of neurons;
    # each element shows the gap between
    # events for a given neuron
    gap_evs_all = np.array(gap_evs_all, dtype=object)
    gap_evs = np.array(gap_evs, dtype=object)
    begs_evs = np.array(begs_evs, dtype=object)
    ends_evs = np.array(ends_evs, dtype=object)
    bgap_evs = np.array(bgap_evs, dtype=object)
    egap_evs = np.array(egap_evs, dtype=object)

    return gap_evs_all, begs_evs, ends_evs, gap_evs, bgap_evs, egap_evs


def get_traces_evs(traces_y0, th_ag, len_ne):
    """
    Function to get an "active trace" i.e. a trace made by extracting
    and concatenating the active parts of the input trace

    example use:
        len_ne = 20
        th_ag = 10
        [traces_y0_evs, inds_final_all] = get_traces_evs(traces_y0,
                                                         th_ag,
                                                         len_ne)

        or, if need to re-apply to a different input vector:
        traces_active[neuron_y] = traces[neuron_y][inds_final_all[neuron_y]]

    Farzaneh Najafi
    March 2020

    :param traces_y0: numpy.array of size NxM
                      N : number of neurons (rois), M: number of timestamps

    :param th_ag: scalar : threshold to find events on the trace;
                           the higher the more strict on what we call an event.
                           This threshold is applied to the negative of the
                           {erfc} array returned by evaluate_components with
                           {event_kernel}=5 (so, it is a threshold applied on
                           a sum of log probabilities)

    :param len_ne: scalar; number of frames before and after each event that
                           are taken to create traces_events

    :return:
        a dict with keys
        'trace': ndarray, size N (number of neurons);
                          each neuron has size n, which is the size of the
                          "active trace" for that neuron. For each neuron,
                          this array contains the trace values at the active
                          events in that trace.

        'events': ndarray, size number_of_neurons;
                  indices to apply on traces_y0 to get traces_y0_evs:
    """

    #  Andrea Giovannucci's method of identifying "exceptional" events
    [_, _, erfc] = evaluate_components(traces_y0,
                                       event_kernel=5,
                                       robust_std=False)
    erfc = -erfc

    # applying threshold
    evs = (erfc >= th_ag)  # neurons x frames

    # find gaps between events for each neuron
    [_, begs_evs, ends_evs, _, bgap_evs, egap_evs] = find_event_gaps(evs)

    # set traces_evs, ie a trace that contains mostly
    # the active parts of the input trace #
    traces_y0_evs = []
    inds_final_all = []

    for iu in range(traces_y0.shape[0]):

        if sum(evs[iu]) > 0:

            enow = ends_evs[iu]
            bnow = begs_evs[iu]
            e_aft = []
            b_bef = []
            for ig in range(len(bnow)):
                e_aft.append(np.arange(enow[ig],
                                       min(evs.shape[1], enow[ig] + len_ne)))
                b_bef.append(np.arange(bnow[ig] + 1 - len_ne,
                                       min(evs.shape[1], bnow[ig] + 2)))

            e_aft = np.array(e_aft, dtype=object)
            b_bef = np.array(b_bef, dtype=object)

            if len(e_aft) > 1:
                e_aft_u = np.hstack(e_aft)
            else:
                e_aft_u = []

            if len(b_bef) > 1:
                b_bef_u = np.hstack(b_bef)
            else:
                b_bef_u = []

            # below sets frames that cover the duration of all events,
            # but excludes the first and last event
            ev_dur = []
            for ig in range(len(bnow) - 1):
                ev_dur.append(np.arange(bnow[ig], enow[ig + 1]))

            ev_dur = np.array(ev_dur, dtype=object)

            if len(ev_dur) > 1:
                ev_dur_u = np.hstack(ev_dur)
            else:
                ev_dur_u = []
            # ev_dur_u.shape

            evs_inds = np.argwhere(evs[iu]).flatten()  # includes ALL events.

            if len(bgap_evs[iu]) > 0:
                # get len_ne frames before the 1st event
                ind1 = np.arange(np.array(bgap_evs[iu]) - len_ne, bgap_evs[iu])
                # if the 1st event is immediately followed by more events,
                # add those to ind1, because they dont appear in any of the
                # other vars that we are concatenating below.
                if len(ends_evs[iu]) > 1:
                    ii = np.argwhere(
                        np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
                    ind1 = np.concatenate((ind1, evs_inds[:ii]))
            else:
                # first event was already going when the
                # recording started; add these events to ind1
                jj = np.argwhere(np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
                #            jj = ends_evs[iu][0]
                ind1 = evs_inds[:jj + 1]

            if len(egap_evs[iu]) > 0:
                # get len_ne frames after the last event
                indl = np.arange(evs.shape[1] - np.array(egap_evs[iu]) - 1,
                                 min(evs.shape[1], evs.shape[1] -
                                     np.array(egap_evs[iu]) + len_ne))
                # if the last event is immediately preceded by
                # more events, add those to indl,
                # because they dont appear in any
                # of the other vars that we are concatenating
                # below.
                if len(begs_evs[iu]) > 1:
                    # find the fist event of the last event bout
                    ii = np.argwhere(
                        np.in1d(evs_inds, 1 + begs_evs[iu][-1])).squeeze()
                    indl = np.concatenate((evs_inds[ii:], indl))
            else:
                # last event was already going when the recording ended;
                # add these events to ind1
                jj = np.argwhere(
                    np.in1d(evs_inds, begs_evs[iu][-1] + 1)).squeeze()
                indl = evs_inds[jj:]

            inds_final = np.unique(np.concatenate(
                (e_aft_u, b_bef_u, ev_dur_u, ind1, indl))).astype(int)

            # all evs_inds must exist in inds_final,
            # otherwise something is wrong!
            if not np.in1d(evs_inds, inds_final).all():
                # there was only one event bout in the trace
                if not np.array([len(e_aft) > 1,
                                 len(b_bef) > 1,
                                 len(ev_dur) > 1]).all():
                    inds_final = np.unique(np.concatenate(
                        (inds_final, evs_inds))).astype(int)
                else:
                    print(np.in1d(evs_inds, inds_final))
                    msg = 'error in neuron %d! ' % iu
                    msg += 'some of the events dont exist in inds_final! '
                    msg += 'all events must exist in inds_final!'
                    sys.exit(msg)

            # to avoid the negative values that can happen due to
            # taking 20 frames before an even
            inds_final = inds_final[inds_final >= 0]
            traces_y0_evs_now = traces_y0[iu][inds_final]

        else:
            # there are no events in the neuron;
            # assign a nan vector of length 10 to the following vars
            inds_final = np.full((10,), np.nan)
            traces_y0_evs_now = np.full((10,), np.nan)

        inds_final_all.append(inds_final)
        traces_y0_evs.append(traces_y0_evs_now)  # neurons

    inds_final_all = np.array(inds_final_all, dtype=object)
    traces_y0_evs = np.array(traces_y0_evs, dtype=object)  # neurons

    return {'trace': traces_y0_evs,
            'events': inds_final_all}


def get_trace_events(traces, threshold_parameters):
    """
    Wrapper around event detection code.

    Params:
    -------
    traces -- an NxM numpy array. N is the number of ROIs. M is the number
    of timesteps. Contains the trace flux values

    threshold_parameters -- a dict of event detection parameters that
    will be passed to the the actual event detection code
    """

    return get_traces_evs(traces, **threshold_parameters)
