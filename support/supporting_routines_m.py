
import pandas as pd
import numpy as np
import xarray as xr
import datetime
from datetime import datetime 
import time

import matplotlib
import matplotlib.pyplot as plt



#  *************************************

def maskPrior(prior_all, time_index, lat_long_mask):
    # This function takes a dataset prior_all (xarray dataset)
    # and applies a mask "lat_long_mask" (i.e. indices of 
    # latitude,longitud combinations)
    # This mask is applied to the time index as well (time_index)
    prior_local = xr.Dataset(data_vars={"Temperature":(["time","height"],
                                                   prior_all['Temperature'].values[:,lat_long_mask].T,
                                                   {'units':'K'}), 
                           "Humidity":(["time","height"], 
                                       prior_all['Humidity'].values[:,lat_long_mask].T*1000,
                                       {'units':'g/kg'}),
                           "Pressure":(["time","height"],
                                       prior_all['Pressure'].values[:,lat_long_mask].T,
                                       {'units':'hPa'}),
                          "CLW":(["time","height"], 
                                 prior_all['CLW'].values[:,lat_long_mask].T,
                                 {'units':'kg/kg'}),
                          "Rain":(["time","height"], 
                                  prior_all['Rain'].values[:,lat_long_mask].T,
                                  {'units':'kg/(m2 *s)'}), 
                          "surfPressure":(["time"], 
                                          prior_all['surfPressure'].values[lat_long_mask],
                                          {'units':'hPa'}),
                          "surfTemperature":(["time"],
                                             prior_all['surfTemperature'].values[lat_long_mask],
                                             {'units':'K'}),
                          "Temperature2m":(["time"], 
                                           prior_all['Temperature2m'].values[lat_long_mask],
                                           {'units':'K'}),
                          "DewTemperature2m":(["time"], 
                                           prior_all['DewTemperature2m'].values[lat_long_mask],
                                           {'units':'K'}),
                          "U10":(["time"], 
                                 prior_all['U10'].values[lat_long_mask],
                                 {'units':'m/s'}),
                          "V10":(["time"], 
                                 prior_all['V10'].values[lat_long_mask],
                                 {'units':'m/s'}),
                          "W10":(["time"], 
                                 prior_all['W10'].values[lat_long_mask],
                                 {'units':'m/s'}),
                          "lat":(["time"], 
                                 prior_all['lat'].values[lat_long_mask],
                                 {'units':'deg'}),
                          "long":(["time"], 
                                  prior_all['long'].values[lat_long_mask],
                                  {'units':'deg'})}, 
                coords={"time": (["time"], time_index), 
                        "height": (["height"], prior_all['height'].values)},
                
               )
    return prior_local


# **************************************

def timestamp2datetime(t):
    # Converts the integer "t" (timestamp or seconds since 1970-01-01)
    # in a datetime object, taking into account the leap seconds (i.e.
    # a second difference is still measured). This function is 
    # necessary in our application, because we are using 
    # a time array with differences of 1 second between the elements
    # the method "fromtimestamp() from "datetime" ignores the leap seconds
    g = time.localtime(t)
    c = datetime(year=g.tm_year,month=g.tm_mon,day=g.tm_mday,
                 hour=g.tm_hour,minute=g.tm_min,second=g.tm_sec)
    return(c)




def datetime64_to_datetime(dt64):
# input time dt64 is in numpy's datetime64 format
# output time 'dt' is in datetime format
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    dt = datetime.utcfromtimestamp(ts)

    return dt


# Routines from PyOpEst (Maahn) that Mario modified to use in our application:

def splitX(x):
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    w_index = [i for i in x.index if i.endswith('nw')]

    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('q')]
    s_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nw')]

    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert (len(t_index)*2 +len(w_index)) == len(x)

    xt = x[t_index]
    xt.index = h_index

    xq = x[q_index]
    xq.index = h_index

    xw = x[w_index]
    xw.index = s_index

    xt.index.name = 'height'
    xq.index.name = 'height'
    xw.index.name = 'height'

    return xt, xq, xw

# *****************************************************

def splitXUV(x):
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    u_index = [i for i in x.index if i.endswith('nu')]
    v_index = [i for i in x.index if i.endswith('nv')]  

    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('q')]
    su_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nu')]
    sv_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nv')]

    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert (len(t_index)*2 +len(u_index)+len(v_index)) == len(x)

    xt = x[t_index]
    xt.index = h_index

    xq = x[q_index]
    xq.index = h_index

    xu = x[u_index]
    xu.index = su_index

    xv = x[v_index]
    xv.index = sv_index

    xt.index.name = 'height'
    xq.index.name = 'height'
    xu.index.name = 'height'
    xv.index.name = 'height'

    return xt, xq, xu, xv

# *****************************************************

def splitQX(x):
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    w_index = [i for i in x.index if i.endswith('nw')]

    h_index_T = [float(i.split('_')[0]) for i in x.index if i.endswith('t')]
    h_index_Q = [float(i.split('_')[0]) for i in x.index if i.endswith('q')]
    s_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nw')]

    #assert len(t_index) == len(q_index)
    #assert len(t_index) == len(h_index)
    assert (len(t_index) + len(q_index) +len(w_index)) == len(x)

    xt = x[t_index]
    xt.index = h_index_T

    xq = x[q_index]
    xq.index = h_index_Q

    xw = x[w_index]
    xw.index = s_index

    xt.index.name = 'height'
    xq.index.name = 'height'
    xw.index.name = 'height'

    return xt, xq, xw

# *****************************************************

def splitX_all(x):
    # State vector
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    u_index = [i for i in x.index if i.endswith('nu')]
    v_index = [i for i in x.index if i.endswith('nv')]
    # State vector indices
    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('t')]
    su_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nu')]
    sv_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nv')]

    # Parameters vector:
    bp2m_index = [i for i in x.index if i.endswith('bp2m')] 
    bt2m_index = [i for i in x.index if i.endswith('bt2m')]
    #bh2m_index = [i for i in x.index if i.endswith('bh2m')] 
    btsk_index = [i for i in x.index if i.endswith('btsk')]  
    # Parameters vector indices:
    sbp2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('bp2m')]
    sbt2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('bt2m')]  
    #sbh2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('bh2m')]
    sbtsk_index = [float(i.split('_')[0]) for i in x.index if i.endswith('btsk')]


    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert (len(t_index)*2 +len(u_index)+
                            len(v_index)+
                            len(bp2m_index)+
                            len(bt2m_index)+
                            len(btsk_index) ) == len(x)
                            #len(bh2m_index)+len(btsk_index) ) == len(x)
    # State vector output
    xt = x.loc[t_index] #x[t_index]
    xt.index = h_index
    xq = x.loc[q_index]  #x[q_index]
    xq.index = h_index
    xu = x.loc[u_index]  #x[u_index]
    xu.index = su_index
    xv = x.loc[v_index]  #x[v_index]
    xv.index = sv_index
    # Parameters vector output:
    xbp2m = x.loc[bp2m_index]  #x[bp2m_index]
    xbp2m.index = sbp2m_index 
    xbt2m = x.loc[bt2m_index]  #x[bt2m_index]
    xbt2m.index = sbt2m_index
    #xbh2m = x[bh2m_index]
    #xbh2m.index = sbh2m_index
    xbtsk = x.loc[btsk_index]  #x[btsk_index]
    xbtsk.index = sbtsk_index

    xt.index.name = 'height'
    xq.index.name = 'height'
    xu.index.name = 'height'
    xv.index.name = 'height'
    xbp2m.index.name = 'height'
    xbt2m.index.name = 'height'
    #xbh2m.index.name = 'height'
    xbtsk.index.name = 'height'

    return xt, xq, xu, xv, xbp2m, xbt2m, xbtsk
   
# *****************************************************

def splitX_all_2(x):

    # State vector
    skt_index = [i for i in x.index if i.endswith('skt')]  
    u_index = [i for i in x.index if i.endswith('u10')]
    v_index = [i for i in x.index if i.endswith('v10')]
    # State vector indices   
    su_index = [float(i.split('_')[0]) for i in x.index if i.endswith('u10')]
    sv_index = [float(i.split('_')[0]) for i in x.index if i.endswith('v10')]
    sskt_index = [float(i.split('_')[0]) for i in x.index if i.endswith('skt')]

    # Parameters vector:
    t_index = [i for i in x.index if i.endswith('temp')]
    q_index = [i for i in x.index if i.endswith('hum')]   
    t2m_index = [i for i in x.index if i.endswith('t2m')]
    sp_index = [i for i in x.index if i.endswith('sp')]
    
    # Parameters vector indices:
    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('temp')]
    st2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('t2m')]  
    ssp_index = [float(i.split('_')[0]) for i in x.index if i.endswith('sp')]

    
    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert (len(t_index)*2 +len(u_index)+
                            len(v_index)+
                            len(t2m_index)+
                            len(sp_index)+
                            len(sskt_index) ) == len(x)

    # State vector output

    xtsk = x[skt_index]
    xtsk.index = sskt_index   
    xu = x[u_index]
    xu.index = su_index
    xv = x[v_index]
    xv.index = sv_index
    
    # Parameters vector output:

    xt = x[t_index]
    xt.index = h_index
    xq = x[q_index]
    xq.index = h_index 
    xt2m = x[t2m_index]
    xt2m.index = st2m_index
    xsp = x[sp_index]
    xsp.index = ssp_index           

    xu.index.name = 'pressure'
    xv.index.name = 'pressure'
    xtsk.index.name = 'pressure'
    xt.index.name = 'pressure'
    xq.index.name = 'pressure'    
    xt2m.index.name = 'pressure'
    xsp.index.name = 'pressure'

    return xt, xq, xu, xv, xt2m, xtsk, xsp    

# *****************************************************

# *****************************************************


def splitXW_all(x):
    # State vector
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    w_index = [i for i in x.index if i.endswith('nw')]
    # State vector indices
    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('t')]
    sw_index = [float(i.split('_')[0]) for i in x.index if i.endswith('nw')]

    # Parameters vector:
    bp2m_index = [i for i in x.index if i.endswith('bp2m')] 
    bt2m_index = [i for i in x.index if i.endswith('bt2m')]
    #bh2m_index = [i for i in x.index if i.endswith('bh2m')] 
    btsk_index = [i for i in x.index if i.endswith('btsk')]  
    # Parameters vector indices:
    sbp2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('bp2m')]
    sbt2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('bt2m')]  
    #sbh2m_index = [float(i.split('_')[0]) for i in x.index if i.endswith('bh2m')]
    sbtsk_index = [float(i.split('_')[0]) for i in x.index if i.endswith('btsk')]


    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert (len(t_index)*2 +len(w_index)+
                            len(bp2m_index)+
                            len(bt2m_index)+
                            len(btsk_index) ) == len(x)
                            #len(bh2m_index)+len(btsk_index) ) == len(x)
    # State vector output
    xt = x[t_index]
    xt.index = h_index
    xq = x[q_index]
    xq.index = h_index
    xw = x[w_index]
    xw.index = sw_index

    # Parameters vector output:
    xbp2m = x[bp2m_index]
    xbp2m.index = sbp2m_index 
    xbt2m = x[bt2m_index]
    xbt2m.index = sbt2m_index
    #xbh2m = x[bh2m_index]
    #xbh2m.index = sbh2m_index
    xbtsk = x[btsk_index]
    xbtsk.index = sbtsk_index

    xt.index.name = 'height'
    xq.index.name = 'height'
    xw.index.name = 'height'

    xbp2m.index.name = 'height'
    xbt2m.index.name = 'height'
    #xbh2m.index.name = 'height'
    xbtsk.index.name = 'height'

    return xt, xq, xw, xbp2m, xbt2m, xbtsk

# *****************************************************

def plotMwrResultsX(oe1, title=None, oe2=None, title2=None, oe3=None, title3=None, h=None, hlabel='Height [m]', xlimT=(None, None), xlimH=(None, None)):

    if oe2 is None:
        gridspec = dict(wspace=0.0)
        fig, (axA, axB) = plt.subplots(ncols=2, sharey=True,
                                       gridspec_kw=gridspec, figsize=[5.0, 4.0])
        vals = [oe1], [axA], [axB], [title]
    elif oe3 is None:

        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.25, 1, 1])
        fig, (axA, axB, ax0, axC, axD) = plt.subplots(
            ncols=5, sharey=True, figsize=[10.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2], [axA, axC], [axB, axD], [title, title2]
        ax0.set_visible(False)
    else:

        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.1, 1, 1, 0.1, 1, 1])
        fig, (axA, axB, ax0, axC, axD, ax1, axE, axF) = plt.subplots(
            ncols=8, sharey=True, figsize=[12.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2, oe3], [axA, axC, axE], [
            axB, axD, axF], [title, title2, title3]
        ax0.set_visible(False)
        ax1.set_visible(False)

    for oe, ax1, ax2, tit in zip(*vals):

        t_op, q_op, _,_,_,_,_ = splitXW_all(oe.x_op)
        t_op_err, q_op_err, _,_,_,_,_ = splitXW_all(oe.x_op_err)
        t_a, q_a, _,_,_,_,_ = splitXW_all(oe.x_a)
        t_a_err, q_a_err, _,_,_,_,_ = splitXW_all(oe.x_a_err)
        t_truth, q_truth, _,_,_,_,_ = splitXW_all(oe.x_truth)

        nProf = len(t_op)

        if h is None:
            hvar = t_op.index
        else:
            hvar = h

        ax1.plot(t_op, hvar, color='C0', label='Optimal')
        ax1.fill_betweenx(hvar, t_op+t_op_err, t_op-t_op_err,
                          color='C0', alpha=0.2)

        ax1.plot(t_a, hvar, color='C1', label='Prior')
        ax1.fill_betweenx(hvar, t_a+t_a_err, t_a-t_a_err,
                          color='C1', alpha=0.2)
        ax1.plot(t_truth, hvar, color='C2', label='Truth')

        ax2.plot(q_op, hvar, color='C0')
        ax2.fill_betweenx(hvar, q_op+q_op_err, q_op-q_op_err,
                          color='C0', alpha=0.2)

        ax2.plot(q_a, hvar, color='C1')
        ax2.fill_betweenx(hvar, q_a+q_a_err, q_a-q_a_err,
                          color='C1', alpha=0.2)
        ax2.plot(q_truth, hvar, color='C2')

        ax1.set_xlabel('Temperature [K]')
        ax2.set_xlabel('Specific humidity\n[log$_{10}$(g/kg)]')

        ax1.set_xlim(xlimT)
        ax2.set_xlim(xlimH)
        
        ax1.grid(True)
        ax2.grid(True)

        ax1.set_title(tit, loc='left')

    if h is not None:
        axA.invert_yaxis()

    axA.set_ylabel(hlabel)

    axA.legend(loc='upper right')

    return fig

# *****************************************

def plotMwrResultsQX(oe1, title=None, oe2=None, title2=None, oe3=None, title3=None, h=None, hlabel='Height [m]', xlimT=(None, None), xlimH=(None, None)):

    if oe2 is None:
        gridspec = dict(wspace=0.0)
        fig, (axA, axB) = plt.subplots(ncols=2, sharey=True,
                                       gridspec_kw=gridspec, figsize=[5.0, 4.0])
        vals = [oe1], [axA], [axB], [title]
    elif oe3 is None:

        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.25, 1, 1])
        fig, (axA, axB, ax0, axC, axD) = plt.subplots(
            ncols=5, sharey=True, figsize=[10.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2], [axA, axC], [axB, axD], [title, title2]
        ax0.set_visible(False)
    else:

        gridspec = dict(wspace=0.0, width_ratios=[1, 1, 0.1, 1, 1, 0.1, 1, 1])
        fig, (axA, axB, ax0, axC, axD, ax1, axE, axF) = plt.subplots(
            ncols=8, sharey=True, figsize=[12.0, 4.0], gridspec_kw=gridspec)
        vals = [oe1, oe2, oe3], [axA, axC, axE], [
            axB, axD, axF], [title, title2, title3]
        ax0.set_visible(False)
        ax1.set_visible(False)

    for oe, ax1, ax2, tit in zip(*vals):

        t_op, q_op, w_op = splitQX(oe.x_op)
        t_op_err, q_op_err, w_op_err = splitQX(oe.x_op_err)
        t_a, q_a, w_a = splitQX(oe.x_a)
        t_a_err, q_a_err, w_a_err = splitQX(oe.x_a_err)
        t_truth, q_truth, w_truth = splitQX(oe.x_truth)

        nProf = len(t_op)

        if h is None:
            hvar = t_op.index
            hvar_Q = q_op.index
        else:
            hvar = h
            

        ax1.plot(t_op, hvar, color='C0', label='Optimal')
        ax1.fill_betweenx(hvar, t_op+t_op_err, t_op-t_op_err,
                          color='C0', alpha=0.2)

        ax1.plot(t_a, hvar, color='C1', label='Prior')
        ax1.fill_betweenx(hvar, t_a+t_a_err, t_a-t_a_err,
                          color='C1', alpha=0.2)
        ax1.plot(t_truth, hvar, color='C2', label='Truth')

        ax2.plot(q_op, hvar_Q, color='C0')
        ax2.fill_betweenx(hvar_Q, q_op+q_op_err, q_op-q_op_err,
                          color='C0', alpha=0.2)

        ax2.plot(q_a, hvar_Q, color='C1')
        ax2.fill_betweenx(hvar_Q, q_a+q_a_err, q_a-q_a_err,
                          color='C1', alpha=0.2)
        ax2.plot(q_truth, hvar_Q, color='C2')

        ax1.set_xlabel('Temperature [K]')
        ax2.set_xlabel('Specific humidity\n[log$_{10}$(g/kg)]')

        ax1.set_xlim(xlimT)
        ax2.set_xlim(xlimH)
        
        ax1.grid(True)
        ax2.grid(True)

        ax1.set_title(tit, loc='left')

    if h is not None:
        axA.invert_yaxis()

    axA.set_ylabel(hlabel)

    axA.legend(loc='upper right')

    return fig

# **************************************

def priors2seasons(prior_local_1):

    # Splitting the priors per season:

    priors = {'all': prior_local_1}

    seasons = [
        'DJF',
        'MAM',
        'JJA',
        'SON',
    ]
    months = (
        [12, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    )
    h_conc = [] # added by mario to create wind dataframe with season column

    for season, month in zip(seasons, months):
        h = np.where(np.isin(prior_local_1['time.month'], month))
        priors[season] = prior_local_1.isel({'time': np.array(h).reshape(-1)}).dropna(
            'time', how='all')
        # added by mario to create wind dataframe with season column
        h_conc = np.concatenate((h_conc, np.array(h).reshape(-1)),axis=0) 
    
    # added by mario to create wind dataframe with season column
    h_season = np.empty(len(h_conc),dtype=object)
    for season, month in zip(seasons, months):
        h = np.where(np.isin(prior_local_1['time.month'], month))
        h_season[np.array(h).reshape(-1)[:]] = season

    pressure = []
    for k in ['all'] + seasons:
        pressure.append(priors[k].Pressure.mean('time'))
    pressure = xr.concat(pressure, dim='season')
    pressure['season'] = ['all'] + seasons
    pressure = pressure.to_pandas()    
    
    return priors, pressure, seasons, months, h_season

# *****************************************************

def priors2pandas(priors, wind_components = False, h_season = None):

    prior_qts = {}

    for season in priors.keys():
        prior_t = priors[season]['Temperature'].to_pandas()
        prior_t.columns = ['%05i_t' % (i) for i in prior_t.columns]
        prior_q = priors[season]['Humidity'].to_pandas()
        prior_q.columns = ['%05i_q' % (i) for i in prior_q.columns]

        prior_w = pd.DataFrame({10.0: priors[season]['W10'].values[:]}, 
                                index=priors[season]['time'].values[:])  
        prior_w.columns = ['%05i_nw' % (i) for i in prior_w.columns]

        prior_u = pd.DataFrame({10.0: priors[season]['U10'].values[:]}, 
                                index=priors[season]['time'].values[:])  
        prior_u.columns = ['%05i_nu' % (i) for i in prior_u.columns]

        prior_v = pd.DataFrame({10.0: priors[season]['V10'].values[:]}, 
                                index=priors[season]['time'].values[:])  
        prior_v.columns = ['%05i_nv' % (i) for i in prior_v.columns]

        prior_sp = pd.DataFrame({2.0:priors[season]['surfPressure'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_sp.columns = ['%05i_bp2m' % (i) for i in prior_sp.columns]

        prior_t2m = pd.DataFrame({2.0:priors[season]['Temperature2m'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_t2m.columns = ['%05i_bt2m' % (i) for i in prior_t2m.columns]

        # prior 2m humidity, check "spec_humidity(...)" for documentation
        prior_h2m = pd.DataFrame({2.0:np.log10( spec_humidity(priors[season]['surfPressure'].values[:],
                            priors[season]['DewTemperature2m'].values[:],latent='water') ) }, 
                            index=priors[season]['time'].values[:])  
        prior_h2m.columns = ['%05i_bh2m' % (i) for i in prior_h2m.columns]
    
        prior_ts = pd.DataFrame({0.0:priors[season]['surfTemperature'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_ts.columns = ['%05i_btsk' % (i) for i in prior_ts.columns]    


        if wind_components:
            prior_qts[season] = pd.concat((prior_t, prior_q, prior_u, prior_v), axis=1)
        else:
            prior_qts[season] = pd.concat((prior_t, prior_q, prior_w), axis=1)
    
    if h_season is None:
        prior_wind = []
    else:
        
        if wind_components:     
            prior_wind = pd.DataFrame({'00010_nw':priors['all']['W10'].values[:],
                           '00010_nu':priors['all']['U10'].values[:],
                           '00010_nv':priors['all']['V10'].values[:]}, 
                            index=priors['all']  ['time'].values[:])
        else:
            prior_wind = pd.DataFrame({10.0:priors['all']['W10'].values[:]}, 
                                  index=priors['all']['time'].values[:])
            prior_wind.columns = ['%05i_nw' % (i) for i in prior_wind.columns]

        prior_wind['season'] = h_season

    return prior_qts, prior_wind

# ******************************************************

def b_fromPrior(priors):

    prior_b = {}
    for season in priors.keys():

        prior_sp = pd.DataFrame({2.0:priors[season]['surfPressure'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_sp.columns = ['%05i_bp2m' % (i) for i in prior_sp.columns]

        prior_t2m = pd.DataFrame({2.0:priors[season]['Temperature2m'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_t2m.columns = ['%05i_bt2m' % (i) for i in prior_t2m.columns]

        # prior 2m humidity, check "spec_humidity(...)" for documentation
        prior_h2m = pd.DataFrame({2.0:np.log10( spec_humidity(priors[season]['surfPressure'].values[:],
                            priors[season]['DewTemperature2m'].values[:],latent='water') ) }, 
                            index=priors[season]['time'].values[:])  
        prior_h2m.columns = ['%05i_bh2m' % (i) for i in prior_h2m.columns]
    
        prior_ts = pd.DataFrame({0.0:priors[season]['surfTemperature'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_ts.columns = ['%05i_btsk' % (i) for i in prior_ts.columns]    
    
        prior_b[season] = pd.concat((prior_sp, prior_t2m, prior_h2m, prior_ts), axis=1)

    return prior_b

# *****************************************************

def priors2Pandas(priors, flavor = 1, h_season = None):

    prior_xa = {} # a-priori variables
    prior_b = {}  # b parameters for the forward model
    for season in priors.keys():
        prior_t = priors[season]['Temperature'].to_pandas()
        prior_t.columns = ['%05i_t' % (i) for i in prior_t.columns]
        prior_q = priors[season]['Humidity'].to_pandas()
        prior_q.columns = ['%05i_q' % (i) for i in prior_q.columns]

        prior_w = pd.DataFrame({10.0: priors[season]['W10'].values[:]}, 
                                index=priors[season]['time'].values[:])  
        prior_w.columns = ['%05i_nw' % (i) for i in prior_w.columns]

        prior_u = pd.DataFrame({10.0: priors[season]['U10'].values[:]}, 
                                index=priors[season]['time'].values[:])  
        prior_u.columns = ['%05i_nu' % (i) for i in prior_u.columns]

        prior_v = pd.DataFrame({10.0: priors[season]['V10'].values[:]}, 
                                index=priors[season]['time'].values[:])  
        prior_v.columns = ['%05i_nv' % (i) for i in prior_v.columns]

        prior_sp = pd.DataFrame({2.0:priors[season]['surfPressure'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_sp.columns = ['%05i_bp2m' % (i) for i in prior_sp.columns]

        prior_t2m = pd.DataFrame({2.0:priors[season]['Temperature2m'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_t2m.columns = ['%05i_bt2m' % (i) for i in prior_t2m.columns]

    
        prior_ts = pd.DataFrame({0.0:priors[season]['surfTemperature'].values[:]}, 
                            index=priors[season]['time'].values[:])  
        prior_ts.columns = ['%05i_btsk' % (i) for i in prior_ts.columns]    

        
        if flavor == 1:
           prior_xa[season] = pd.concat((prior_w, prior_ts), axis=1)
           prior_b[season] = pd.concat((prior_t, prior_q, prior_sp, prior_t2m), axis=1)    
        elif flavor == 2:
           prior_xa[season] = pd.concat((prior_u, prior_v, prior_ts), axis=1)
           prior_b[season] = pd.concat((prior_t, prior_q, prior_sp, prior_t2m), axis=1) 
        elif flavor == 3:
           prior_xa[season] = pd.concat((prior_t, prior_q, prior_w), axis=1)
           prior_b[season] = pd.concat(( prior_sp, prior_t2m, prior_ts), axis=1) 
        elif flavor == 4: 
           prior_xa[season] = pd.concat((prior_t, prior_q, prior_u, prior_v), axis=1)
           prior_b[season] = pd.concat(( prior_sp, prior_t2m, prior_ts), axis=1) 
        else: 
            print('Error splitting a-priori data in Pandas datasets')
            print('Splitting by default:')
            print('Parameters:  surf. 2m pressure, surf. 2m temp, skin temp.')
            print('State vector: temp. profile, hum. profile, u & v 10m wind speed')
            prior_xa[season] = pd.concat((prior_t, prior_q, prior_u, prior_v), axis=1)
            prior_b[season] = pd.concat(( prior_sp, prior_t2m, prior_ts), axis=1) 

    
    if h_season is None:
        prior_wind = []
    else:
        prior_wind = []
        #if wind_components:     
        #    prior_wind = pd.DataFrame({'00010_nw':priors['all']['W10'].values[:],
        #                   '00010_nu':priors['all']['U10'].values[:],
        #                   '00010_nv':priors['all']['V10'].values[:]}, 
        #                    index=priors['all']  ['time'].values[:])
        #else:
        #    prior_wind = pd.DataFrame({10.0:priors['all']['W10'].values[:]}, 
        #                          index=priors['all']['time'].values[:])
        #    prior_wind.columns = ['%05i_nw' % (i) for i in prior_wind.columns]
        #
        #prior_wind['season'] = h_season

    return prior_xa, prior_b

# ***************************************************

def meanCov(prior_qts, seasons):

    x_cov = []
    x_mean = []
    
    for season in ['all'] + seasons:

        x_cov1 = prior_qts[season].cov().rename_axis('state', axis=0).rename_axis('stateT', axis=1) 
        x_mean1 = prior_qts[season].mean().rename_axis('state', axis=0)
        
        x_cov.append(xr.DataArray(x_cov1))
        x_mean.append(xr.DataArray(x_mean1))

    x_cov = xr.concat(x_cov, dim='season')
    x_mean = xr.concat(x_mean, dim='season')

    x_cov['season'] = ['all'] + seasons
    x_mean['season'] = ['all'] + seasons

    return x_cov, x_mean

# ****************************************************

def createTrueState(profiles, flavor=1):

    profiles_t = profiles['Temperature'].to_pandas()
    profiles_t.columns = ['%05i_t' % (i) for i in profiles_t.columns]
    profiles_q = profiles['Humidity'].to_pandas()
    profiles_q.columns = ['%05i_q' % (i) for i in profiles_q.columns]

    profiles_w = pd.DataFrame({10.0: profiles['W10'].values[:]}, index=profiles['time'].values[:])
    profiles_w.columns.names=['height']
    profiles_w.columns = ['%05i_nw' % (i) for i in profiles_w.columns]

    profiles_u = pd.DataFrame({10.0: profiles['U10'].values[:]}, index=profiles['time'].values[:])
    profiles_u.columns.names=['height']
    profiles_u.columns = ['%05i_nu' % (i) for i in profiles_u.columns]

    profiles_v = pd.DataFrame({10.0: profiles['V10'].values[:]}, index=profiles['time'].values[:])
    profiles_v.columns.names=['height']
    profiles_v.columns = ['%05i_nv' % (i) for i in profiles_v.columns]

    # Parameters vector:

    profiles_bp2m = pd.DataFrame({2.0: profiles['surfPressure'].values[:]},
                                 index=profiles['time'].values[:])
    profiles_bp2m.columns.names=['height']
    profiles_bp2m.columns = ['%05i_bp2m' % (i) for i in profiles_bp2m.columns]

    profiles_bt2m = pd.DataFrame({2.0:profiles['Temperature2m'].values[:]},
                                 index=profiles['time'].values[:])
    profiles_bt2m.columns.names=['height']
    profiles_bt2m.columns = ['%05i_bt2m' % (i) for i in profiles_bt2m.columns]

    profiles_bh2m = pd.DataFrame({2.0: np.log10( spec_humidity( 
                                                 profiles['surfPressure'].values[:],
                                                 profiles['DewTemperature2m'].values[:],latent='water') 
                                               )}, 
                                  index=profiles['time'].values[:])
    profiles_bh2m.columns.names=['height']
    profiles_bh2m.columns = ['%05i_bh2m' % (i) for i in profiles_bh2m.columns]

    profiles_btsk = pd.DataFrame({0.0: profiles['surfTemperature'].values[:]},     
                                 index=profiles['time'].values[:])
    profiles_btsk.columns.names=['height']
    profiles_btsk.columns = ['%05i_btsk' % (i) for i in profiles_btsk.columns] 


    if flavor == 1 or flavor == 3:
   
       x_truths = pd.concat((profiles_t, profiles_q, profiles_w,
                                 profiles_bp2m, profiles_bt2m, profiles_btsk), 1)
    elif flavor == 2 or flavor == 4:

       x_truths = pd.concat((profiles_t, profiles_q, profiles_u, profiles_v,
                                 profiles_bp2m, profiles_bt2m, profiles_btsk), 1)
    else: 
        print('Error defining x_true for synthetic data creation')
        print('True state by default:')
        print('Parameters:  surf. 2m pressure, surf. 2m temp, skin temp.')
        print('State vector: temp. profile, hum. profile, u & v 10m wind speed')
        
        x_truths = pd.concat((profiles_t, profiles_q, profiles_u, profiles_v,
                                 profiles_bp2m, profiles_bt2m, profiles_btsk), 1)

    x_truths.columns.name = 'state'
    x_truths.index.name = 'time'

    assert np.all(np.isfinite(x_truths))

    return x_truths




# **************************************************************

def UV2Wvar(varU,varV,covarUV,u,v):
    # Uncertainty propagation:
    # https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty

    w = np.sqrt(u**2+v**2) # wind speed from components (Y = f(X1,X2) ==> w = f(u,v))

    sigmaW = np.sqrt((1/(w**2))*(varU*(u)**2 + varV*(v)**2 + 2*u*v*covarUV))  # Standard deviation   

    return w, sigmaW # returns wind and sigma

# ****************************************************************

def generate_masked_array(xarray, mask, threshold, operator, drop=True):
# Routine from: EUMETSAT Online Short Course #4 (2020):
# "Summer(s) of fires - The 2020 wildfires in Australia, Siberia and California"
# Author: (probably) Julia Wagemann


    """ 
    Applies a mask (e.g. cloud fraction values or masking out certain data values) onto a given data array, based on a given threshold.
    
    Parameters:
        xarray (xarray DataArray): a three-dimensional xarray DataArray object
        mask (xarray DataArray): 1-dimensional xarray DataArray, e.g. cloud fraction values
        threshold (float): any number between 0 and 1, specifying the degree of cloudiness which is acceptable
        operator (str): operator how to mask the array, e.g. '<', '>' or '='
        drop(boolean): whether to drop the values that are masked out. Default is True.
        
    Returns:
        Masked xarray DataArray with flagged negative values
    """
    if(operator=='<'):
        cloud_mask = xr.where(mask < threshold, 1, 0) #Generate cloud mask with value 1 for the pixels we want to keep
    elif(operator=='!='):
        cloud_mask = xr.where(mask != threshold, 1, 0)
    elif(operator=='>'):
        cloud_mask = xr.where(mask > threshold, 1, 0)
    else:
        cloud_mask = xr.where(mask == threshold, 1, 0)
            
    xarray_masked = xr.where(cloud_mask ==1, xarray, np.nan) #Apply mask onto the DataArray
    print(xarray_masked)
    xarray_masked.attrs = xarray.attrs #Set DataArray attributes 
    if(drop):
        return xarray_masked[~np.isnan(xarray_masked)] #Return masked DataArray and flag negative values
    else:
        return xarray_masked


# ****************************************************88
def date2season(adate):
    seasons = [
        'DJF',
        'MAM',
        'JJA',
        'SON',
    ]
    months = (
        [12, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    )
    for season, month in zip(seasons, months):
        if(np.isin(adate.month, month)):
            break;
   
    return season


