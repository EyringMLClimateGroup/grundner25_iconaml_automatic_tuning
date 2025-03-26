#!/usr/bin/env python
# coding: utf-8

# Import libraries
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd

from subprocess import call
from scipy.optimize import minimize

# AG (8/13/24):
# - Now loading 3-hourly ICON data as a reference to be able to always compute the variability we expect in short sims
# - For relative measure: Divide by the difference instead of the upper bound (min max feature scaling)
# - Difference between max/mean/min instead of standard deviation

# Idea: When the LOSS falls under the given threshold 'thr', then increase ICON_SIM_LENGTH to a week, later to a month, a year.
loss_tol = 1e-5

# Get command to run ICON run script
EXPNAME = sys.argv[1]
ICON_JOB = "exp." + EXPNAME + ".run"
ICON_SIM_LENGTH = sys.argv[2]
INIT_PARAMS = sys.argv[3]
RELAX_BOUNDS = 1 # 1 by default
RELAX_BOUNDS_SE_clt = 1 # 1 by default
RELAX_BOUNDS_net_toa = 1 # 1 by default

# Define clt box (here: -45 to -15 deg lat, -105 to -75 deg lon, south east pacific):
box_lat_min = -0.785398163397
box_lat_max = -0.2617993878
box_lon_min = -1.83259571459
box_lon_max = -1.308996939

# Add more metrics when necessary
OBJECTIVE_METRICS = ['rsut', 'rlut', 'net_toa', 'clt', 'pr', 'cllvi', 'clivi', 'prw', 'swcre', 'lwcre', 'clt_SE_pac']

def compute_relax_bounds(DS_avrg):
    # Take averages over ICON_SIM_LENGTH to be able to quantify how much we expect our variable of interest varies in our short ICON-ML simulations
    dates = DS_avrg['time'].values # Given floating point dates
    date_int = dates.astype(int) # Separate the integer part (YYYYMMDD) and fractional part (day fractions)
    fractional_part = dates - date_int
    date_dt = pd.to_datetime(date_int.astype(str), format='%Y%m%d') # Convert integer part to string and then to datetime
    fractional_timedelta = pd.to_timedelta(fractional_part, unit='D') # Convert fractional part to timedeltas (fraction of a day)
    datetime_index = pd.DatetimeIndex(date_dt + fractional_timedelta) # Combine the date and fractional timedelta and convert to DatetimeIndex 
    DS_avrg['time'] = datetime_index
    if ICON_SIM_LENGTH == 'day':
        DS_avrg = DS_avrg.resample(time='D').mean()
    elif ICON_SIM_LENGTH == 'week':
        DS_avrg = DS_avrg.resample(time='W').mean()
    elif ICON_SIM_LENGTH == 'month':
        DS_avrg = DS_avrg.resample(time='M').mean()
    elif ICON_SIM_LENGTH == 'halfyear':
        DS_avrg = DS_avrg.resample(time='6M').mean()
    elif ICON_SIM_LENGTH == 'year':
        DS_avrg = DS_avrg.resample(time='Y').mean()
    relax_max = DS_avrg.max() - DS_avrg.mean()
    relax_min = DS_avrg.mean() - DS_avrg.min()
    return relax_max, relax_min

# Compute bounds for the objective function. 
def define_bounds():
    # Observation bounds extracted from https://swift.dkrz.de/v1/dkrz_adf90815-96ce-4411-81ae-4db5e76adcbc/esmvaltool_output/index.html.
    obs_bounds = {}
    obs_bounds['rsut_min'] = 100 # 98
    obs_bounds['rsut_max'] = 110 # 108
    obs_bounds['clt_min'] = 0.62 # Lauer et al., 2023, Table 5
    obs_bounds['clt_max'] = 0.67 # Lauer et al., 2023, Table 5
    obs_bounds['pr_min'] = 2.7/(24*60*60) # in kg m-2 s-1
    obs_bounds['pr_max'] = 3/(24*60*60)   # in kg m-2 s-1
    obs_bounds['cllvi_min'] = 0.036 # Lauer et al., 2023, Table 5
    obs_bounds['cllvi_max'] = 0.105 # Lauer et al., 2023, Table 5
    obs_bounds['clivi_min'] = 0.036 # Lauer et al., 2023, Table 5
    obs_bounds['clivi_max'] = 0.061 # Lauer et al., 2023, Table 5
    obs_bounds['rlut_min'] = 231 # Added on 24/09/13, after 12610632
    obs_bounds['rlut_max'] = 242 # Added on 24/09/13, after 12610632
    obs_bounds['net_toa_min'] = 0.5 # Added on 24/09/13, after 12610632; total net downward radiation rsdt - rsut - rlut
    obs_bounds['net_toa_max'] = 0.9 # Added on 24/09/13, after 12610632; total net downward radiation rsdt - rsut - rlut
    obs_bounds['prw_min'] = 25 # Lauer et al., 2023, Table 5
    obs_bounds['prw_max'] = 26 # Lauer et al., 2023, Table 5
    obs_bounds['swcre_min'] = -56 # Lauer et al., 2023, Table 5
    obs_bounds['swcre_max'] = -46 # Lauer et al., 2023, Table 5
    obs_bounds['lwcre_min'] = 25 # Lauer et al., 2023, Table 5
    obs_bounds['lwcre_max'] = 28 # Lauer et al., 2023, Table 5
    # Could be added: Surface wind stress tauu & zonal means?

    objective_bounds = {}
    if ICON_SIM_LENGTH in ['day', 'week']:
        # Has three-hourly averaged output
        DS = xr.open_mfdataset('/work/bd1179/b309170/icon-ml_models/icon-a-ml/experiments/ag_atm_amip_r2b5_cvtfall_entrmid_crt_05_3hourly/*atm_2d_ml*.nc')
    elif ICON_SIM_LENGTH in ['month', 'halfyear', 'year']:
        # Has monthly averaged output
        DS = xr.open_mfdataset('/work/bd1179/b309170/icon-ml_models/icon-a-ml/experiments/ag_atm_amip_r2b5_cvtfall_entrmid_crt_05_20yrs/*atm_2d_ml*.nc')
    # Assign special metrics
    DS = DS.assign( swcre=DS["rsutcs"]-DS["rsut"] )
    DS = DS.assign( lwcre=DS["rlutcs"]-DS["rlut"] )
    DS = DS.assign( net_toa=DS["rsdt"]-DS["rsut"]-DS["rlut"] )
    DS_avrg = DS.sel(time=slice(19790102, 19981231)).mean('ncells') # Global averages, discard first timestep
    # Relax upper/lower bounds considering the variability of our metric in short simulations
    relax_max, relax_min = compute_relax_bounds(DS_avrg)
    for metric in OBJECTIVE_METRICS:
        if metric != 'clt_SE_pac':
            relax_max_val = getattr(relax_max, metric).values # Just one value. Convert it to numpy.
            relax_min_val = getattr(relax_min, metric).values
        elif metric == 'clt_SE_pac':
            # Add clt over Chilean waters (9/19/24)
            ISCCP_ref = xr.open_dataset('/home/b/b309170/bd1179_work/ISCCP/cltisccp_obs4MIPs_ISCCP_L3_V1.0_198307-200806_mean_R2B5.nc')
            ISCCP_box = xr.where( (ISCCP_ref.clat<box_lat_max)&(ISCCP_ref.clat>box_lat_min)&(ISCCP_ref.clon<box_lon_max)&(ISCCP_ref.clon>box_lon_min), ISCCP_ref, np.nan).dropna(dim='cell',how='any').mean('cell')
            obs_bounds['%s_max'%metric] = float(ISCCP_box.cltisccp_mean)/100 # is 0.7114
            obs_bounds['%s_min'%metric] = obs_bounds['%s_max'%metric]
            # Focus only on clt, discard first timestep, take only SE pacific, then average
            DS_clt = DS.clt.sel(time=slice(19790102, 19981231))
            DS_avrg = xr.where( (DS_clt.clat<box_lat_max)&(DS_clt.clat>box_lat_min)&(DS_clt.clon<box_lon_max)&(DS_clt.clon>box_lon_min), DS_clt, np.nan).dropna(dim='ncells',how='any').mean('ncells')
            relax_max_val, relax_min_val = compute_relax_bounds(DS_avrg) # Relax upper/lower bounds considering the variability of our metric in short simulations
        if metric == 'clt_SE_pac': # Trying different relax_bounds
            objective_bounds['%s_max'%metric] = obs_bounds['%s_max'%metric] + RELAX_BOUNDS_SE_clt*float(relax_max_val)
            objective_bounds['%s_min'%metric] = obs_bounds['%s_min'%metric] - RELAX_BOUNDS_SE_clt*float(relax_min_val)
        elif metric == 'net_toa':
            objective_bounds['%s_max'%metric] = obs_bounds['%s_max'%metric] + RELAX_BOUNDS_net_toa*float(relax_max_val)
            objective_bounds['%s_min'%metric] = obs_bounds['%s_min'%metric] - RELAX_BOUNDS_net_toa*float(relax_min_val)
        else:
            objective_bounds['%s_max'%metric] = obs_bounds['%s_max'%metric] + RELAX_BOUNDS*float(relax_max_val)
            objective_bounds['%s_min'%metric] = obs_bounds['%s_min'%metric] - RELAX_BOUNDS*float(relax_min_val)
    return objective_bounds

def objective_norm(objective_bounds):
    DS = xr.open_mfdataset('/scratch/b/b309170/experiments/%s_%s_%s/*atm_2d_ml*.nc'%(EXPNAME, ICON_SIM_LENGTH, INIT_PARAMS))
    # Assign special metrics
    DS = DS.assign( swcre=DS["rsutcs"]-DS["rsut"] )
    DS = DS.assign( lwcre=DS["rlutcs"]-DS["rlut"] )
    DS = DS.assign( net_toa=DS["rsdt"]-DS["rsut"]-DS["rlut"] )
    DS_avrg = DS.sel(time=slice(19790102, 19990101)).mean('ncells')

    LOSS = 0 # Compute loss
    for metric in OBJECTIVE_METRICS:
        if metric != 'clt_SE_pac':
            metric_values = getattr(DS_avrg, metric).values
        # Added on 9/19/2024: Punish errors in total cloud cover map
        elif metric == 'clt_SE_pac':
            DS_clt = DS.clt.sel(time=slice(19790102, 19990101))
            DS_avrg = xr.where( (DS_clt.clat<box_lat_max)&(DS_clt.clat>box_lat_min)&(DS_clt.clon<box_lon_max)&(DS_clt.clon>box_lon_min), DS_clt, np.nan).dropna(dim='ncells',how='any').mean('ncells')
            metric_values = DS_avrg.values
        # Punish greater values than the respective upper bound. Is a relative measure. Divide by the difference instead of the upper bound (min max feature scaling)
        LOSS += np.sum(np.maximum(metric_values - objective_bounds['%s_max'%metric], 0))/(objective_bounds['%s_max'%metric] - objective_bounds['%s_min'%metric])
        # Punish smaller values than the respective lower bound. Is a relative measure. Divide by the difference instead of the upper bound (min max feature scaling)
        LOSS += np.sum(np.maximum(objective_bounds['%s_min'%metric] - metric_values, 0))/(objective_bounds['%s_max'%metric] - objective_bounds['%s_min'%metric])
        # Print value of metric
        print('Metric %s:\n'%metric, metric_values, flush=True)
        # Print contribution to loss
        print('Loss after adding %s: %.3f'%(metric, LOSS), flush=True)
    return LOSS

def call_icon(X, objective_bounds):
    try:
        # Remove output from previous iteration to make sure that in the new iteration new output is generated
        call('rm /scratch/b/b309170/experiments/%s_%s_%s/ag_atm_amip*.nc'%(EXPNAME, ICON_SIM_LENGTH, INIT_PARAMS), shell=True)
    except: print('No data from the previous iteration had to be deleted')
    # Calling job script
    print("Call ICON job from python: ", "./"+ICON_JOB, flush=True)
    # Build call line
    cmd = "./"+ICON_JOB
    for k in range(len(X)):
        cmd = cmd + " " + str(X[k])
    call(cmd, shell=True)
    # Evaluate error
    res = objective_norm(objective_bounds)
    return res

# Executed after each minimize iteration (so after each 20 ICON simulations if we have 19 tunable parameters)
def early_stopping(Xi):
    loss = objective_norm(objective_bounds)
    print('Parameter values after the iteration:',Xi)
    print('Final loss value, testing for early stopping: %3.4f'%(loss))
    if loss < loss_tol:
        raise StopIteration("Early stopping: Loss below tolerance")
    return 0

def optimize(objective_bounds):
    # From the JAMES, 2024 publication
    abcdefghij_publication_240101 = (38.6562122, 43.53500518, 19.78403208, 1.13637902, 0.35299939, 4.04888686, 44.21730274, 2.03128527, 0.66971589, 0.6409019)
    # After eight iterations of abcdefghij-optimization w.r.t. ERA5 total cloud cover. Probably works much better. Surprisingly a worse starting point to do online tuning with!
    abcdefghij_refined_240807 = (89.4108448, 44.091932, 2.9846493e-07, 0.004366126, 2.05999313, 2.4936977, 3.374941, 0.119426, 0.109901, 0.4340517)
    # Additional parameters that need to be considered if the desired goal cannot be reached with the cloud cover equation parameters only. pr0 should be between 0.6 and 1.0.
    additional_params = (1, 2.1e-4, 2.0e-4, 4.0e-4, 2.25, 15, 0.8, 0.8, 0.4, 0.8, 20, 180, 20, 80) # (pr0, entrmid, entrpen, entrdd, cvtfall, ccraut, cinhomi, cinhoml1, cinhoml2, cinhoml3, cn1lnd, cn2lnd, cn1sea, cn2sea)
    match INIT_PARAMS:
        case 'orig':
            initial_params = abcdefghij_publication_240101 + additional_params
        case 'refined_day':
            # From log.auto_tune.12876603.o
            initial_params = (41.259656886942565, 67.8109702599885, 22.483463435229012, 1.3192779774470287, 0.35363688998917087, 4.366122438910146, 11.444942442010479, 2.5552489206743356, 0.7526545846986172, 0.6674430156349467, 1.0090331805824593, 0.00019718876503407813, 0.0002399272423143485, 0.0003463500243590359, 1.969002702771121, 10.278545049226935, 1.0151015672235233, 0.823738411797081, 0.37885296617492814, 0.8821987655771708, 18.93658471783531, 214.6122550163775, 20.73982102500709, 41.71264009829898)
        case 'refined_week':
            # From log.auto_tune.12889494.o
            initial_params = (41.259656886942565, 67.8109702599885, 23.607636606990464, 1.3192779774470287, 0.35363688998917087, 4.366122438910146, 11.444942442010479, 2.5552489206743356, 0.7526545846986172, 0.6674430156349467, 1.0090331805824593, 0.00019718876503407813, 0.0002399272423143485, 0.0003463500243590359, 1.969002702771121, 10.278545049226935, 1.0151015672235233, 0.823738411797081, 0.37885296617492814, 0.8821987655771708, 18.93658471783531, 214.6122550163775, 20.73982102500709, 41.71264009829898)
        case 'refined_month':
            # From log.auto_tune.12892616.o.
            initial_params = (40.999209779803124, 68.3394558219378, 23.484231887639147, 1.3295597852519498, 0.35639296308199575, 4.400149863408263, 11.534138730085267, 2.5751632819728894, 0.7585204066979654, 0.6649951451999998, 1.0168970920619893, 0.00019872555789959162, 0.00024179711798483726, 0.0003490493071823422, 1.9843481475548161, 10.358651006058587, 0.9151582312253206, 0.8301582264050795, 0.38180556104206254, 0.8890741916120648, 19.084167204527887, 216.2848380476869, 20.901457054208564, 42.03772802962055)
        case 'refined_month_240925':
            # As previous runs but with b noticeably increased to hopefully fix the TOA balance.
            initial_params = (40.999209779803124, 76, 23.484231887639147, 1.3295597852519498, 0.35639296308199575, 4.400149863408263, 11.534138730085267, 2.5751632819728894, 0.7585204066979654, 0.6649951451999998, 1.0168970920619893, 0.00019872555789959162, 0.00024179711798483726, 0.0003490493071823422, 1.9843481475548161, 10.358651006058587, 0.9151582312253206, 0.8301582264050795, 0.38180556104206254, 0.8890741916120648, 19.084167204527887, 216.2848380476869, 20.901457054208564, 42.03772802962055)
            
        ## In the following auto tuning chain we tested whether starting from ERA5-refined parameters would be beneficial (it wasn't) ##
        case 'refined':
            initial_params = abcdefghij_refined_240807 + additional_params
        case 'refined_day': # abcdefghij_refined_day_240909
            # From log.auto_tune.12510106.o 
            initial_params = (87.58950300, 45.25064470, 0.00000030, 0.00438131, 2.11412866, 2.55923076, 3.46363267, 0.11513616, 0.10723502, 0.44545835, 1.00523145, 0.00021231, 0.00020526, 0.00040487, 2.22300583, 14.94260500, 0.81060691, 0.82102358, 0.39746594, 0.78891072, 20.19449260, 180.28390900, 20.16690120, 73.60235790)
        case 'refined_month': # abcdefghij_refined_month_240909
            # From log.auto_tune.12510850.o
            initial_params = (89.4108448, 44.0919320, 0.000000298464930, 0.00436612600, 2.05999313, 2.49369770, 3.37494100, 0.119426000, 0.109901000, 0.434051700, 1.00000000, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_day_240913': # abcdefghij_refined_day_240913
            initial_params = (86.34907012929227, 49.19140251875169, 3.067134263904597e-07, 0.004460079158222819, 1.972580399144226, 2.593565913770661, 3.5770816507201264, 0.1122312448291119, 0.13653795919099143, 0.5013932460518511, 1.00000000, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_week_old_day_240913': # abcdefghij_refined_week_old_day_240913
            initial_params = (87.44994900134552, 48.85235245091333, 3.0142477796449966e-07, 0.00429300037050165, 2.1278215254085646, 2.492810521160025, 3.4756190049034537, 0.11752374193788537, 0.1066751225716385, 0.4635277245651046, 1.00000000, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_week_240913': # abcdefghij_refined_week_240913
            initial_params = (87.85395173716813, 50.94172758665081, 3.116677626659539e-07, 0.004508603618591463, 1.9342781940272493, 2.6520796064795746, 3.649698325081971, 0.11208789284632698, 0.13548983078257515, 0.5109332986217789, 1.00000000, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_month_240913': # abcdefghij_refined_month_240913
            initial_params = (87.85395173716813, 53.488813965983354, 3.116677626659539e-07, 0.004508603618591463, 1.9342781940272493, 2.6520796064795746, 3.649698325081971, 0.11208789284632698, 0.13548983078257515, 0.5109332986217789, 1.00000000, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_month_v2_240913': # abcdefghij_refined_month_v2_240913
            initial_params = (85.65760294373902, 51.1009204853591, 3.12641724424285e-07, 0.004522693004899556, 1.9403228133835855, 2.6603673552498233, 3.661103632347852, 0.11243816751147168, 0.13591323650377057, 0.5125299651799724, 1.0031250000000003, 0.00021065625000000012, 0.000200625, 0.00040125, 2.25703125, 15.046875, 0.8025000000000003, 0.8025000000000003, 0.40125000000000016, 0.8025000000000005, 20.0625, 180.5625, 20.0625, 80.25)
        case 'refined_month_240916': # abcdefghij_refined_month_240916
            initial_params = (87.44994900134552, 51.294970073459, 3.0142477796449966e-07, 0.00429300037050165, 2.1278215254085646, 2.492810521160025, 3.4756190049034537, 0.11752374193788537, 0.1066751225716385, 0.4635277245651046, 1.0, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_month_v2_240916': # abcdefghij_refined_month_v2_240916
            initial_params = (87.85395173716813, 67.374, 3.116677626659539e-07, 0.004508603618591463, 1.9342781940272493, 2.6520796064795746, 3.649698325081971, 0.11208789284632698, 0.13548983078257515, 0.5109332986217789, 1.0, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_year_great_toa_240919':
            initial_params = (92.24664932402654, 67.374, 3.116677626659539e-07, 0.004508603618591463, 1.9342781940272493, 2.6520796064795746, 3.649698325081971, 0.11208789284632698, 0.13548983078257515, 0.5109332986217789, 1.0, 0.00021, 0.0002, 0.0004, 2.25, 15.0, 0.8, 0.8, 0.4, 0.8, 20.0, 180.0, 20.0, 80.0)
        case 'refined_day_240923':
            # From log.auto_tune.12775227.o
            initial_params = (98.92576701793945, 55.547007702116076, 3.276435284891971e-07, 0.003219434103487201, 2.0733592813817836, 3.023777958013186, 3.7015828054941444, 0.11176915697262242, 0.10889014894153963, 0.5407751622475109, 0.8386663825220118, 0.00021524869316366733, 0.00013886676662742053, 0.000409124693665948, 1.7341853303061636, 18.193445773762313, 0.983224885119813, 0.8211408034304782, 0.35805868232813, 0.7968531827027974, 19.904587595478432, 179.3143570732816, 19.570736068639505, 54.24322183459618)
        case 'refined_week_240925':
            # From log.auto_tune.12876817.o
            initial_params = (97.81799684933156, 56.36130691240604, 3.2531990342778263e-07, 0.0032446640043990267, 2.087171033174861, 3.040920401857408, 3.7144744421342057, 0.11215779678073426, 0.10976710641428558, 0.5460298924354665, 0.8579880151217771, 0.00021785205734049876, 0.00013809789894812948, 0.0004107210208478399, 1.7534331840678492, 18.281650388240536, 0.9734564263399033, 0.8059692105121299, 0.35873987846203004, 0.7904773736561536, 19.998905079684214, 179.23077032470542, 19.51701457687504, 53.79489690706375)
        case 'refined_month_12964956':
            # From log.auto_tune.12964956.o
            initial_params = (97.94216605269102, 55.43352875104718, 3.311727037045482e-07, 0.0032541118738612265, 2.0711672879173237, 3.0563482403421327, 3.741453986087752, 0.11100034309219893, 0.10416483046595407, 0.5466000607537688, 0.8476999826202445, 0.00021756722012054934, 0.00013882994250661158, 0.00041353153403765293, 1.7528649114801411, 18.38941441749742, 0.9750155293112136, 0.8216201776009305, 0.3585483102924344, 0.7997400901030021, 19.819641897841365, 181.24582139262253, 19.78154003903417, 53.94375060563119)
        case 'refined_month_12964956_v2':
            # From log.auto_tune.12964956.o
            initial_params = (97.02117431628534, 54.96691243808803, 3.32815305873008e-07, 0.0032674576902865997, 2.057585503648391, 3.0746322736225498, 3.748698507273354, 0.11101247269463722, 0.1090882163078583, 0.5461180010887748, 0.8436024635936978, 0.00021838583186893596, 0.00013933045409905427, 0.00041993294520417223, 1.779999018783134, 18.26096524539539, 0.962281743066738, 0.8199807634748119, 0.3571808302592213, 0.8008634184022407, 19.891597004293594, 174.33862072583835, 19.738651467432724, 53.82001039510589)
    # Could also try BFGS or L-BFGS here
    res = minimize(call_icon, (initial_params), args=(objective_bounds), method='Nelder-Mead', options={'disp': True}, callback=early_stopping)
    return res

# Define objective bounds for calculation of loss function
objective_bounds = define_bounds()

# Optimize function
res = optimize(objective_bounds)

print("============================", flush=True)
print(" Tuned parameters: ",res, flush=True)
print("============================", flush=True)

