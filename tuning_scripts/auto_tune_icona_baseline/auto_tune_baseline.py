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

# Bound crs to a maximum of 1 

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

# Specify bounds
bmin=-np.inf; bmax=np.inf
bounds = ((bmin,0.99),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax),(bmin,bmax))

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
    # Some OBS bounds extracted from https://swift.dkrz.de/v1/dkrz_adf90815-96ce-4411-81ae-4db5e76adcbc/esmvaltool_output/index.html.
    obs_bounds = {}
    obs_bounds['rsut_min'] = 100
    obs_bounds['rsut_max'] = 110
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

# To extract all the output from a log file
def create_list(file_name, log_path):
    '''
        Returns a list of parameters, losses, losses after the TOA balance and the TOA metric value from a given log_file (file_name) that is located in a given path (log_path)
    '''
    no_params = 19 # For the ICON-A baseline
    param_values = np.zeros((1000, no_params))
    loss_values = []; net_toa_metric = []; loss_after_toa_values = []
    file_path = os.path.join(log_path, file_name)
    print_bool = False
    eval_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if print_bool:
                for param_ind in range(no_params):
                    # Append certain parameter value
                    param_values[eval_count, param_ind] = float(line.split(',')[param_ind])
                print_bool = False
                eval_count += 1
            # Append line after "Current list of parameters to tune"
            if line.startswith("Current list of parameters to tune"):
                print_bool = True
            if line.startswith("Loss after adding clt_SE_pac:"):
                loss_values.append(float(line.split(':')[1]))
            if line.startswith("Loss after adding net_toa:"):
                loss_after_toa_values.append(float(line.split(':')[1]))
            # TOA value
            if line.startswith("Metric net_toa:"):
                net_toa_metric.append(float(file.readline()[2:-2]))
    return param_values[:len(loss_values), :], np.array(loss_values), np.array(loss_after_toa_values), np.array(net_toa_metric)

def optimize(objective_bounds):
    # Default Sundqvist parameters
    abcdefghij_publication_240101 = (0.968, 0.8, 0.7, 2, 0.25) #crs, crt, csatsc, nex, cinv
    # Additional parameters that need to be considered if the desired goal cannot be reached with the cloud cover equation parameters only. pr0 should be between 0.6 and 1.0.
    additional_params = (1, 2.1e-4, 2.0e-4, 4.0e-4, 2.25, 15, 0.8, 0.8, 0.4, 0.8, 20, 180, 20, 80) # (pr0, entrmid, entrpen, entrdd, cvtfall, ccraut, cinhomi, cinhoml1, cinhoml2, cinhoml3, cn1lnd, cn2lnd, cn1sea, cn2sea)
    match INIT_PARAMS:
        case 'orig':
            initial_params = abcdefghij_publication_240101 + additional_params
        case 'day_refined': # From log.auto_tune_baseline.13758205.o. Loss of 0.071.
            initial_params = (0.9109554235257106, 0.5762940529431084, 0.6989086994289555, 2.1703722259096887, 0.24787901658913522, 1.0660473428441537, 0.00021731161660424106, 0.00020656433187313266, 0.00039733106447235075, 2.174323606149831, 16.61692827620488, 0.8497956283517695, 0.8663534546751761, 0.38911562467775485, 0.8310420045733715, 19.899284380894603, 177.2935044348356, 20.093970814347646, 82.62605287779985)
        case 'week_refined': # From log.auto_tune_baseline.13774617.o. Loss of 0.006
            initial_params = (0.9142084454021847, 0.5959131468894869, 0.7016983204535419, 2.169446548166456, 0.24862183830475587, 1.0588948615278229, 0.00021809514111887957, 0.000205913392405527, 0.00039846920997366024, 2.186463783358046, 16.697049053748103, 0.8457431126829968, 0.8648751522165565, 0.3898800218636592, 0.8324526207821763, 19.98802406295018, 177.60991737322595, 20.20550621794748, 81.94998525907712)
        case 'month_refined': # From log.auto_tune_baseline.13850767.o. Loss of 0.
            initial_params = (0.9142084454021847, 0.5959131468894869, 0.7016983204535419, 2.169446548166456, 0.24862183830475587, 1.0588948615278229, 0.00021809514111887957, 0.000205913392405527, 0.00039846920997366024, 2.186463783358046, 16.697049053748103, 0.8457431126829968, 0.8648751522165565, 0.3898800218636592, 0.8324526207821763, 20.98742526609769, 177.60991737322595, 20.20550621794748, 81.94998525907712)
        case 'year_refined': # From log.auto_tune_baseline.13852199.o.
            log_path_icona = "/work/bd1083/b309170/published_code/grundner25pnas_iconaml_automatic_tuning/tuning_scripts/auto_tune_icona_baseline"
            # First year-long tuning. Read parameter values and corresponding losses
            param_values, loss_values, loss_after_toa_values, _ = create_list("log.auto_tune.13852199.o", log_path_icona)
            # Index where loss is minimal.
            min_ind = np.argmin(loss_values)
            # Adjust initial guess by adding a weighted difference, motivated by reducing loss_after_toa_values
            C = loss_after_toa_values[0]/(loss_after_toa_values[0] - loss_after_toa_values[min_ind])
            initial_params = param_values[0] + C*(param_values[min_ind] - param_values[0])
        case 'year_refined_toa': # From log.auto_tune_baseline.13852199.o. Manual nudging for good TOA balance. Combination of crs and crt.
            initial_params = (0.968, 0.665, 0.7016983204535419, 2.169446548166456, 0.24862183830475587, 1.0588948615278229, 0.00021809514111887957, 0.000205913392405527, 0.00039846920997366024, 2.186463783358046, 16.697049053748103, 0.8457431126829968, 0.8648751522165565, 0.3898800218636592, 0.8324526207821763, 20.98742526609769, 177.60991737322595, 20.20550621794748, 81.94998525907712)
    # Could also try BFGS or L-BFGS here
    res = minimize(call_icon, (initial_params), args=(objective_bounds), bounds=bounds, method='Nelder-Mead', options={'disp': True}, callback=early_stopping)
    return res

# Define objective bounds for calculation of loss function
objective_bounds = define_bounds()

# Optimize function
res = optimize(objective_bounds)

print("============================", flush=True)
print(" Tuned parameters: ",res, flush=True)
print("============================", flush=True)

