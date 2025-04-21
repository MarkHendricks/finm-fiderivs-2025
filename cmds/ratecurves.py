import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root




def spotcurve_to_discountcurve(spotcurve,n_compound=None):
    if isinstance(spotcurve,pd.DataFrame):
        spotcurve = spotcurve.iloc[:,0]
    
    disc_cum = 1
    discountcurve = pd.Series(dtype=float, index = spotcurve.index)

    for t in spotcurve.index:    
        spot = spotcurve.loc[t]
        if n_compound is None:
            discountcurve.loc[t] = np.exp(-t * spot)
        else:
            discountcurve.loc[t] = 1 / (1+(spot / n_compound))**(n_compound * t)

    return discountcurve



def discountcurve_to_spotcurve(discountcurve, n_compound=None):
    if isinstance(discountcurve,pd.DataFrame):
        discountcurve = discountcurve.iloc[:,0]
    
    spotcurve = pd.Series(dtype=float, index = discountcurve.index)

    for t, t_next in zip(discountcurve.index, discountcurve.index[1:]):
        Z = discountcurve.loc[t]
        if t>0:
            if n_compound is None:
                spotcurve.loc[t_next] = -np.log(Z) / t
            else:            
                spotcurve.loc[t_next] = n_compound * (Z**(-1/(n_compound * t)) - 1)
        
    return spotcurve








def spotcurve_to_forwardcurve(spotcurve, n_compound=None, dt=None):
    if isinstance(spotcurve,pd.DataFrame):
        spotcurve = spotcurve.iloc[:,0]
        
    if dt is None:
        dt = spotcurve.index[1] - spotcurve.index[0]
        
    discountcurve = ratecurve_to_discountcurve(spotcurve, n_compound=n_compound)
    
    F = discountcurve / discountcurve.shift()
    
    if n_compound is None:
        forwardcurve = -np.log(F) / dt
    else:
        forwardcurve = n_compound * (1/(F**(n_compound * dt)) - 1)
    
    return forwardcurve





def spotcurve_to_forwardcurve_old(spotcurve, n_compound=None, dt=None):
    if isinstance(spotcurve,pd.DataFrame):
        spotcurve = spotcurve.iloc[:,0]

    discountcurve = spotcurve_to_discountcurve(spotcurve, n_compound=n_compound)
    
    F = discountcurve.shift() / discountcurve
    
    forwardcurve = pd.Series(dtype=float, index = spotcurve.index)

    for t, t_next in zip(spotcurve.index[:-1], spotcurve.index[1:]):
        dt = t_next - t
        Fval = F.loc[t_next]
        if n_compound is None:
            forwardcurve.loc[t] = np.log(Fval) / dt
        else:
            forwardcurve.loc[t] = n_compound * (Fval**(1/(n_compound * dt)) - 1)
        
    return forwardcurve





# def spotcurve_to_forwardcurve_older(spotcurve, n_compound=None, dt=None):
#     if isinstance(spotcurve,pd.DataFrame):
#         spotcurve = spotcurve.iloc[:,0]
        
#     discountcurve = spotcurve_to_discountcurve(spotcurve, n_compound=n_compound)
    
#     F = discountcurve / discountcurve.shift()
    
#     forwardcurve = pd.Series(dtype=float, index = spotcurve.index)

#     for t, t_next in zip(spotcurve.index, spotcurve.index[1:]):
#         dt = t_next - t
#         Fval = F.loc[t_next]
#         if n_compound is None:
#             forwardcurve.loc[t_next] = -np.log(Fval) / dt
#         else:
#             forwardcurve.loc[t_next] = n_compound * (Fval**(-n_compound * dt) - 1) / dt
        
#     return forwardcurve





def ratecurve_to_discountcurve(ratecurve, n_compound=None):

    if isinstance(ratecurve,pd.DataFrame):
        ratecurve = ratecurve.iloc[:,0]
        
    if n_compound is None:
        discountcurve = np.exp(-ratecurve * ratecurve.index)
    else:
        discountcurve = 1 / (1+(ratecurve / n_compound))**(n_compound * ratecurve.index)

    return discountcurve  





def ratecurve_to_forwardcurve(ratecurve, n_compound=None, dt=None):
    if isinstance(ratecurve,pd.DataFrame):
        ratecurve = ratecurve.iloc[:,0]
        
    if dt is None:
        dt = ratecurve.index[1] - ratecurve.index[0]
        
    discountcurve = ratecurve_to_discountcurve(ratecurve, n_compound=n_compound)
    
    F = discountcurve / discountcurve.shift()
    
    if n_compound is None:
        display('TODO')
    else:
        forwardcurve = n_compound * (1/(F**(n_compound * dt)) - 1)
    
    return forwardcurve




def discount_to_intrate(discount, maturity, n_compound=None):
        
    if n_compound is None:
        intrate = - np.log(discount) / maturity
    
    else:
        intrate = n_compound * (1/discount**(1/(n_compound * maturity)) - 1)    
        
    return intrate



def interp_curves(data,dt=None, date=None, interp_method='linear',order=None, extrapolate=True):

    if dt is None:
        dt = data.columns[1] - data.columns[0]
    
    freq = 1/dt
    
    if date is None:
        temp = data
    else:
        temp = data.loc[date,:]

    newgrid = pd.DataFrame(dtype=float, index=np.arange(dt,temp.index[-1]+dt,dt),columns=['quotes'])
    # sofr curve last index often 10.02 command above extends to 10+. If barely overruns, toss last value
    overrun = (temp.index[-1] % dt)/dt
    if overrun>0 and overrun < .1:
        newgrid = newgrid.iloc[:-1,:]
        
    #newgrid.index = (freq*newgrid.index.values).round(0)/freq

    curves = temp.to_frame().rename(columns={temp.name:'quotes'})
    curves = pd.concat([curves,newgrid],axis=0)
    curves['interp'] = curves['quotes']

    if extrapolate:
        curves['interp'].interpolate(method=interp_method, order=order, limit_direction='both', fill_value = 'extrapolate',inplace=True)
    else:
        curves['interp'].interpolate(method=interp_method, order=order,inplace=True)
    
    curves = curves.loc[newgrid.index,:]
    curves = curves[~curves.index.duplicated()].sort_index()
    
    return curves




def plot_interp_curves(curves,plot_contin=True):
    fig, ax = plt.subplots()
    curves['quotes'].plot.line(ax=ax, linestyle='None',marker='*')
    curves.iloc[:,1:].plot.line(ax=ax, linestyle='--',marker='')
            
    plt.legend()
    plt.show()

    
    
    
def price_bond(ytm, T, cpn, cpnfreq=2, face=100, accr_frac=None):
    ytm_n = ytm/cpnfreq
    cpn_n = cpn/cpnfreq
    
    if accr_frac is None:
        #accr_frac = 1 - (T-round(T))*cpnfreq        
        accr_frac = 0

    if cpn==0:
        accr_frac = 0
        
    N = T * cpnfreq
    price = face * ((cpn_n / ytm_n) * (1-(1+ytm_n)**(-N)) + (1+ytm_n)**(-N)) * (1+ytm_n)**(accr_frac)
    return price




def duration_formula(tau, ytm, cpnrate=None, freq=2):

    if cpnrate is None:
        cpnrate = ytm
        
    y = ytm/freq
    c = cpnrate/freq
    T = tau * freq
        
    if cpnrate==ytm:
        duration = (1+y)/y  * (1 - 1/(1+y)**T)
        
    else:
        duration = (1+y)/y - (1+y+T*(c-y)) / (c*((1+y)**T-1)+y)

    duration /= freq
    
    return duration






def ytm(price, T, cpn, cpnfreq=2, face=100, accr_frac=None,solver='fsolve',x0=.01):
    
    pv_wrapper = lambda y: price - price_bond(y, T, cpn, cpnfreq=cpnfreq, face=face, accr_frac=accr_frac)

    if solver == 'fsolve':
        ytm = fsolve(pv_wrapper,x0)
    elif solver == 'root':
        ytm = root(pv_wrapper,x0)
    return ytm





def calc_swaprate(discounts,T,freqswap):
    freqdisc = round(1/discounts.index.to_series().diff().mean())
    step = round(freqdisc / freqswap)
    
    periods_swap = discounts.index.get_loc(T)
    # get exclusive of left and inclusive of right by shifting both by 1
    periods_swap += 1

    swaprate = freqswap * (1 - discounts.loc[T])/discounts.iloc[step-1:periods_swap:step].sum()
    return swaprate


def discounts_to_swaprates(discounts,freqswap):
    index_array = discounts.index.to_numpy()
    dgrid = 1/freqswap
    grid_points = np.arange(dgrid, index_array[-1]+dgrid, dgrid)
    nearest_points = [index_array[np.abs(index_array - g).argmin()] for g in grid_points]

    Zgrid = discounts.loc[nearest_points]
    swaprates = freqswap * (1-Zgrid) / Zgrid.expanding().sum()

    return swaprates
    
    


def calc_fwdswaprate(discounts, Tfwd, Tswap, freqswap):
    freqdisc = round(1/discounts.index.to_series().diff().mean())
    step = round(freqdisc / freqswap)
    
    periods_fwd = discounts.index.get_loc(Tfwd)
    periods_swap = discounts.index.get_loc(Tswap)
    # get exclusive of left and inclusive of right by shifting both by 1
    periods_fwd += step
    periods_swap += 1
    
    fwdswaprate = freqswap * (discounts.loc[Tfwd] - discounts.loc[Tswap]) / discounts.iloc[periods_fwd:periods_swap:step].sum()
    return fwdswaprate





def extract_fedpath(curves,feddates,spotfedrate):

    r0 = spotfedrate
    
    tag = [dt.strftime('%Y-%m') for dt in curves['last_tradeable_dt']]
    curves['date'] = tag
    curves.reset_index(inplace=True)
    curves.set_index('date',inplace=True)

    tag = [dt.strftime('%Y-%m') for dt in feddates['meeting dates']]
    feddates['date'] = tag
    feddates.set_index('date',inplace=True)

    curves = curves.join(feddates)
    curves['meeting day'] = [dt.day for dt in curves['meeting dates']]
    curves['contract days'] = [dt.day for dt in curves['last_tradeable_dt']]

    curves['futures rate'] = (100-curves['px_last'])/100
    curves.drop(columns=['px_last'],inplace=True)
    curves['expected fed rate'] = np.nan

    for step, month in enumerate(curves.index[:-1]):
        if step==0:
            Eprev = r0
        else:
            Eprev = curves['expected fed rate'].iloc[step-1]

        if np.isnan(curves['meeting day'].iloc[step]):
            curves['expected fed rate'].iloc[step] = Eprev
        else:
            if np.isnan(curves['meeting day'].iloc[step+1]):
                curves['expected fed rate'].iloc[step] = curves['futures rate'].iloc[step+1]
            else:
                n = curves['contract days'].iloc[step]
                m = curves['meeting day'].iloc[step]
                curves['expected fed rate'].iloc[step] = (n * curves['futures rate'].iloc[step] - m * Eprev)/(n-m)
                
    return curves









def calc_spot_rates(discounts, compounding=None):
    """
    Calculate spot rates from discount factors.
    
    Parameters:
        discounts (pd.Series): Discount factors indexed by maturity (e.g., 1, 2, 3, ...).
        compounding (float or None): 
            - If a number (e.g. 1, 2, 4, etc.), it represents the discrete compounding frequency for the output spot rates.
            - If None, continuous compounding is assumed.
    
    Returns:
        pd.Series: Spot rates as decimals.
    """
    spot_rates = pd.Series(index=discounts.index, dtype=float)
    
    for t, discount in discounts.items():
        if compounding is None:
            spot_rates[t] = -np.log(discount) / t
        else:
            m = compounding
            spot_rates[t] = m * ((1 / discount) ** (1 / (m * t)) - 1)
    
    return spot_rates














def recalc_rates_from_spot(spot_rates, compounding=None):
    """
    Recalculate discount factors, forward rates, and swap rates from a spot rate Series.
    
    The input spot rates are assumed to be annualized and are provided for maturities 
    (in years) that may be spaced arbitrarily (e.g., 1, 0.5, 0.25, etc.).
    
    Parameters:
        spot_rates (pd.Series): Annualized spot rates (as decimals) indexed by maturity.
        compounding (float or None): 
            - If a number (e.g., 1, 2, 4, etc.), it represents the discrete compounding frequency per year.
              In this case, discount factors are computed as 
                  D(t) = 1 / (1 + R(t)/m)^(m*t)
              and the forward rate from t1 to t2 is
                  F = m * { [D(t1)/D(t2)]^(1/(m*(t2-t1))) - 1 }.
            - If None, continuous compounding is assumed with
                  D(t) = exp(-R(t)*t)
              and the forward rate is
                  F = ln(D(t1)/D(t2)) / (t2-t1).
    
    Swap rates are computed assuming that payments occur at the times given by the index.
    For a swap maturing at time T, the swap rate is:
          S(T) = (1 - D(T)) / (sum_i alpha_i * D(t_i))
    where the accrual factors alpha_i are computed as:
          alpha_1 = t1 (assuming time 0 to t1)
          alpha_i = t_i - t_(i-1)  for i > 1.
    
    Returns:
        pd.DataFrame: A DataFrame with the input "spot rates" and the computed "discounts",
                      "forwards", and "swap rates" as columns.
    """
    # Ensure spot_rates is sorted by maturity.
    spot_rates = spot_rates.copy()
    spot_rates = spot_rates.sort_index()
    
    # Initialize Series for calculations.
    discounts = pd.Series(index=spot_rates.index, dtype=float)
    forwards  = pd.Series(index=spot_rates.index, dtype=float)
    swap_rates = pd.Series(index=spot_rates.index, dtype=float)
    
    # Calculate discount factors.
    for t in spot_rates.index:
        R_t = spot_rates[t]
        if compounding is None:
            # Continuous compounding
            discounts[t] = np.exp(-R_t * t)
        else:
            m = compounding  # discrete compounding frequency per year
            discounts[t] = 1 / ((1 + R_t / m) ** (m * t))
    
    # Calculate forward rates for the interval from the previous payment time to the current one.
    index_vals = list(spot_rates.index)
    for i, t in enumerate(index_vals):
        if i == 0:
            forwards[t] = spot_rates[t]  # For the first period, use the spot rate.
        else:
            t_prev = index_vals[i - 1]
            delta = t - t_prev  # interval between payment times
            if compounding is None:
                forwards[t] = np.log(discounts[t_prev] / discounts[t]) / delta
            else:
                m = compounding
                forwards[t] = m * (((discounts[t_prev] / discounts[t]) ** (1 / (m * delta))) - 1)
    
    # Calculate swap rates using accrual factors that reflect the actual spacing.
    for t in index_vals:
        # Get all payment times up to and including t.
        payment_times = [pt for pt in index_vals if pt <= t]
        accruals = []
        for j, pt in enumerate(payment_times):
            if j == 0:
                accrual = pt  # from time 0 to first payment time
            else:
                accrual = pt - payment_times[j - 1]
            accruals.append(accrual)
        # Sum of (accrual * discount factor) for each payment.
        weighted_sum = sum(accrual * discounts[pt] for pt, accrual in zip(payment_times, accruals))
        swap_rates[t] = (1 - discounts[t]) / weighted_sum
    
    # Build the resulting DataFrame.
    results = pd.DataFrame({
        "spot rates": spot_rates,
        "discounts": discounts,
        "forwards": forwards,
        "swap rates": swap_rates
    })
    
    return results