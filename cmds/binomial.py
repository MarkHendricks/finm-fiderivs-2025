import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from treasury_cmds import compound_rate




def create_probability_matrix(n):
    # Create the initial nxn DataFrame with NaNs in the lower diagonal
    data = pd.DataFrame(np.nan, index=range(n), columns=range(n))

    # Set the initial probability for the starting point (0,0)
    data.iloc[0, 0] = 1.0

    # Iterate over each element in the upper triangular part of the DataFrame
    for i in range(n):
        for j in range(i, n):
            if j + 1 < n:  # Move directly to the right
                if np.isnan(data.iloc[i, j+1]):
                    data.iloc[i, j+1] = 0.0
                data.iloc[i, j+1] += data.iloc[i, j] * 0.5
            if i + 1 < n and j + 1 < n:  # Move diagonally down-and-to-the-right
                if np.isnan(data.iloc[i+1, j+1]):
                    data.iloc[i+1, j+1] = 0.0
                data.iloc[i+1, j+1] += data.iloc[i, j] * 0.5

    # Replace NaNs with 0 for clarity and ensure lower diagonal is 0
    for i in range(n):
        for j in range(i):
            data.iloc[i, j] = 0

    return data

def tree_label(type_tree,type_security, note=None):
    if note is None:
        label = f'Tree: <b>{type_tree} for {type_security}<b>'
    else:
        label = f'Tree: <b>{type_tree} for {type_security} {note}<b>'
    return label


def apply_shading(val, shade_values):
    color_matrix = pd.DataFrame("", index=val.index, columns=val.columns)
    
    for i in range(len(val)):
        for j in range(len(val.columns)):
            shading_value = shade_values.iloc[i, j]
            color = f'background-color: rgba(0, 0, 255, {shading_value})'
            color_matrix.iloc[i, j] = color
            
    return color_matrix



def format_bintree(df,style='{:.2f}',probs=None,label=None):
    if probs is None:
        out = df.style.format(style,na_rep='').format_index('{:.2f}',axis=1)
    else:
        normalized_probs = probs.div(probs.max())
        out = df.style.apply(apply_shading, shade_values=normalized_probs, axis=None).format(style,na_rep='').format_index('{:.2f}',axis=1)

    if label is None:
        return out
    else:
        return out.set_caption(label)



def construct_rate_tree(dt,T):
    timegrid = pd.Series((np.arange(0,round(T/dt)+1)*dt).round(6),name='time',index=pd.Index(range(round(T/dt)+1),name='state'))
    tree = pd.DataFrame(dtype=float,columns=timegrid,index=timegrid.index)
    return tree

def construct_quotes(maturities,prices,face=100):
    quotes = pd.DataFrame({'maturity':maturities,'price':prices})    
    quotes['continuous ytm'] = -np.log(quotes['price']/face) / quotes['maturity']
    quotes.set_index('maturity',inplace=True)
    
    return quotes





def payoff_bond(r,dt,facevalue=100):
    price = np.exp(-r * dt) * facevalue
    return price

def payoff_swap(r,swaprate,freqswap,ispayer=True,N=100):
    if ispayer:
        payoff = N * (r-swaprate) / freqswap 
    else:
        payoff = N * (swaprate-r) / freqswap 
        
    return payoff


def payoff_cap(r,strike,freq,ispayer=True,face=100):
    if ispayer:
        payoff = np.max(face * (r-strike) / freq ,0)
    else:
        payoff = np.max(face * (strike-r) / freq, 0)
        
    return payoff




def replicating_port(quotes,undertree,derivtree,dt=None,Ncash=100):
    if dt is None:
        dt = undertree.columns[1] - undertree.columns[0]
    
    delta = (derivtree.loc[0,dt] - derivtree.loc[1,dt]) / (undertree.loc[0,dt] - undertree.loc[1,dt]) 
    cash = (derivtree.loc[0,dt] - delta * undertree.loc[0,dt]) / Ncash
    
    out = pd.DataFrame({'positions':[cash,delta], 'value':quotes},index=['cash','under'])
    out.loc['derivative','value'] = out['positions'] @ out['value']
    return out





def bintree_pricing_old(payoff=None, ratetree=None, undertree=None,cftree=None, pstars=None,timing=None,style='european'):
        
    if payoff is None:
        payoff = lambda r: payoff_bond(r,dt)
    
    if undertree is None:
        undertree = ratetree
        
    if cftree is None:
        cftree = pd.DataFrame(0, index=undertree.index, columns=undertree.columns)
        
    if pstars is None:
        pstars = pd.Series(.5, index=undertree.columns)

    if undertree.columns.to_series().diff().std()>1e-5:
        display('time grid is unevenly spaced')
    dt = undertree.columns[1]-undertree.columns[0]

    
    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back==0:                           
            valuetree[t] = payoff(undertree[t])
            if timing== 'deferred':
                valuetree[t] *= np.exp(-ratetree[t]*dt)
        else:
            for state in valuetree[t].index[:-1]:
                valuetree.loc[state,t] = np.exp(-ratetree.loc[state,t]*dt) * (pstars[t] * valuetree.iloc[state,-steps_back] + (1-pstars[t]) * valuetree.iloc[state+1,-steps_back] + cftree.loc[state,t])

            if style=='american':
                valuetree.loc[:,t] = np.maximum(valuetree.loc[:,t],payoff(undertree.loc[:,t]) + np.exp(-ratetree.loc[:,t]*dt) * cftree.loc[:,t])

    return valuetree







def bintree_pricing(payoff=None, ratetree=None, undertree=None,cftree=None, dt=None, pstars=None, timing=None, cfdelay=False,style='european',Tamerican=0,compounding=None):
    
    if payoff is None:
        payoff = lambda r: 0
    
    if undertree is None:
        undertree = ratetree
        
    if cftree is None:
        cftree = pd.DataFrame(0, index=undertree.index, columns=undertree.columns)
        
    if pstars is None:
        pstars = pd.Series(.5, index=undertree.columns)

    if dt is None:
        dt = undertree.columns.to_series().diff().mean()
        dt = undertree.columns[1]-undertree.columns[0]
    
    if timing == 'deferred':
        cfdelay = True
    
    if dt<.25 and cfdelay:
        display('Warning: cfdelay setting only delays by dt.')
        
    if compounding is not None:
        ratetree_cont = compounding * np.log(1+ratetree/compounding)
    else:
        ratetree_cont = ratetree
    
    valuetree = pd.DataFrame(dtype=float, index=undertree.index, columns=undertree.columns)

    for steps_back, t in enumerate(valuetree.columns[-1::-1]):
        if steps_back==0:                           
            valuetree[t] = payoff(undertree[t])
            if cfdelay:
                valuetree[t] *= np.exp(-ratetree_cont[t]*dt)
        else:
            for state in valuetree[t].index[:-1]:
                val_avg = pstars[t] * valuetree.iloc[state,-steps_back] + (1-pstars[t]) * valuetree.iloc[state+1,-steps_back]
                
                if cfdelay:
                    cf = cftree.loc[state,t]
                else:                    
                    cf = cftree.iloc[state,-steps_back]
                
                valuetree.loc[state,t] = np.exp(-ratetree_cont.loc[state,t]*dt) * (val_avg + cf)

            if style=='american':
                if t>= Tamerican:
                    valuetree.loc[:,t] = np.maximum(valuetree.loc[:,t],payoff(undertree.loc[:,t]))
        
    return valuetree










def bond_price_error(quote, pstars, ratetree, style='european'):
    FACEVALUE = 100
    dt = ratetree.columns[1] - ratetree.columns[0]    
    payoff = lambda r: payoff_bond(r,dt)
    modelprice = bintree_pricing(payoff, ratetree, pstars=pstars, style=style).loc[0,0]
    error = modelprice - quote

    return error            








def estimate_pstar(quotes,ratetree,style='european'):

    pstars = pd.Series(dtype=float, index= ratetree.columns[:-1], name='pstar')
    p0 = .5
    
    for steps_forward, t in enumerate(ratetree.columns[1:]):        
        ratetreeT = ratetree.copy().loc[:,:t].dropna(axis=0,how='all')
        t_prev = ratetreeT.columns[steps_forward]
        
        pstars_solved = pstars.loc[:t_prev].iloc[:-1]
        wrapper_fun = lambda p: bond_price_error(quotes['price'].iloc[steps_forward+1], pd.concat([pstars_solved, pd.Series(p,index=[t_prev])]), ratetreeT, style=style)

        pstars[t_prev] = fsolve(wrapper_fun,p0)[0]

    return pstars



def exercise_decisions(payoff, undertree, derivtree):
    exer = (derivtree == payoff(undertree)) & (derivtree > 0)
    return exer






def rates_to_BDTstates(ratetree):
    ztree = np.log(100*ratetree)
    return ztree

def BDTstates_to_rates(ztree):
    ratetree = np.exp(ztree)/100
    return ratetree

def incrementBDTtree(ratetree, theta, sigma, dt=None):
    if dt is None:
        dt = ratetree.columns[1] - ratetree.columns[0]

    tstep = len(ratetree.columns)-1
    
    ztree = rates_to_BDTstates(ratetree)
    ztree.iloc[:,-1] = ztree.iloc[:,-2] + theta * dt + sigma * np.sqrt(dt)
    ztree.iloc[-1,-1] = ztree.iloc[-2,-2] + theta * dt - sigma * np.sqrt(dt)
    
    newtree = BDTstates_to_rates(ztree)
    return newtree

def incremental_BDT_pricing(tree, theta, sigma_new, dt=None):
    if dt==None:
        dt = tree.columns[1] - tree.columns[0]
    
    payoff = lambda r: payoff_bond(r,dt)
    newtree = incrementBDTtree(tree, theta, sigma_new)
    model_price = bintree_pricing(payoff, newtree)
    return model_price


def estimate_theta(sigmas,quotes_zeros,dt=None,T=None):
    if dt is None:
        dt = quotes_zeros.index[1] - quotes_zeros.index[0]

    if T is None:
        T = quotes_zeros.index[-2]

    if quotes_zeros.mean() < 1:
        scale = 1
    else:
        scale = 100
        
    ratetree = construct_rate_tree(dt,T)
    theta = pd.Series(dtype=float, index=ratetree.columns, name='theta')
    dt = ratetree.columns[1] - ratetree.columns[0]
    
    if type(sigmas) is float:
        sigmas = pd.Series(sigmas,index=theta.index)

    for tsteps, t in enumerate(quotes_zeros.index):
        if tsteps==0:
            ratetree.loc[0,0] = -np.log(quotes_zeros.iloc[tsteps]/scale)/dt
        else:
            subtree = ratetree.iloc[:tsteps+1,:tsteps+1]
            wrapper = lambda theta: incremental_BDT_pricing(subtree, theta, sigmas.iloc[tsteps]).loc[0,0] - quotes_zeros.iloc[tsteps] * 100 / scale
            
            theta.iloc[tsteps] = fsolve(wrapper,.5)[0]
            ratetree.iloc[:tsteps+1,tsteps] = incrementBDTtree(subtree, theta.iloc[tsteps], sigmas.iloc[tsteps]).iloc[:,tsteps]
            
            #print(f'Completed: {tsteps/len(quotes_zeros.index):.1%}')
            
    return theta, ratetree




def incrementBDTtree_topnode(ratetree, state, sigma, dt=None):
    if dt is None:
        dt = ratetree.columns[1] - ratetree.columns[0]

    tstep = len(ratetree.columns)-1
    
    ztree = rates_to_BDTstates(ratetree)
    ztree.iloc[0,-1] = state
    col = ztree.columns[-1]
    for idx in ztree.index[1:]:
        ztree.loc[idx,col] = ztree.loc[idx-1,col] - 2 * sigma * np.sqrt(dt)
    
    newtree = BDTstates_to_rates(ztree)
    return newtree



def incremental_BDT_pricing_topnode(tree, state, sigma_new, dt=None):
    if dt==None:
        dt = tree.columns[1] - tree.columns[0]
    
    payoff = lambda r: payoff_bond(r,dt)
    newtree = incrementBDTtree_topnode(tree, state, sigma_new)
    model_price = bintree_pricing(payoff, newtree)
    return model_price



def estimate_topnode(sigmas,quotes_zeros,dt=None,T=None,round_digits=None):
    if dt is None:
        dt = quotes_zeros.index[1] - quotes_zeros.index[0]
        #dt = quotes_zeros.index[0]

    if T is None:
        T = quotes_zeros.index[-2]

    if quotes_zeros.mean() < 1:
        scale = 1
    else:
        scale = 100
        
    ratetree = construct_rate_tree(dt,T)
    if round_digits is not None:
        ratetree.columns = np.round(ratetree.columns,round_digits)
    topnode = pd.Series(dtype=float, index=ratetree.columns, name='top node')
    dt = ratetree.columns[1] - ratetree.columns[0]
        
    if type(sigmas) is float:
        sigmas = pd.Series(sigmas,index=topnode.index)

    for state, t in enumerate(ratetree.columns):
        if state==0:
            ratetree.loc[state,t] = -np.log(quotes_zeros.loc[t+dt]/scale)/dt
        else:
            subtree = ratetree.loc[:state,:t]
            sigma_val = sigmas.loc[t]
            
            print(t)            
            tau = np.round(t+dt,round_digits)
            wrapper = lambda topnode_val: incremental_BDT_pricing_topnode(subtree, topnode_val, sigma_val).loc[0,0] - quotes_zeros.loc[tau] * 100 / scale
            
            topnode_val = fsolve(wrapper,.5)[0]
            topnode.loc[t] = topnode_val
            ratetree.loc[:state,t] = incrementBDTtree_topnode(subtree, topnode_val, sigma_val).loc[:,t]
            
            
    return topnode, ratetree





def estimate_topnode_vol(quotes_zeros,quotes_caps,quote_swaprates,dt=None,T=None):
    if dt is None:
        dt = quotes_zeros.index[1] - quotes_zeros.index[0]

    if T is None:
        T = quotes_zeros.index[-2]

    if quotes_zeros.mean() < 1:
        scale = 1
    else:
        scale = 100
        
    ratetree = construct_rate_tree(dt,T)
    params = pd.DataFrame(dtype=float, index=ratetree.columns, columns=['ivol','top node','flag'])
    dt = ratetree.columns[1] - ratetree.columns[0]
    
    for state, t in enumerate(ratetree.columns):
        if state==0:
            ratetree.loc[state,t] = -np.log(quotes_zeros.loc[t+dt]/scale)/dt
        else:
            subtree = ratetree.loc[:state,:t]
            
            wrapper = lambda params: [incremental_BDT_pricing_topnode(subtree, params[1], np.abs(params[0])).loc[0,0] - quotes_zeros.loc[t+dt] * 100 / scale,
                                      incremental_cap_pricing_topnode(subtree, params[1], np.abs(params[0]),strike=quote_swaprates.loc[t+dt],compound=4).loc[0,0] - quotes_caps.loc[t+dt]]
            
            sol = fsolve(wrapper,[.2,.7],full_output=True)
            params_val = sol[0]
            params.loc[t,['ivol','top node']] = params_val
            params.loc[t,'flag'] = sol[2]

            ratetree.loc[:state,t] = incrementBDTtree_topnode(subtree, params_val[1], params_val[0]).loc[:,t]
            # if state==3:
            #     mark
    return params, ratetree








def incremental_cap_pricing_topnode(tree, topnode, fwdvol, strike=None, compound=4, isPayer=False,dt=None, face=100):

    if dt==None:
        dt = tree.columns[1] - tree.columns[0]
    
    if isPayer:
        payoff = lambda r: face * dt * np.maximum(r-strike,0)
    else:
        payoff = lambda r: face * dt * np.maximum(strike-r,0)
    
    newtree = incrementBDTtree_topnode(tree, topnode, fwdvol)

    if compound is None:
        refratetree = compound * (np.exp(newtree / compound)-1)
    else:
        refratetree = newtree

    cftree = payoff(refratetree)
    model_price = bintree_pricing(payoff=payoff, ratetree=newtree, undertree= refratetree, cftree=cftree, timing='deferred')

    return model_price













def construct_bond_cftree(T, compound, cpn, cpn_freq=2, face=100,drop_final_period=True):
    step = int(compound/cpn_freq)

    cftree = construct_rate_tree(1/compound, T)
    cftree.iloc[:,:] = 0
    cftree.iloc[:, -1:0:-step] = (cpn/cpn_freq) * face
    
    if drop_final_period:
    # final cashflow is accounted for in payoff function
    # drop final period cashflow from cashflow tree
        cftree = cftree.iloc[:-1,:-1]
    else:
        cftree.iloc[:,-1] += face
    
    # clean up cf tree to be upper diagonal
    cftree.values[np.tril_indices_from(cftree, k=-1)] = np.nan
    return cftree



# def construct_accinttree_old(cftree, compound, cpn, cpn_freq=2, face=100, cleancall=True):
#     accinttree = cftree.copy()
#     step = int(compound/cpn_freq)
#     if cleancall is True:
#         accinttree.iloc[:,-1::-step] = face * (cpn/compound)
        
#     return accinttree


def construct_accint(timenodes, freq, cpn, cpn_freq=2, face=100):

    mod = freq/cpn_freq
    cpn_pmnt = face * cpn/cpn_freq

    temp = np.arange(len(timenodes)) % mod
    # shift to ensure end is considered coupon (not necessarily start)
    temp = (temp - temp[-1] - 1) % mod
    temp = cpn_pmnt * temp.astype(float)/mod

    accint = pd.Series(temp,index=timenodes)

    return accint



def idx_payoff_periods(series_periods, freq_payoffs, freq_periods=None):
    return ((series_periods * freq_periods) % (freq_periods / freq_payoffs)) ==0


def construct_swap_cftree(ratetree, swaprate, freqswap=1, T=None, freq=None, ispayer=True, N=100,cmpnd_ratetree=None):
    cftree = pd.DataFrame(0, index=ratetree.index, columns=ratetree.columns)
    cftree[ratetree.isna()] = np.nan

    if freq is None:
        freq = round(1/cftree.columns.to_series().diff().mean())
    
    if T is None:
        T = cftree.columns[-1] + 1/freq
        
    mask_swap_dates = idx_payoff_periods(cftree.columns, freqswap, freq)
    mask_cols = cftree.columns[mask_swap_dates]
    
    payoff = lambda r: payoff_swap(r,swaprate,freqswap,ispayer=ispayer,N=100)
    
    refratetree = compound_rate(ratetree,cmpnd_ratetree,freqswap)
    
    cftree[mask_cols] = payoff(refratetree[mask_cols])

    # final cashflow is accounted for in payoff function
    # will not impact bintree_pricing, but should drop them for clarity
    #cftree.iloc[:,-1] = 0
    
    return cftree, refratetree







def price_callable(quotes, fwdvols, cftree, accint, wrapper_bond, payoff_call,cleanstrike=True):

    theta, ratetree = estimate_theta(fwdvols,quotes)
    bondtree = bintree_pricing(payoff=wrapper_bond, ratetree=ratetree, cftree= cftree)
    if cleanstrike:
        cleantree = np.maximum(bondtree.subtract(accint,axis=1),0)
        undertree = cleantree
    else:
        undertree = bondtree
        
    calltree = bintree_pricing(payoff=payoff_call, ratetree=ratetree, undertree= undertree, style='american')
    callablebondtree = bondtree - calltree
    model_price_dirty = callablebondtree.loc[0,0]

    return model_price_dirty



def BDTtree(thetas, sigmas, r0=None, px_bond0=None, dt=None, T=None):

    if dt is None:
        dt = thetas.index[1] - thetas.index[0]

    if T is None:
        T = thetas.index[-1]

    if r0 is None:
        r0 = -np.log(px_bond0)/dt

    ztree = construct_rate_tree(dt,T)
    ztree.iloc[0,0] = rates_to_BDTstates(r0)

    # sigmas is indexed starting at dt, so tsteps is lagged
    for tsteps, t in enumerate(sigmas.index):
        ztree.iloc[:,tsteps+1] = ztree.iloc[:,tsteps] + thetas.iloc[tsteps] * dt + sigmas.iloc[tsteps] * np.sqrt(dt)
        ztree.iloc[tsteps+1,tsteps+1] = ztree.iloc[tsteps,tsteps] + thetas.iloc[tsteps] * dt - sigmas.iloc[tsteps] * np.sqrt(dt)
            
    bdttree = BDTstates_to_rates(ztree)

    return bdttree




def align_days_interval_to_tree_periods(days,freq):
    yrs = days / 365.25
    treeyrs = round(round(yrs * freq)/freq,6)

    return treeyrs


