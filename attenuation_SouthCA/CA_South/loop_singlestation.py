# %% [markdown]
# # Welcome to the tutorial measuring intrinsic absorption parameter!
# 
# This Notebook calculate the scattering & intrinsic absorption parameters of the Rayleigh waves following the instruction proposed by Hirose et al. (2019).
# 
# ## **Publication about this script**:
# Hirose, T., Nakahara, H., & Nishimura, T. (2019). A passive estimation method of scattering and intrinsic absorption parameters from envelopes of seismic ambient noise cross‐correlation functions. Geophysical Research Letters, 46(7), 3634-3642. https://doi.org/10.1029/2018GL080553
# 
# Hirose, T., Ueda, H., & Fujita, E. (2022). Scattering and intrinsic absorption parameters of Rayleigh waves at 18 active volcanoes in Japan inferred using seismic interferometry. Bulletin of Volcanology, 84(3), 34. https://doi.org/10.1007/s00445-022-01536-w
# 
# ### This notebook demonstrates single-station measurements
# 
# Step: <br>
# 0) Data preparing and filtering <br> 1) Calculation of mean-squared (MS) envelopes --> observed energy densities (Eobs) <br> 
# 2) Calculation of synthesized energy densities (Esyn) via a grid search <br>
# 3) Determination of best-fit parameters: intrinsic absorption parameter *b* and intrinsic *Q-value* (for single station)<br>

# %%
import os
import numpy as np

import matplotlib.pyplot as plt

from obspy.signal.filter import bandpass
from noisepy.monitoring.attenuation_utils import *
from noisepy.monitoring.monitoring_utils import *
from noisepy.seis.noise_module import mad
from noisepy.seis.io.asdfstore import ASDFStackStore

from datetimerange import DateTimeRange
from datetime import datetime, timezone

# %%
def plot_waveforms(ncmp, wav, fname, comp_arr):
    fig, ax = plt.subplots(1, ncmp, figsize=(16, 3), sharex=False)

    for n in range(ncmp):
        absy = max(wav[n][1], key=abs)
        ax[n].set_ylim(absy * -1, absy)
        ax[n].plot(wav[n][0], wav[n][1], linewidth=0.5,)
        ax[n].set_xlim(wav[n,0,0],wav[n,0,-1])
        ax[n].set_xlabel("time [s]")
        ax[n].set_title(f'{fname} {comp_arr[n]}')
    fig.tight_layout()
    # print("save figure as Waveform_readin_%s.png"%(fname))
    plt.savefig("Waveform_readin_%s.png" % (fname), format="png", dpi=100)
    plt.close(fig)

def plot_filtered_waveforms(freq, wav, fname, comp_arr):
    nfreq = len(freq) - 1
    ncmp = len(comp_arr)
    fig, ax = plt.subplots(ncmp,nfreq, figsize=(16,10), sharex=False)
    tt=wav[0,0]
    for cmp, ccomp in enumerate(comp_arr):
        for fb in range(nfreq):
            fmin=freq[fb]
            fmax=freq[fb+1]
            absy=max(wav[cmp,fb+1], key=abs)
            ax[cmp,fb].set_ylim(absy*-1,absy)
            ax[cmp,fb].plot(tt,wav[cmp,fb+1], "k-", linewidth=0.2)
            ax[cmp,fb].set_xlim(tt[0],tt[-1])
            ax[cmp,fb].set_xlabel("Time [s]")
            ax[cmp,fb].set_ylabel("Amplitude")
            ax[cmp,fb].set_title( "%s   %s   @%4.2f-%4.2f Hz" % ( fname,ccomp,fmin,fmax ) )
    plt.tight_layout()
    plt.savefig("Waveform_filtered_%s.png" % (fname), format="png", dpi=100)
    plt.close(fig)

def plot_envelope(comp_arr, freq, msv, msv_mean, fname, vdist):
    nfreq = len(freq) - 1
    ncmp = len(comp_arr)

    fig, ax = plt.subplots(ncmp+1, nfreq, figsize=(16,10), sharex=False)   
    for n in range(len(comp_arr)):

        for fb in range(nfreq):
            fmin=freq[fb]
            fmax=freq[fb+1]    
            ax[n,fb].plot(msv[n][0][:], msv[n][fb+1], "k-", linewidth=0.5)
            ax[n,fb].set_title("%s   %.2fkm  %s   @%4.2f-%4.2f Hz" % (fname,vdist,comp_arr[n],fmin,fmax))
            ax[n,fb].set_xlabel("Time [s]")
            ax[n,fb].set_ylabel("Amplitude")
            ax[n,fb].set_yscale('log', base=10)
            ax[n,fb].set_xlim(msv[n,0,0],msv[n,0,-1])
            ax[n,fb].set_ylim(10**(-6),5)

    for fb in range(nfreq):
        fmin=freq[fb]
        fmax=freq[fb+1]
        ax[-1,fb].plot(msv_mean[0], msv_mean[fb+1], "b-", linewidth=1)
        ax[-1,fb].set_title(" Mean Squared Value %.2fkm  @%4.2f-%4.2f Hz" % (vdist,fmin,fmax))
        ax[-1,fb].set_xlabel("Time [s]")
        ax[-1,fb].set_ylabel("Amplitude")
        ax[-1,fb].set_yscale('log', base=10)
        ax[-1,fb].set_xlim(msv_mean[0,0],msv_mean[0,-1])
        ax[-1,fb].set_ylim(10**(-6),5)
        
    plt.tight_layout()
    plt.savefig("Waveform_envelope_%s.png" % (fname), format="png", dpi=100)
    plt.close(fig)

def plot_fmsv_waveforms(freq,wav,fname,noise_level,twin):
    nfreq = len(freq) - 1
    fig, ax = plt.subplots(1,nfreq, figsize=(16,3), sharex=False)
    
    for fb in range(nfreq):
        fmin=freq[fb]
        fmax=freq[fb+1]
        absy=1 #max(wav[fb], key=abs)
        ax[fb].plot([wav[0][0],wav[0][-1]],[noise_level[fb],noise_level[fb]],c='blue',marker='.',ls='--', linewidth=2)
        ax[fb].plot([twin[fb][0],twin[fb][0]],[-0.1,absy],c='orange',marker='.',ls='--', linewidth=2)
        ax[fb].plot([twin[fb][1],twin[fb][1]],[-0.1,absy],c='orange',marker='.',ls='--', linewidth=2)
        ax[fb].set_yscale('log', base=10)
        ax[fb].plot(wav[0],wav[fb+1], "k-", linewidth=0.5)
        ax[fb].set_xlabel("Time [s]")
        ax[fb].set_ylabel("Amplitude in log-scale")
        ax[fb].set_title( "%s   @%4.2f-%4.2f Hz" % ( fname,fmin,fmax ) )
        ax[fb].set_xlim(wav[0,0],wav[0,-1])
    fig.tight_layout()
    plt.savefig("Waveform_fmsv_%s.png" % (fname), format="png", dpi=100)
    plt.close(fig)

def plot_singwindow_fitting_result(mean_free,intrinsic_b,tt,Eobs,Esyn,fname,dist,twin,fmin,fmax,win_num):
    plt.figure(figsize=(4,2))
    plt.yscale('log', base=10)
    
    intrinsic_Q=(2.0*np.pi*((fmin+fmax)/2))/intrinsic_b
    
    pymax=np.max(Eobs[win_num-1,:-2]*5)
    pymin=10**(-6)
    plt.ylim( pymin , pymax )
    plt.plot( tt, Eobs[win_num-1], "k-", linewidth=1)
    plt.plot( tt, Esyn[win_num-1], "b--", linewidth=1)
    plt.plot([twin[0],twin[0],twin[-1],twin[-1],twin[0]],[pymin, pymax,pymax,pymin,pymin],"r", linewidth=2)

    plt.title("%s  %.1fkm @%4.2f-%4.2f Hz, \nintrinsic b: %.2f, intrinsic Q: %.2f"
            % ( fname,dist,fmin,fmax,intrinsic_b, intrinsic_Q))
    plt.xlabel("Time [s]")
    plt.ylabel("Energy density Amp")
    plt.tight_layout()   
    plt.savefig("Waveform_singwin_%s_F%4.2f-%4.2f.png" % (fname,fmin,fmax), format="png", dpi=100)
    plt.close()

# def plot_fmsv_multiwindows(freq,wav,fname,noise_level,twin,pretwin):
#     nfreq = len(freq) - 1
#     fig, ax = plt.subplots(1,nfreq, figsize=(12,2), sharex=False)

#     for fb in range(nfreq):
#         fmin=freq[fb]
#         fmax=freq[fb+1]
#         absy=1 #max(wav[fb], key=abs)
#         ax[fb].plot([wav[0][0],wav[0][-1]],[noise_level[fb],noise_level[fb]],c='blue',marker='.',ls='--', linewidth=2)
#         for ncoda in range(len(twin[fb])):
#             ax[fb].plot([twin[fb][ncoda][0],twin[fb][ncoda][0]],[-0.1,absy],c='red',ls='-', linewidth=0.25)
#             ax[fb].plot([twin[fb][ncoda][1],twin[fb][ncoda][1]],[-0.1,absy],c='red',ls='-', linewidth=0.25)
#         ax[fb].plot([pretwin[fb][0],pretwin[fb][0]],[-0.1,absy],c='orange',marker='.',ls='--', linewidth=2)
#         ax[fb].plot([pretwin[fb][1],pretwin[fb][1]],[-0.1,absy],c='orange',marker='.',ls='--', linewidth=2)
#         ax[fb].set_yscale('log', base=10)
#         ax[fb].plot(wav[0],wav[fb+1], "k-", linewidth=0.5)
#         ax[fb].set_xlabel("Time [s]")
#         ax[fb].set_ylabel("Amplitude in log-scale")
#         ax[fb].set_title( "%s   @%4.2f-%4.2f Hz" % ( fname,fmin,fmax ) )
#     fig.tight_layout()
#     plt.show()    

# def plot_multiwindow_fitting_result(mean_free,intrinsic_b,tt,Eobs,Esyn,fname,dist,twin,fmin,fmax,win_num):
#     nwindows=twin.shape[0]
    
#     pymax=np.max(Eobs[:-2]*5)
#     pymin=10**(-6)
#     fig, ax= plt.subplots(nwindows, figsize=(6,8))
#     for k in range(nwindows):
#         ax[k].set_yscale('log', base=10)
#         ax[k].set_ylim( pymin , pymax )
#         ax[k].plot( tt, Eobs[k], "k-", linewidth=1)
#         ax[k].plot( tt, Esyn[k], "b--", linewidth=1)
    
#         ax[k].plot([twin[k,0],twin[k,0]], [0, pymax], 'r--', label=f'[{twin[k,0]:.2f}, {twin[k,1]:.2f}] sec')
#         ax[k].plot([twin[k,1], twin[k,1]], [0, pymax], 'r--')
#         ax[k].legend(loc='upper right')
#         # ax[k].set_xlim(0, tt[-20])
#     ax[0].set_title("Window no. %d  %s @%4.2f-%4.2f Hz, intrinsic b: %.2f, mean free path: %.2f" \
#             % (  win_num, fname,fmin,fmax,intrinsic_b, mean_free ) )
#     ax[-1].set_xlabel("Lag time (sec)")
#     ax[-1].set_ylabel("Energy density Amp")
#     plt.tight_layout()   
#     plt.show()

# %% [markdown]
# ### Step 0 ---  Data preparing and filtering 

# %%
data_path = "/home/kffeng/DATA_CaSC/scedc_data_2020-2022/STACK_acfs_CI/"

os.makedirs(data_path,exist_ok=True)
print(os.listdir(data_path))
wave_store = ASDFStackStore(data_path)

# %%
start = datetime(2020, 1, 1, tzinfo=timezone.utc)
end = datetime(2023, 1, 1, tzinfo=timezone.utc)
timerange = DateTimeRange(start, end)

# %%
# --- print results to file
outdir="./"
fcsv=outdir+"/OUTPUT.csv"
file = open(fcsv, "w")

line='netst,stlo,stla,timerange,fband,fmin,fmax,fwcen,intb,intQ,mean_free,scaling_amp,tbeg,tend,noiselevel,madratio,data_path\n'
file.write(line)
print("Output file: ",fcsv)

fn_sta=outdir+"/station_list.csv"
file_sta = open(fn_sta, "w")
line='netst,stlo,stla\n'
file_sta.write(line)

# %%
config_monito = ConfigParameters_monitoring() # default config parameters which can be customized
# --- parameters for measuring attenuation ---
config_monito.smooth_winlen = 5.0  # smoothing window length
config_monito.cvel = 2.6  # Rayleigh wave velocities over the freqency bands
config_monito.atten_tbeg = 2.0
config_monito.atten_tend = 10.0 # or it will be determined by a ratio of MAD value
config_monito.ratio = 10.0 # ratio for determining noise level by Mean absolute deviation (MAD)
config_monito.intb_interval_base=0.005 # interval base for a grid-searching process

# basic parameters
config_monito.freq = [0.5, 1.0, 2.0, 4.0]  # targeted frequency band for waveform monitoring
nfreq = len(config_monito.freq) - 1

# %%
pairs = wave_store.get_station_pairs()
print(f"Found {len(pairs)} station pairs")
print('pairs: ',pairs)

stations = set(pair[0] for pair in pairs)
print('Stations: ', stations)

sta_stacks = wave_store.read_bulk(timerange, pairs) # no timestamp used in ASDFStackStore

# %% [markdown]
# ### Step 1 --- Calculation of mean-squared (MS) envelopes --> observed energy densities (***Eobs***)
# --> normalized MS envelope is referred to as the observed energy density Eobs 

for nsta, target_pair in enumerate(pairs):
    
    print("Target pair: ", nsta, target_pair)
    print("Processed station pair: ",sta_stacks[nsta][0])
    
    stacks = sta_stacks[nsta][1]
    target_sta=sta_stacks[nsta][0][0].network+"."+sta_stacks[nsta][0][0].name

    params = stacks[0].parameters
    dt, lag , dist , stlo, stla = (params[p] for p in ["dt", "maxlag", "dist","lonR","latR"])
    line=f'{target_sta},{stlo:.6f},{stla:.6f}\n'
    file_sta.write(line)
    
    npts = int(lag*(1/dt)*2+1)
    tvec = np.arange(-npts // 2 + 1, npts // 2 + 1) * dt
    print("Lag-time: ",lag,", sampling rate: ",(1/dt) ,", total data length in points: ",npts)

    # detremine the component for processing
    comp_arr = ["EN","EZ","NZ"] 
    num_cmp=len(comp_arr)
    fnum=len([target_pair])
    
    # define the array being used
    stackf=np.ndarray((fnum, num_cmp, 2, npts))  
    vdist=np.zeros((fnum,1))  # S-R distance array

    # file name array
    fname=[]                  
    stack_name="Allstack_robust"
    # loop through each station-pair
    for aa, sfile in enumerate([target_sta]):
        fname.append(sfile)
        # read stacked waveforms accordingly to the cross-component
        for ncmp, ccomp in enumerate(comp_arr):
                         
            stacks_data = list(filter(lambda x: (x.component == ccomp) and (x.name == stack_name), stacks))
            #print(aa, ncmp, stacks_data)
            stackf[aa][ncmp]=[tvec, stacks_data[0].data]
            vdist[aa]=dist
            
        plot_waveforms(num_cmp,stackf[aa],sfile,comp_arr)
    fnum=len(fname)

    # %%
    MSE=np.ndarray((fnum,num_cmp,nfreq+1,npts)) # filtered two-side averaged stack CF

    for aa in range (fnum):
        dafbp=np.ndarray((nfreq,npts))
        
        for ncmp, ccomp in enumerate(comp_arr):
            for fb in range(nfreq):
                fmin=config_monito.freq[fb]
                fmax=config_monito.freq[fb+1]
                tt = np.arange(0, npts) * dt
                data = stackf[aa][ncmp][1]
                dafbp[fb] = bandpass(data, fmin, fmax, int(1 / dt), corners=4, zerophase=True)

            MSE[aa][ncmp]=[stackf[aa][ncmp][0],dafbp[0],dafbp[1],dafbp[2]] 

        plot_filtered_waveforms(config_monito.freq, MSE[aa], fname[aa], comp_arr)

    # %%
    # get the mean-squared value on each componet and also the average waveform
    msv=np.zeros((fnum,num_cmp,nfreq+1,npts))
    msv_mean=np.zeros((fnum,nfreq+1,npts))

    # smoothing window lengths corresponding to the frequency bands
    #winlen=np.full(nfreq, config_monito.smooth_winlen)
    for aa in range(fnum):

        for ncmp in  range(len(comp_arr)):
            ccomp=comp_arr[ncmp]
            msv[aa][ncmp][0]=MSE[aa][ncmp][0][:]
            for fb in range(nfreq):
                winlen=(1/config_monito.freq[fb])*2
                data=MSE[aa][ncmp][fb+1][:]
                fmin=config_monito.freq[fb]
                fmax=config_monito.freq[fb+1]
                
                para = { 'winlen':winlen, 'dt':dt , 'npts': len(data)}
                msv[aa][ncmp][fb+1]=get_smooth(data, para)
                
                msv[aa][ncmp][fb+1]=msv[aa][ncmp][fb+1]/np.max(msv[aa][ncmp][fb+1])  # self-normalized 
        
        # get average waveforms from components
        msv_mean[aa][0]=msv[aa][0][0][:]
        for fb in range(nfreq):
            fmin=config_monito.freq[fb]
            fmax=config_monito.freq[fb+1]
            for ncmp in range(len(comp_arr)):
                msv_mean[aa][fb+1]+=msv[aa][ncmp][fb+1][:]
            msv_mean[aa][fb+1]=msv_mean[aa][fb+1]/len(comp_arr)
            
        plot_envelope(comp_arr, config_monito.freq, msv[aa], msv_mean[aa], fname[aa], vdist[aa])

    # %%
    # get symmetric waveforms and determine the measuring window based on noise level (here is using 3 times of mad)
    half_npts = npts // 2                      # half-side number of points
    data_sym=np.ndarray((nfreq,half_npts+1)) # two-side averaged stack CF
    fmsv_mean=np.ndarray((fnum,nfreq+1,half_npts+1))

    # noise level setting
    ratio=config_monito.ratio
    level=np.ndarray((fnum,nfreq,1))
    twinbe=np.ndarray((fnum,nfreq,2))
    too_short=False
    
    for aa in range (fnum):
        if too_short:
            break 
        for fb in range(nfreq):
            fmin=config_monito.freq[fb]
            fmax=config_monito.freq[fb+1]            # stack positive and negative lags  
            sym=get_symmetric(msv_mean[aa][fb+1],half_npts)
            data_sym[fb]=sym
            Val_mad=mad(sym)
            level[aa][fb]=Val_mad*ratio
            if too_short:
                break
            for pt in range(len(sym)):
                if (sym[pt] < float(level[aa][fb])):
                    twinbe[aa][fb][0]=int((1/fmin)*3)
                    twinbe[aa][fb][1]=int(msv[aa][0][0][half_npts+pt])
                    #print(aa,fb,pt,sym[pt],level[aa][fb],twinbe[aa][fb])
                    if (twinbe[aa][fb][1] - twinbe[aa][fb][0]) < (1/fmin)*2:
                        print("Warning: the measuring window is too short")
                        too_short=True
                        break
                    else:
                        too_short=False
                        break
        if too_short:
            break            
        else:
            fmsv_mean[aa]=[msv[aa][0][0][half_npts:],data_sym[0],data_sym[1],data_sym[2]]
            plot_fmsv_waveforms(config_monito.freq,fmsv_mean[aa],fname[aa],level[aa],twinbe[aa])
        
    if too_short:
        print("Warning: skip station %s due to too short measuring window"%(fname[aa]))
        continue

    # %% [markdown]
    # ### Step 2 --- Calculation of synthesized energy densities (***Esyn***) via a grid search 
    # The 2-D radiative transfer equation for scalar waves  ***(Shang and Gao 1988; Sato 1993)*** is assuming isotropic scattering and source radiation in infinite medium to calculate ***synthesized energy densities  Esyn*** :

    # %%
    cvel=np.full(nfreq, config_monito.cvel)    # Rayleigh wave velocities over the freqency bands
    mfpx=np.zeros(1)        # mean_free_path search array
    intby=np.zeros(400)       # intrinsic_b search array
    nwindows=1               # number of sliced coda windows for fitting (1 or 6 fixed)

    # %%
    new_twin = np.zeros((fnum,nfreq,nwindows,2))
    if nwindows == 1:
        for aa in range(fnum):
            for fb in range(nfreq):
                new_twin[aa,fb,0]=[twinbe[aa,fb,0],twinbe[aa,fb,1]]
    else:
        for aa in range(fnum):
            for fb in range(nfreq):
                tb=twinbe[aa,fb,0]
                te=twinbe[aa,fb,1]
                new_twin[aa,fb] = window_determine(tb,te)
        plot_fmsv_multiwindows(config_monito.freq,fmsv_mean[aa],fname[aa],level[aa],new_twin[aa], twinbe[aa])

    # %%
    # getting the sum of squared residuals (SSR) between Eobs and Esyn  
    SSR_final=np.zeros((len(mfpx),len(intby)))
    SSR=np.zeros((nfreq,len(mfpx),len(intby)))

    for fb in range(nfreq):
        fmin=config_monito.freq[fb]
        fmax=config_monito.freq[fb+1]
        c=cvel[fb]
        
        fmsv_mean_single = np.zeros((1,half_npts+1))
        fmsv_mean_single[0] = fmsv_mean[0,fb+1,:] # get the mean-squared value on the targeted frequency band
        
        coda_single_band = new_twin[:,fb,:]
        
        # parameters for getting the sum of squared residuals (SSR) between Eobs and Esyn 
        para={ 'vdist':vdist, 'npts':npts, 'dt':dt, 'cvel':c, \
            'mfp':mfpx, 'intb':intby, 'twin':coda_single_band, 'fmsv':fmsv_mean_single  }
        # call function get_SSR
        SSR_final, mfpx, intby = get_SSR_codawindows(para)
        
        SSR[fb]=SSR_final


    # %% [markdown]
    # ### Step 3 --- Determination of best-fit parameters 
    # %%
    # getting the optimal value from the SSR
    result_intb=np.ndarray((nfreq))
    result_mfp=np.ndarray((nfreq))

    Eobs=np.ndarray((fnum, nfreq, nwindows, half_npts+1))
    Esyn=np.ndarray((fnum, nfreq, nwindows, half_npts+1))
    scaling_amp=np.ndarray((nfreq, nwindows))

    for fb in range(nfreq): 
        fmin=config_monito.freq[fb]
        fmax=config_monito.freq[fb+1]
        c=cvel[fb]
        
        fmsv_mean_single = np.zeros((1,half_npts+1))
        fmsv_mean_single[0] = fmsv_mean[0,fb+1,:] # get the mean-squared value on the targeted frequency band
        
        coda_single_band = new_twin[:,fb,:]
        # parameters for getting the sum of squared residuals (SSR) between Eobs and Esyn 
        para={ 'fmin':fmin, 'fmax':fmax, 'vdist':vdist, 'npts':npts, 'dt':dt, 'cvel':c,  \
                'mfp':mfpx, 'intb':intby, 'twin':coda_single_band, 'fmsv':fmsv_mean_single, \
                'SSR':SSR[fb] , 'sta':fname }
        # call function get_SSR
        result_intb[fb], result_mfp[fb], Eobs[fnum-1, fb], Esyn[fnum-1, fb], scaling_amp[fb] = get_optimal_Esyn(para)
        
        # plotting fitting results    
        if nwindows==1:
            plot_singwindow_fitting_result(result_mfp[fb], result_intb[fb], fmsv_mean[fnum-1,0,:], Eobs[fnum-1, fb], Esyn[fnum-1, fb],
                            target_sta, vdist, coda_single_band[0], fmin, fmax, nwindows)  
        else:
            plot_multiwindow_fitting_result(result_mfp[fb], result_intb[fb], fmsv_mean[fnum-1,0,:], Eobs[fnum-1, fb], Esyn[fnum-1, fb],
                            target_sta, vdist, coda_single_band[0], fmin, fmax, nwindows)  

        fwcen=(fmin+fmax)/2 *(2*np.pi)
        intQ=fwcen/result_intb[fb]

        #line='netst,stlo,stla,timerange,fband,fmin,fmax,fwcen,intb,intQ,mean_free,scaling_amp,tbeg,tend,noiselevel,madratio,data_path\n'
        line=f'{target_sta},{stlo:.6f},{stla:.6f},{timerange},{fmin}-{fmax},{fmin},{fmax},{fwcen:.2f},' \
                f'{result_intb[fb]:.3f},{intQ:.3f},{result_mfp[fb]:.1f},{scaling_amp[fb,0]:.6f},' \
                f'{coda_single_band[0,0,0]},{coda_single_band[0,0,1]},{level[aa,fb,0]:.8f},{config_monito.ratio},{data_path}\n'
        print(line)
        file.write(line)
file.close()
file_sta.close()