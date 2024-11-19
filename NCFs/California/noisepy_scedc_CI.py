# %% [markdown]
# # NoisePy SCEDC Tutorial
# 
# Noisepy is a python software package to process ambient seismic noise cross correlations. This tutorial aims to introduce the use of noisepy for a toy problem on the SCEDC data. It can be ran locally or on the cloud.
# 
# 
# The data is stored on AWS S3 as the SCEDC Data Set: https://scedc.caltech.edu/data/getstarted-pds.html
# 

from noisepy.seis import cross_correlate, stack_cross_correlations, __version__       # noisepy core functions
from noisepy.seis.io.asdfstore import ASDFCCStore, ASDFStackStore          # Object to store ASDF data within noisepy
from noisepy.seis.io.s3store import SCEDCS3DataStore # Object to query SCEDC data from on S3
from noisepy.seis.io.channel_filter_store import channel_filter
from noisepy.seis.io.datatypes import CCMethod, ConfigParameters, FreqNorm, RmResp, StackMethod, TimeNorm        # Main configuration object
from noisepy.seis.io.channelcatalog import XMLStationChannelCatalog        # Required stationXML handling object
import os
import shutil
from datetime import datetime, timezone
from datetimerange import DateTimeRange


def main():

    path = "/home/kffeng/DATA_CaSC/scedc_data_2020-2022" 

    os.makedirs(path, exist_ok=True)
    cc_data_path = os.path.join(path, "ACFs_CI")
    stack_data_path = os.path.join(path, "STACK_acfs_CI")
    S3_STORAGE_OPTIONS = {"s3": {"anon": True}}

    # %% [markdown]
    # We will work with a single day worth of data on SCEDC. The continuous data is organized with a single day and channel per miniseed (https://scedc.caltech.edu/data/cloud.html). For this example, you can choose any year since 2002. We will just cross correlate a single day.

    # %%
    # SCEDC S3 bucket common URL characters for that day.
    S3_DATA = "s3://scedc-pds/continuous_waveforms/"
    # timeframe for analysis
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 1, 1, tzinfo=timezone.utc)
    timerange = DateTimeRange(start, end)
    print(timerange)
    print("# DATA at --- ",S3_DATA)

    # %% [markdown]
    # The station information, including the instrumental response, is stored as stationXML in the following bucket

    # %%
    S3_STATION_XML = "s3://scedc-pds/FDSNstationXML/CI/"            # S3 storage of stationXML

    # %%
    # Initialize ambient noise workflow configuration
    config = ConfigParameters() # default config parameters which can be customized

    # %%
    config.start_date = start
    config.end_date = end
    
    config.samp_freq= 20  # (int) Sampling rate in Hz of desired processing (it can be different than the data sampling rate)
    config.cc_len= 3600  # (float) basic unit of data length for fft (sec)
        # criteria for data selection
    config.ncomp = 3  # 1 or 3 component data (needed to decide whether do rotation)


    config.acorr_only = True  # only perform auto-correlation or not
    config.xcorr_only = False  # only perform cross-correlation or not

    # config.inc_hours = 24 # if the data is first 

    # pre-processing parameters
    config.step= 1800.0  # (float) overlapping between each cc_len (sec)
    config.stationxml= False  # station.XML file used to remove instrument response for SAC/miniseed data
    config.rm_resp= RmResp.INV  # select 'no' to not remove response and use 'inv' if you use the stationXML,'spectrum',
    config.freqmin = 0.01
    config.freqmax = 18
    config.max_over_std  = 10  # threshold to remove window of bad signals: set it to 10*9 if prefer not to remove them

    # TEMPORAL and SPECTRAL NORMALISATION
    config.freq_norm= FreqNorm.RMA  # choose between "rma" for a soft whitenning or "no" for no whitening. Pure whitening is not implemented correctly at this point.
    config.smoothspect_N = 1  # moving window length to smooth spectrum amplitude (points)
        # here, choose smoothspect_N for the case of a strict whitening (e.g., phase_only)

    config.time_norm = TimeNorm.ONE_BIT  # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain,
        # TODO: change time_norm option from "no" to "None"
    config.smooth_N= 10  # moving window length for time domain normalization if selected (points)

    config.cc_method= CCMethod.XCORR  # 'xcorr' for pure cross correlation OR 'deconv' for deconvolution;
        # FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"

    # OUTPUTS:
    config.substack = False  # True = smaller stacks within the time chunk. False: it will stack over inc_hours
    config.substack_len = 12 * config.cc_len  # how long to stack over (for monitoring purpose): need to be multiples of cc_len
        # if substack=True, substack_len=2*cc_len, then you pre-stack every 2 correlation windows.
        # for instance: substack=True, substack_len=cc_len means that you keep ALL of the correlations

    config.maxlag= 100  # lags of cross-correlation to save (sec)
    config.single_freq = False
    
    # %%
    # For this tutorial make sure the previous run is empty
    #os.system(f"rm -rf {cc_data_path}")
    if os.path.exists(cc_data_path):
        shutil.rmtree(cc_data_path)

    # %%
    #stations = "RPV,STS,LTP,LGB,WLT,CPP,PDU,CLT,SVD,BBR".split(",") # filter to these stations
    stations = "*"
    
    # There are 2 ways to load stations: You can either pass a list of stations or load the stations from a text file.
    # TODO : will be removed with issue #270
    #config.load_stations(stations)

    # For loading it from a text file, write the path of the file in stations_file field of config instance as below
    # config.stations_file = os.path.join(os.path.dirname(__file__), "path/my_stations.txt")

    catalog = XMLStationChannelCatalog(S3_STATION_XML, storage_options=S3_STORAGE_OPTIONS) # Station catalog
    raw_store = SCEDCS3DataStore(S3_DATA, catalog, 
                                channel_filter(config.net_list, stations, ["HHE", "HHN", "HHZ"]), 
                                timerange, storage_options=S3_STORAGE_OPTIONS) # Store for reading raw data from S3 bucket
    cc_store = ASDFCCStore(cc_data_path) # Store for writing CC data

    # %%
    cross_correlate(raw_store, config, cc_store)

    # %%
    # open a new cc store in read-only mode since we will be doing parallel access for stacking
    cc_store = ASDFCCStore(cc_data_path, mode="r")
    stack_store = ASDFStackStore(stack_data_path)
    config.stack_method = StackMethod.ALL
    cc_store.get_station_pairs()
    config.stations=["*"]
    config.net_list=["*"]
    stack_cross_correlations(cc_store, stack_store, config)
    
if __name__ == '__main__':
    main()