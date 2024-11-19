from noisepy.seis import cross_correlate, stack_cross_correlations       # noisepy core functions
from noisepy.seis.io.asdfstore import ASDFCCStore, ASDFStackStore                          # Object to store ASDF data within noisepy
from noisepy.seis.io.channel_filter_store import channel_filter
from noisepy.seis.io.pnwstore import PNWDataStore
from noisepy.seis.io.datatypes import CCMethod, ConfigParameters, Channel, ChannelData, ChannelType, FreqNorm, RmResp, Station, TimeNorm, StackMethod    # Main configuration object
from noisepy.seis.io.channelcatalog import XMLStationChannelCatalog        # Required stationXML handling object
import os
from datetime import datetime, timezone
from datetimerange import DateTimeRange
import sys

def main():
    data_year=sys.argv[1]
    # "/1-fnp/pnwstore1/p-wd00/PNW2020"
    store_path=sys.argv[2]

    path = "/home/kffeng/DATA_PNW/pnw_"+str(data_year)+"/"

    os.makedirs(path, exist_ok=True)
    ccfdir = "ACFs_BH_"+str(data_year)
    stackdir = "STACK_acfs_BH_"+str(data_year)
    cc_data_path = os.path.join(path, ccfdir)
    stack_data_path = os.path.join(path, stackdir)

    # %%
    STATION_XML = "/1-fnp/pnwstore1/p-wd11/PNWStationXML/"            # storage of stationXML
    DATA = str(store_path)+"/__/"                      # __ indicates any two chars (network code)
    DB_PATH = str(store_path)+"/timeseries.sqlite"
    # timeframe for analysis
    start = datetime(int(data_year), 1, 1, tzinfo=timezone.utc)
    end = datetime(int(data_year)+1, 1, 1, tzinfo=timezone.utc)
    range = DateTimeRange(start, end)
    print(range)
    print("# DATA at --- ",DATA)
    print("# Database -- ",DB_PATH)


    # %%
    # Initialize ambient noise workflow configuration
    config = ConfigParameters() # default config parameters which can be customized

    config.sampling_rate= 20  # (int) Sampling rate in Hz of desired processing (it can be different than the data sampling rate)
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
    config.freqmin = 0.05
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

    config.maxlag= 60  # lags of cross-correlation to save (sec)
    config.single_freq = False
    
    # For this tutorial make sure the previous run is empty
    #os.system(f"rm -rf {cc_data_path}")

    # cross network, cross channel type
    #stations = allsta.split(",")
    config.stations = ["*"]
    config.networks = ["CC", "TA","BK","HW","IU","LI","NC","PN","US","XA","XC","XD","XN","XQ","XT","XU","YW","ZG"]
    config.channels = ["BH?"]
    config.start_date = start
    config.end_date = end
    
    catalog = XMLStationChannelCatalog(STATION_XML, path_format="{network}/{network}.{name}.xml")
    #catalog = XMLStationChannelCatalog(STATION_XML, path_format="{network}" + os.path.sep + "{network}.{name}.xml")
    raw_store = PNWDataStore(DATA, DB_PATH, catalog, \
                             channel_filter(config.networks, config.stations, \
                                        config.channels ), date_range=range) # Store for reading raw data from S3 bucket
    cc_store = ASDFCCStore(cc_data_path) # Store for writing CC data
    # print the configuration parameters. Some are chosen by default but we can modify them
    # print(config)
    
    # %% 
    # Save config parameters
    xcorr_config_fn=f'{path}/config_BH_{str(data_year)}.yml'
    config.save_yaml(xcorr_config_fn)
    # %%
    cross_correlate(raw_store, config, cc_store)
    
    # open a new cc store in read-only mode since we will be doing parallel access for stacking
    cc_store.get_station_pairs()
    config.stations=["*"]
    config.networks=["*"]
    config.start_date = start
    config.end_date = end
    os.system(f"rm -rf {stack_data_path}")
    cc_store = ASDFCCStore(cc_data_path, mode="r")
    config.stack_method = StackMethod.ALL
    stack_store = ASDFStackStore(stack_data_path)
    stack_cross_correlations(cc_store, stack_store, config)

    # %%
    #pairs = stack_store.get_station_pairs()
    #print(f"Found {len(pairs)} station pairs")
    #sta_stacks = stack_store.read_bulk(None, pairs) # no timestamp used in ASDFStackStore
    #plotting_modules.plot_all_moveout(sta_stacks, 'Allstack_linear', 0.1, 0.2, 'ZZ', 1)

if __name__ == '__main__':
    main()
