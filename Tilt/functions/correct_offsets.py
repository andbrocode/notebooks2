#!/bin/python3
#
# correct for known manual resets of tiltmeter components (=recentering) 
# and also data gaps when one or both components were clipping
#

def __correct_offsets(st, offset_correction, plot=False):
    
    from numpy import nanmedian, nanmean, nan
    from obspy import UTCDateTime, Stream
    
    st_out = Stream()
    
    for cc in ["N", "E", "T"]:

        if cc not in offset_correction.keys():
            st_out += st.select(channel=f"*{cc}").copy()
            continue
        
        st0 = st.select(channel=f"*{cc}").copy()

        tbeg, tend = st0[0].stats.starttime, st0[0].stats.endtime
        
        for nn in range(len(offset_correction[cc])):
            nn +=1

            if offset_correction[cc][nn]['time_reset'] < tbeg or offset_correction[cc][nn]['time_reset'] > tend:
                continue
                        
            step_time = UTCDateTime(offset_correction[cc][nn]['time_reset'])
            offset_time_before = offset_correction[cc][nn]['time_before']
            offset_time_after = offset_correction[cc][nn]['time_after']

            st0_before = st0.copy()
            st0_before.trim(tbeg, step_time-offset_time_before)

            st0_after  = st0.copy()
            st0_after.trim(step_time+offset_time_after, tend)    

            median_before = nanmean(st0_before[0].data[-100:])
            median_after  = nanmean(st0_after[0].data[:100]) 
            
            
            if (median_after - median_before) < 0: 
                st0_after[0].data += abs(median_after - median_before)
            elif (median_after - median_before) >= 0: 
                st0_after[0].data -= abs(median_after - median_before)

            st0_before += st0_after
            
            st0 = st0_before.merge(fill_value=nan)

        st_out += st0
        
#     st_out.trim(tbeg, tend, nearest_sample=False)

    if plot:
        fig, ax = plt.subplots(3,1, figsize=(15,8), sharex=True)

        ax[0].plot(st.select(channel="*N")[0].times()/86400, st.select(channel="*N")[0].data, label="before")
        ax[0].plot(st_out.select(channel="*N")[0].times()/86400, st_out.select(channel="*N")[0].data, label="after")

        ax[1].plot(st.select(channel="*E")[0].times()/86400, st.select(channel="*E")[0].data, label="before")
        ax[1].plot(st_out.select(channel="*E")[0].times()/86400, st_out.select(channel="*E")[0].data, label="after")

        ax[2].plot(st.select(channel="*T")[0].times()/86400, st.select(channel="*T")[0].data, label="before")
        ax[2].plot(st_out.select(channel="*T")[0].times()/86400, st_out.select(channel="*T")[0].data, label="after")

        ax[0].set_ylabel("MAN (counts)")
        ax[1].set_ylabel("MAE (counts)")
        ax[2].set_ylabel("MAT (counts)")
        ax[2].set_xlabel("Time (days)")
        
        for i in range(2):
            ax[i].legend(loc=1)
        
        plt.show();

    return st_out

## End of File