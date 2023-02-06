/***************************************************************************
 * saniac_plugin.c
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <regex.h>
#include <math.h>
#include "libslink.h"
#include <plugin.h>
#include "seedutil.h"
#include "cwp.h"


static SLCD * slconn;               /* connection parameters */

#define PACKAGE "sagnac_plugin"
#define VERSION "2017.114"

#define RECSIZE 512

static char  stopsig    = 0;    /* Stop/termination signal */
static int   verbose    = 0;    /* Verbosity level */
static int   stateint   = 300;  /* State saving interval in seconds */
static char *statefile  = 0;    /* State file for saving/restoring time stamps */
static char *map_station;
static int32_t *datasamples;
static int32_t *rest_samples;
static double freqres = 0.1;
static long backsamples;
static double filt_resp;
static int first;
	long nfft;
static long curr_samp = 0;
static long *ndat;
static long counter = 0;
static float flow = 0;
static float fhigh = 0;
static double first_dtime;

static void packet_handler (char *msrecord, int packet_type, int seqnum, int packet_size);
void de_modulate (int32_t *datasamples,long num_samples, double samp_rate, double carrier, long max);
static int parameter_proc (int argcount, char **argvec);
double third_order_derive(double *x, int pos);
static void  term_handler();
static int   lprintf (int level, const char *fmt, ...);
void spr_bp_fast_bworth(double *tr, long ndat, double tsa, float flo, float fhi, int ns, int zph);
static void usage (void);
static double carrier = 0;
static int first = 0;


int
main (int argc, char** argv)
{
	  SLpacket * slpack;
	  int seqnum;
	  int ptype;

  time_t statetime;

  /* Signal handling using POSIX routines */
  struct sigaction sa;
  
  sa.sa_flags = SA_RESTART;
  sigemptyset(&sa.sa_mask);

  sigaction(SIGUSR1, &sa, NULL);
  
  sa.sa_handler = term_handler;
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);
  
  sa.sa_handler = SIG_IGN;
  sigaction(SIGHUP, &sa, NULL);
  sigaction(SIGPIPE, &sa, NULL);

  /* Allocate and initialize a new connection description */
  slconn = sl_newslcd();
  
  
  /* Process command line parameters */
  if (parameter_proc (argc, argv) < 0){
      fprintf(stderr, "Parameter processing failed\n\n");
      fprintf(stderr, "Try '-h' for detailed help\n");
      return -1;
  }
  
  statetime = time(NULL);
  while (stopsig == 0){
    /* Loop with the connection manager */
    while ( sl_collect (slconn, &slpack) )
    {
         ptype  = sl_packettype (slpack);
	 seqnum = sl_sequence (slpack);
	 packet_handler ((char *) &slpack->msrecord, ptype, seqnum, SLRECSIZE);

     }
  }

      /* Make sure everything is shut down and save the state file */
      if (slconn->link != -1)
	          sl_disconnect (slconn);

        if (statefile)
		    sl_savestate (slconn, statefile);

	  return 0;
}                               /* End of main() */

  

/***************************************************************************
 * packet_handler:
 * 
 * Here we do all the things necessary for demodulation
 *
 * 
 ***************************************************************************/
static void packet_handler (char *msrecord, int packet_type, int seqnum, int packet_size)
{
  static MSrecord * msr; 

  double dtime;                 /* Epoch time */
  struct ptime pt;
  double secfrac;               /* Fractional part of epoch time */
  time_t itime;                 /* Integer part of epoch time */
  char chan[4];
  char map_chan[9];
  char timestamp[20];
  struct tm *timep;
  double samp_rate;
  int len;
  char *ptr;
  int32_t *buffer;
  char station[10];
  int timing_quality;
  long next_fft;
  int i;
  int32_t ave;
  long long max;


  /* The following is dependent on the packet type values in libslink.h */
  char *type[]  = { "Data", "Detection", "Calibration", "Timing",
                    "Message", "General", "Request", "Info",
                    "Info (terminated)", "KeepAlive" };

  /* Build a current local time string */
  dtime   = sl_dtime ();
  secfrac = (double) ((double)dtime - (int)dtime);
  itime   = (time_t) dtime;
  timep   = localtime (&itime);
  snprintf (timestamp, 20, "%04d.%03d.%02d:%02d:%02d.%01.0f",
            timep->tm_year + 1900, timep->tm_yday + 1, timep->tm_hour,
            timep->tm_min, timep->tm_sec, secfrac);

  /* Process waveform data */
  if ( packet_type == SLDATA )
    {
        msr = msr_new();

#if 0
        sl_log (0, 1, "%s, seq %d, Received %s blockette:\n",
             timestamp, seqnum, type[packet_type]);
#endif
        msr_parse(NULL, msrecord, &msr, 1, 1);
        msr_dsamprate(msr,&samp_rate);
        dtime = msr_depochstime(msr);
	if(first == 0){
	 filt_resp = 1./(fabs(fhigh) - fabs(flow));
	 filt_resp *= 10;
	 backsamples = (int)(filt_resp * samp_rate);
	 backsamples = 5*(int)((float)backsamples/2. + 0.5);
	 filt_resp = (double)backsamples/samp_rate;
	 /* copy back the first two filter response times */
	 /* ++++++++++++++++++++-- */
	 /*                   --+++++++++++++++++++++-- */
	 rest_samples = (int32_t *)calloc(2*backsamples,sizeof(int32_t));
	}
	/* we have to estimate the length of the time window */
	if(counter == 0){
  		nfft = (long)(1./freqres * samp_rate);
		if(nfft > 1024*1024*6){
			nfft = 1024*1024*6;
                       fprintf(stderr,"%ld nfft which translates in a %f s long time window\n",nfft,nfft/samp_rate);
		}else{
  		   next_fft = 2;
  		   while(next_fft < nfft) next_fft *= 2;
  		   nfft = next_fft;
                   fprintf(stderr,"%ld nfft which translates in a %f s long time window\n",nfft,nfft/samp_rate);
		}
//  		datasamples = (int32_t *) calloc(nfft+2*backsamples+2*512,sizeof(int32_t));
  		datasamples = (int32_t *) calloc(nfft*3+2*backsamples,sizeof(int32_t));
                if(datasamples == NULL){
                    fprintf(stderr,"Memory not allocated  \n");
                }
	        if(first != 0){
		   memcpy((void *)datasamples,(void *)rest_samples,2*backsamples*sizeof(int32_t));
		}
	}

        timing_quality = 100;
        if(msr->Blkt1001 != NULL)
            timing_quality = msr->Blkt1001->timing_qual;

        if(msr->numsamples < 0 || msr->numsamples > 512)
          {
            fprintf(stderr,"%d error decoding Mini-SEED packet \n",msr->numsamples );
            msr_free(&msr);
            return;
          }
//	if ((curr_samp-2*backsamples) < (nfft-2*msr->numsamples)){
	if ((curr_samp-2*backsamples) < (nfft)){
		if(counter == 0) first_dtime = dtime;
		counter++;
        	for(i = 0; i < msr->numsamples; i++)
        	{
            		datasamples[curr_samp+i] = msr->datasamples[i];
         	}
		curr_samp += msr->numsamples;
	}else{
		/* we have nearly all  packets already so we must use int now */
        	for(i = 0; i < msr->numsamples; i++)
        	{
            		datasamples[curr_samp+i] = msr->datasamples[i];
         	}
		curr_samp += msr->numsamples;
		memcpy((void *)rest_samples,(void *)(datasamples+(curr_samp-2*backsamples)),2*backsamples*sizeof(int32_t));
                fprintf(stderr,"%ld samples collected \n",curr_samp);
		ave = 0;
		max = 0;
#if 0
		for(i=0;i<curr_samp;i++){
			ave += datasamples[i]/(curr_samp);
		}
		for(i=0;i<(curr_samp);i++){
			datasamples[i] -=ave;
		}
		/* we will also remove the linear trend */

		for(i=0;i<(curr_samp);i++){
			if(max < fabs(datasamples[i])) max = fabs(datasamples[i]);
		}
#endif
                fprintf(stderr,"ready to enter demodulation\n");

        	de_modulate(datasamples,curr_samp,samp_rate,carrier,max);
		counter = 0;

      		sl_log (0, 2, "Mapped to %s,  %s \n", msr->fsdh.station, chan);
      		strcpy(station,map_station);
      		len = strlen(station);
      		station[len] = '\0';
      		strncpy(chan,msr->fsdh.channel,3);
      		chan[3] = '\0';

	 	if(first == 0){
     			len =  send_raw_depoch(station,chan,first_dtime,0,timing_quality,datasamples,curr_samp-backsamples);
			first ++;
		}else{
			first_dtime -= filt_resp;
			buffer = (int32_t *)calloc((curr_samp-2*backsamples),sizeof(int32_t));
			memcpy((void *)buffer,(void *)(datasamples+backsamples),(curr_samp-2*backsamples)*sizeof(int32_t));
     			len =  send_raw_depoch(station,chan,first_dtime,0,timing_quality,buffer,(curr_samp-2*backsamples));
			free((void *)buffer);
		}
			
		curr_samp = 2*backsamples;
		free((void *)datasamples);
	}
     msr_free(&msr);

    }
  else if ( packet_type == SLKEEP )
    {
      sl_log (0, 2, "Keep alive packet received\n");
    }
  else
    {
      sl_log (0, 1, "%s, seq %d, Received %s blockette\n",
              timestamp, seqnum, type[packet_type]);
    }
}                               /* End of packet_handler() */


/***************************************************************************
 * de_modulate:
 *
 * digital demodulation of RingLaser signal via the computation of the 
 * instantanuous frequency (derivative of x H[x] and envelope
 *
 ***************************************************************************/
void de_modulate (int32_t *datasamples,long num_samples,double samp_rate, double carrier, long max)
{
        long             l,k;
        double   **extra_trace;
        double   norm;
	double 	ave;
	double 	nava;
        long overrate=2;
	double  overfreq;
        long             pos;
	double digfreq;
	long		nsamp;
	long	oversamp;
	double omax;

        /*******************************************/
        /* allocate 5 extra traces of equal length */
        /* as the original traces                  */
        /*******************************************/
	nsamp = num_samples;
	digfreq = samp_rate;
	oversamp =  num_samples*overrate;
	overfreq =  samp_rate*overrate;
        extra_trace = (double **)calloc(6, sizeof(double *));
        
        for (k=0;k<5;k++) {
            extra_trace[k]  = (double *)calloc(oversamp, sizeof(double));
            if (extra_trace[k] == NULL){
                fprintf(stderr,"Memory for trace %ld is not allocated\n",k);
            }
 
        }
	//here we remove the offset 
	ave = 0.;
	for(l=0;l<nsamp;l++) {
		ave += (double)datasamples[l]/(double)nsamp;
	}

	// upsampling factor overrate
	for(l=0;l<nsamp;l++) extra_trace[0][l*overrate] = (double)datasamples[l] - ave;

/* we filter first the signal to reduce the effective bandwidth */
	if(fhigh > 0.){
	    spr_bp_fast_bworth(extra_trace[0], oversamp, 1./overfreq, flow, fhigh, 2, 1);
	}
        /*************************************************/
        /* make the hilbert transforms first             */
        /*************************************************/
        hilbert(oversamp,extra_trace[0],extra_trace[1]);

	/***************************************/
	/* in order to equilze amplitude we    */
	/* calculate envelope  and normalize it*/
	/****************************************/
	ave = 0.0; 

        for (l=0;l<oversamp;l++) {
            if (l>1 && l<oversamp-2) {
                    pos = 2;
            } else {
                if (l<2) {
                     pos = l;
                } else {
                   if (l==oversamp-2) {
                      pos = 3;
                   }
                   if (l==oversamp-1) {
                     pos = 4;
                   }
                }
            }
            extra_trace[2][l] = third_order_derive(extra_trace[0]+l-pos,pos)*overfreq;
            extra_trace[3][l] = third_order_derive(extra_trace[1]+l-pos,pos)*overfreq;
	    // f = (x*dH[x] - H[x]*dx)/(2 Pi (X^2+H[x]^2)) that is the inst. freq estimation!!!
            norm = sqrt(extra_trace[0][l]*extra_trace[0][l]+extra_trace[1][l]*extra_trace[1][l]);
            extra_trace[4][l] = -1.*(extra_trace[0][l]*extra_trace[3][l]-extra_trace[1][l]*extra_trace[2][l])/(2.0*M_PI*norm*norm);
	    ave += extra_trace[4][l]/(double)oversamp;
        }
	fprintf(stderr,"Average: %lf\n",ave);
	/* making of counts and downsampling */
	for(l=0;l<nsamp;l++){
		datasamples[l] = (int32_t)((extra_trace[4][overrate*l]-carrier)/freqres);
	}

	for(l=0;l<5;l++)
		free((void *)extra_trace[l]);
        free((void *)extra_trace);

}  /* End of de_modulate() */


/***************************************************************************
 *  * parameter_proc:
 *   *
 *    * Process the command line parameters.
 *     *
 *      * Returns 0 on success, and -1 on failure
 *       ***************************************************************************/
static int
parameter_proc (int argcount, char **argvec)
{
  int optind;
  int error = 0;

  char *streamfile  = 0;        /* stream list file for configuring streams */
  char *multiselect = 0;
  char *selectors   = 0;

  if (argcount <= 1)
    error++;

  /* Process all command line arguments */
  for (optind = 1; optind < argcount; optind++)
    {
      if (strcmp (argvec[optind], "-V") == 0)
        {
          fprintf(stderr, "%s version: %s\n", PACKAGE, VERSION);
          exit (0);
        }
      else if (strcmp (argvec[optind], "-h") == 0)
        {
          usage();
          exit (0);
        }
      else if (strncmp (argvec[optind], "-v", 2) == 0)
        {
          verbose += strspn (&argvec[optind][1], "v");
        }
      else if (strcmp (argvec[optind], "-nt") == 0)
        {
          slconn->netto = atoi (argvec[++optind]);
        }
      else if (strcmp (argvec[optind], "-nd") == 0)
        {
          slconn->netdly = atoi (argvec[++optind]);
        }
      else if (strcmp (argvec[optind], "-k") == 0)
        {
          slconn->keepalive = atoi (argvec[++optind]);
        }
      else if (strcmp (argvec[optind], "-F") == 0)
        {
          carrier = (double) (atof (argvec[++optind]));
        }
      else if (strcmp (argvec[optind], "-fq") == 0)
        {
          freqres = atof (argvec[++optind]);
        }
      else if (strcmp (argvec[optind], "-M") == 0)
        {
          map_station = argvec[++optind];
        }
      else if (strcmp (argvec[optind], "-l") == 0)
        {
          streamfile = argvec[++optind];
        }
      else if (strcmp (argvec[optind], "-fh") == 0)
        {
          fhigh = atof(argvec[++optind]);
        }
      else if (strcmp (argvec[optind], "-fl") == 0)
        {
          flow = atof(argvec[++optind]);
        }
      else if (strcmp (argvec[optind], "-s") == 0)
        {
          selectors = argvec[++optind];
        }
      else if (strcmp (argvec[optind], "-S") == 0)
        {
          multiselect = argvec[++optind];
        }
      else if (strcmp (argvec[optind], "-x") == 0)
        {
          statefile = argvec[++optind];
        }
      else if (strcmp (argvec[optind], "-P") == 0)
        {
          slconn->sladdr = argvec[++optind];
        }
#if 0
      else if (strncmp (argvec[optind], "-", 1 ) == 0)
        {
          fprintf(stderr, "Unknown option: %s\n", argvec[optind]);
          exit (1);
        }
      else
        {
          fprintf(stderr, "Unknown option: %s\n", argvec[optind]);
          exit (1);
        }
#endif
      else if (!slconn->sladdr)
        {
        }
    }

  /* Make sure a server was specified */
  if ( ! slconn->sladdr )
    {
      fprintf(stderr, "No SeedLink server specified\n\n");
      fprintf(stderr, "Usage: %s [options] [host][:port]\n", PACKAGE);
      fprintf(stderr, "We try the local one\n");
      strcpy(slconn->sladdr,"127.0.0.1:18000");
    }

  /* Initialize the verbosity for the sl_log function */
  sl_loginit (verbose, NULL, NULL, NULL, NULL);

  /* Report the program version */
  sl_log (0, 1, "%s version: %s\n", PACKAGE, VERSION);

  /* If errors then report the usage message and quit */
  if (error)
    {
      usage ();
      exit (1);
    }

  /* Load the stream list from a file if specified */
  if ( streamfile )
    sl_read_streamlist (slconn, streamfile, selectors);

  /* Parse the 'multiselect' string following '-S' */
  if ( multiselect )
    {
      if ( sl_parse_streamlist (slconn, multiselect, selectors) == -1 )
        return -1;
    }
  else if ( !streamfile )
    {                    /* No 'streams' array, assuming uni-station mode */
      sl_setuniparams (slconn, selectors, -1, 0);
    }

  /* Attempt to recover sequence numbers from state file */
  if (statefile)
    {
      if (sl_recoverstate (slconn, statefile) < 0)
        {
          sl_log (2, 0, "state recovery failed\n");
        }
    }

  return 0;
}                               /* End of parameter_proc() */



/***************************************************************************
 *  * usage:
 *   * Print the usage message and exit.
 *    ***************************************************************************/
static void
usage (void)
{
  fprintf (stderr, "\nUsage: %s [options] [host][:port]\n\n", PACKAGE);
  fprintf (stderr,
           " ## General program options ##\n"
           " -V             report program version\n"
           " -h             show this usage message\n"
           " -v             be more verbose, multiple flags can be used\n"
           " -p             print details of data packets\n\n"
           " -nd delay      network re-connect delay (seconds), default 30\n"
           " -nt timeout    network timeout (seconds), re-establish connection if no\n"
           "                  data/keepalives are received in this time, default 600\n"
           " -k interval    send keepalive (heartbeat) packets this often (seconds)\n"
           " -x statefile   save/restore stream state information to this file\n"
           "\n"
	   " -F carrier     carrier frequency in Hz of the FM signal\n"
	   " -fq freqres     desired frequency resolution - needed by the int32 format\n"
	   " -M map_station stream selector to map to \n"
           " ## Data stream selection ##\n"
           " -l listfile    read a stream list from this file for multi-station mode\n"
           " -fl bp low     lower corner frequency of bandpass (ignored when negative)\n"
           " -fh bp high    upper corner frequency of bandpass (ignored when negative\n"
           " -s selectors   selectors for uni-station or default for multi-station\n"
           " -S streams     select streams for multi-station (requires SeedLink >= 2.5)\n"
           "   'streams' = 'stream1[:selectors1],stream2[:selectors2],...'\n"
           "        'stream' is in NET_STA format, for example:\n"
           "        -S \"IU_KONO:BHE BHN,GE_WLF,MN_AQU:HH?.D\"\n\n"
           "\n"
           " -P [host][:port]  Address of the SeedLink server in host:port format\n"
           "                  if host is omitted (i.e. ':18000'), localhost is assumed\n"
           "                  if :port is omitted (i.e. 'localhost'), 18000 is assumed\n\n");

}                               /* End of usage() */


/***************************************************************************
 * getoptval:
 * Return the value to a command line option; checking that the value is 
 * itself not an option (starting with '-') and is not past the end of
 * the argument list.
 *
 * argcount: total arguments in argvec
 * argvec: argument list
 * argopt: index of option to process, value is expected to be at argopt+1
 *
 * Returns value on success and exits with error message on failure
 ***************************************************************************/
static char *
getoptval (int argcount, char **argvec, int argopt)
{
  if ( argvec == NULL || argvec[argopt] == NULL )
    {
      lprintf (0, "getoptval(): NULL option requested");
      exit (1);
    }
  
  if ( (argopt+1) < argcount && *argvec[argopt+1] != '-' )
    return argvec[argopt+1];
  
  lprintf (0, "Option %s requires a value", argvec[argopt]);
  exit (1);
}  /* End of getoptval() */



/***************************************************************************
 * term_handler and print_handler:
 * Signal handler routines.
 ***************************************************************************/
static void
term_handler (int sig)
{
  stopsig = 1;
}


/***************************************************************************
 * lprintf:
 *
 * A generic log message handler, pre-pends a current date/time string
 * to each message.  This routine add a newline to the final output
 * message so it should not be included with the message.
 *
 * Returns the number of characters in the formatted message.
 ***************************************************************************/
static int
lprintf (int level, const char *fmt, ...)
{
  int rv = 0;
  char message[100];
  va_list argptr;
  struct tm *tp;
  time_t curtime;
  
  char *day[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
  char *month[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
		   "Aug", "Sep", "Oct", "Nov", "Dec"};

  if ( level <= verbose ) {
    
    /* Build local time string and generate final output */
    curtime = time(NULL);
    tp = localtime (&curtime);
    
    va_start (argptr, fmt);
    rv = vsnprintf (message, sizeof(message), fmt, argptr);
    va_end (argptr);
    
    printf ("%3.3s %3.3s %2.2d %2.2d:%2.2d:%2.2d %4.4d - %s: %s\n",
	    day[tp->tm_wday], month[tp->tm_mon], tp->tm_mday,
	    tp->tm_hour, tp->tm_min, tp->tm_sec, tp->tm_year + 1900,
	    PACKAGE, message);
    
    fflush (stdout);
  }
  
  return rv;
}  /* End of lprintf() */


#define LEFT2 0
#define LEFT1 1
#define MIDDLE 2
#define RIGHT1 3
#define RIGHT2 4

/********************************************************************/
/* numerical derivation by 5 point 3rd order polynomial             */
/* p. 115 Signalverarbeitung E. Schruefer, Carl Hanser Verlag, 1992 */
/* returns result for h = 1!!! this means correct division has to   */
/* be applied afterwards!!!!!!!!!!!!!!!!!!!!!!!!!!!                 */
/********************************************************************/

double third_order_derive(double *x, int pos)
{
    double static        coeff[5][5];
    double static        norm[2];
    int                 i;
    double               sum = 0.;

    norm[0] = 12.;
    norm[1] = 84.;

    coeff[0][0] = -125.;
    coeff[0][1] =  136.;
    coeff[0][2] =   48.;
    coeff[0][3] =  -88.;
    coeff[0][4] =   29.;
    coeff[1][0] =  -38.;
    coeff[1][1] =   -2.;
    coeff[1][2] =   24.;
    coeff[1][3] =   26.;
    coeff[1][4] =  -10.;
    coeff[2][0] =    1.;
    coeff[2][1] =   -8.;
    coeff[2][2] =    0.;
    coeff[2][3] =    8.;
    coeff[2][4] =   -1.;
    coeff[3][0] =   10.;
    coeff[3][1] =  -26.;
    coeff[3][2] =  -24.;
    coeff[3][3] =    2.;
    coeff[3][4] =   38.;
    coeff[4][0] =  -29.;
    coeff[4][1] =   88.;
    coeff[4][2] =  -48.;
    coeff[4][3] = -136.;
    coeff[4][4] =  125.;

    for (i=0;i<5;i++) {
        sum += coeff[pos][i]*x[i];
    }
    if (pos == MIDDLE) {
        sum /= norm[0];
    }
    else {
        sum /= norm[1];
    }

    return (sum);
}

#if 0
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

void four1(double data[], unsigned long nn, int isign)
{
        unsigned long n,mmax,m,j,istep,i;
        double wtemp,wr,wpr,wpi,wi,theta;
        double tempr,tempi;

        n=nn << 1;
        j=1;
        for (i=1;i<n;i+=2) {
                if (j > i) {
                        SWAP(data[j],data[i]);
                        SWAP(data[j+1],data[i+1]);
                }
                m=n >> 1;
                while (m >= 2 && j > m) {
                        j -= m;
                        m >>= 1;
                }
                j += m;
        }
        mmax=2;
        while (n > mmax) {
                istep=mmax << 1;
                theta=isign*(6.28318530717959/mmax);
                wtemp=sin(0.5*theta);
                wpr = -2.0*wtemp*wtemp;
                wpi=sin(theta);
                wr=1.0;
                wi=0.0;
                for (m=1;m<mmax;m+=2) {
                        for (i=m;i<=n;i+=istep) {
                                j=i+mmax;
                                tempr=wr*data[j]-wi*data[j+1];
                                tempi=wr*data[j+1]+wi*data[j];
                                data[j]=data[i]-tempr;
                                data[j+1]=data[i+1]-tempi;
                                data[i] += tempr;
                                data[i+1] += tempi;
                        }
                        wr=(wtemp=wr)*wpr-wi*wpi+wr;
                        wi=wi*wpr+wtemp*wpi+wi;
                }
                mmax=istep;
        }
}
#undef SWAP
#endif

#if 0
void realft(double data[], unsigned long n, int isign)
{
        void four1(double data[], unsigned long nn, int isign);
        unsigned long i,i1,i2,i3,i4,np3;
        double c1=0.5,c2,h1r,h1i,h2r,h2i;
        double wr,wi,wpr,wpi,wtemp,theta;

        theta=3.141592653589793/(double) (n>>1);
        if (isign == 1) {
                c2 = -0.5;
                four1(data,n>>1,1);
        } else {
                c2=0.5;
                theta = -theta;
        }
        wtemp=sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi=sin(theta);
        wr=1.0+wpr;
        wi=wpi;
        np3=n+3;
        for (i=2;i<=(n>>2);i++) {
                i4=1+(i3=np3-(i2=1+(i1=i+i-1)));
                h1r=c1*(data[i1]+data[i3]);
                h1i=c1*(data[i2]-data[i4]);
                h2r = -c2*(data[i2]+data[i4]);
                h2i=c2*(data[i1]-data[i3]);
                data[i1]=h1r+wr*h2r-wi*h2i;
                data[i2]=h1i+wr*h2i+wi*h2r;
                data[i3]=h1r-wr*h2r+wi*h2i;
                data[i4] = -h1i+wr*h2i+wi*h2r;
                wr=(wtemp=wr)*wpr-wi*wpi+wr;
                wi=wi*wpr+wtemp*wpi+wi;
        }
        if (isign == 1) {
                data[1] = (h1r=data[1])+data[2];
                data[2] = h1r-data[2];
        } else {
                data[1]=c1*((h1r=data[1])+data[2]);
                data[2]=c1*(h1r-data[2]);
                four1(data,n>>1,-1);
        }
}
#endif

#define MAX_SEC 10
#if 0
#define TRUE 1
#define FALSE 0
#endif

/**
 *    NAME: spr_bp_fast_bworth
 *       SYNOPSIS:
 *          float flo;          low cut corner frequency
 *             float fhi;          high cut corner frequency
 *                int ns;            number of filter sections
 *                   int zph;          TRUE -> zero phase filter
 *                      spr_bp_bworth(header1,header2,flo,fhi,ns,zph);
 *                         DESCRIPTION: Butterworth bandpass filter.
 *                         **/
void spr_bp_fast_bworth(double *tr, long ndat, double tsa, float flo, float fhi, int ns, int zph)
{
    long k;                   /* index */
    long n,m,mm;
    double a[MAX_SEC+1];
    double b[MAX_SEC+1];
    double c[MAX_SEC+1];
    double d[MAX_SEC+1];
    double e[MAX_SEC+1];
    double f[MAX_SEC+1][6];

    double temp;
    double c1,c2,c3;
    double w1,w2,wc,q,p,r,s,cs,x;


    /* design filter weights */
    /* bandpass */
    w1=sin(flo*M_PI*tsa)/cos(flo*M_PI*tsa);
    w2=sin(fhi*M_PI*tsa)/cos(fhi*M_PI*tsa);
    wc=w2-w1;
    q=wc*wc +2.0*w1*w2;
    s=w1*w1*w2*w2;
    for (k=1;k<=ns;k++)
    {
            c1=(double)(k+ns);
            c2=(double)(4*ns);
            c3=(2.0*c1-1.0)*M_PI/c2;
            cs=cos(c3);
            p = -2.0*wc*cs;
            r=p*w1*w2;
            x=1.0+p+q+r+s;
            a[k]=wc*wc/x;
            b[k]=(-4.0 -2.0*p+ 2.0*r+4.0*s)/x;
            c[k]=(6.0 - 2.0*q +6.0*s)/x;
            d[k]=(-4.0 +2.0*p -2.0*r +4.0*s)/x;
            e[k]=(1.0 - p +q-r +s)/x;
    }

    /* set initial values to 0 */
    for(n=0;n<=MAX_SEC;n++)
    {
            for(m=0;m<=5;m++)
            {
                    f[n][m]=0.0;
            }
    }
    /* filtering */
    for (m=1;m<=ndat;m++)
    {
            f[1][5]= *(tr + m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][5]-2.0*f[n][3] +f[n][1]);
                    temp=temp-b[n]*f[n+1][4]-c[n]*f[n+1][3];
                    f[n+1][5]=temp-d[n]*f[n+1][2]-e[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=4;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1) =f[ns+1][5];
    }
    if (zph == TRUE)
    {
        /* filtering reverse signal*/
        for (m=ndat;m>=1;m--)
        {
                f[1][5]= *(tr+m-1);
                /* go thru ns filter sections */
                for(n=1;n<=ns;n++)
                {
                        temp=a[n]*(f[n][5]-2.0*f[n][3] +f[n][1]);
                        temp=temp-b[n]*f[n+1][4]-c[n]*f[n+1][3];
                        f[n+1][5]=temp-d[n]*f[n+1][2]-e[n]*f[n+1][1];
                }
                /* update past values */
                for(n=1;n<=ns+1;n++)
                {
                        for(mm=1;mm<=4;mm++)
                        {
                                f[n][mm]=f[n][mm+1];
                        }
                }
                /* set present data value and continue */
                *(tr+m-1)=f[ns+1][5];
        }
    }
    return;
}
