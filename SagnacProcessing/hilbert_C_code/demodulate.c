/***************************************************************************
 * demodulation.c
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
#include <math.h>
#include "cwp.h"


void de_modulate (double *datasamples,long numsamples,int oversamp_rate,float flow, float fhigh,double samp_rate, double carrier, float freqres);
double third_order_derive(double *x, int pos);
void spr_bp_fast_bworth(double *tr, int ndat, double tsa, float fc, float fhi, int ns, int zph);
void lp_bworth(double *tr, int ndat, double tsa, float flo, int ns, int zph);



/***************************************************************************
 * de_modulate:
 *
 * digital demodulation of RingLaser signal via the computation of the 
 * instantanuous frequency (derivative of x H[x] and envelope
 *
 ***************************************************************************/
void de_modulate (double *datasamples,long numsamples,int oversamp_rate,float flow, float fhigh,double samp_rate, double carrier,float freqres)
{
    int             l,k;
    double   **extra_trace;
    double   norm;
	double 	ave;
	double  overfreq;
    int             pos;
	long		nsamp;
	long	oversamp;

        /*******************************************/
        /* allocate 5 extra traces of equal length */
        /* as the original traces                  */
        /*******************************************/
	nsamp = numsamples;
	oversamp =  numsamples*oversamp_rate;
	overfreq =  samp_rate*oversamp_rate;
        extra_trace = (double **)calloc(6, sizeof(double *));
        for (k=0;k<5;k++) {
                extra_trace[k]  = (double *)calloc(oversamp, sizeof(double));
        }
	//here we remove a possible offset 
	ave = 0.;
	for(l=0;l<nsamp;l++) {
		ave += (double)datasamples[l]/(double)nsamp;
	}
	// upsampling
	for(l=0;l<nsamp;l++) extra_trace[0][l*oversamp_rate] = (double)datasamples[l] - ave;
    lp_bworth(extra_trace[0], oversamp, 1./overfreq, samp_rate/2.1, 2, 1);

/* we filter first the signal to reduce the effective bandwidth */
	if(fhigh > 0.){
	    spr_bp_fast_bworth(extra_trace[0], oversamp, 1./overfreq, flow, fhigh, 2, 1);
	}
        /*************************************************/
        /* make the hilbert transforms first             */
        /*************************************************/
   hilbert(oversamp,extra_trace[0],extra_trace[1]);

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
		*(datasamples+l) = (float)((extra_trace[4][oversamp_rate*l]-carrier)/freqres);
	}

	for(l=0;l<5;l++)
		free((void *)extra_trace[l]);
        free((void *)extra_trace);

}  /* End of de_modulate() */


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
    double         coeff[5][5];
    double         norm[2];
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
void spr_bp_fast_bworth(double *tr, int ndat, double tsa, float flo, float fhi, int ns, int zph)
{
    int k;                   /* index */
    int n,m,mm;
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

/**
**/
void lp_bworth(double *tr, int ndat, double tsa, float fc, int ns, int zph)
{
    int k;                       /* index */
    int n,m,mm;
    double a[MAX_SEC+1];
    double b[MAX_SEC+1];
    double c[MAX_SEC+1];
    double f[MAX_SEC+1][6];

    double temp;
    double wcp,cs,x;

    /* design filter weights */
    wcp=sin(fc*PI*tsa)/cos(fc*PI*tsa);
    for (k=1;k<=ns;k++)
    {
            cs=cos((2.0*(k+ns)-1.0)*PI/(4.0*ns));
            x=1.0/(1.0+wcp*wcp -2.0*wcp*cs);
            a[k]=wcp*wcp*x;
            b[k]=2.0*(wcp*wcp-1.0)*x;
            c[k]=(1.0 +wcp*wcp +2.0*wcp*cs)*x;
    }
    /* set initial values to 0 */
    for(n=0;n<=MAX_SEC;n++)
    {
            for(m=0;m<=5;m++)
            {
                    f[n][m]=0.0;
            }
    }
    /* set initial values to 0 */
    /* filtering */
    for (m=1;m<=ndat;m++)
    {
            f[1][3]= *(tr+m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][3]+2.0*f[n][2] +f[n][1]);
                    f[n+1][3]=temp-b[n]*f[n+1][2]-c[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=2;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1)=f[ns+1][3];
    }
    if (zph == TRUE)
    {

        /* filtering reverse signal*/
        for (m=ndat;m>=1;m--)
        {
            f[1][3]= *(tr+m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][3]+2.0*f[n][2] +f[n][1]);
                    f[n+1][3]=temp-b[n]*f[n+1][2]-c[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=2;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1)=f[ns+1][3];
        }
    }
}
