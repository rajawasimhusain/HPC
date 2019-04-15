#include "hm_win32_posix.h"

#ifdef _WIN32
void
usleep (long usec)
{
	if(usec >= 1000)
	{
		Sleep(usec/1000);
	}
	else
	{	
		LARGE_INTEGER lFrequency;
		LARGE_INTEGER lEndTime;
		LARGE_INTEGER lCurTime;
		
		QueryPerformanceFrequency (&lFrequency);
		if (lFrequency.QuadPart)
		{
			QueryPerformanceCounter (&lEndTime);
			lEndTime.QuadPart += (LONGLONG) usec * lFrequency.QuadPart / 1000000;
			do
			{
				QueryPerformanceCounter (&lCurTime);
				Sleep(0);			
			} while (lCurTime.QuadPart < lEndTime.QuadPart);
		}
	}
}


int 
gettimeofday (struct timeval *tp, void *tz)
{
	struct _timeb timebuffer;
	_ftime_s(&timebuffer);
	
	tp->tv_sec = (long)timebuffer.time;
	tp->tv_usec = (long)timebuffer.millitm * 1000;
	
	return 0;
}

_uint64 
timespec2nsec( const struct timespec* ts )
{
	_uint64 nsec = (_uint64)((_uint64)(ts->tv_sec*1000000000) + (ts->tv_nsec));
	return nsec;
}

int 
clock_gettime(clockid_t clock_id, struct timespec *tp)
{
	// We don't manage CLOCK_REALTIME here as we are exclusively use this option
	LARGE_INTEGER lFrequency;
	LARGE_INTEGER lCurTime;
	_uint64 t_sec;
	_uint64 t_nsec;

	QueryPerformanceFrequency (&lFrequency);
	
	if (lFrequency.QuadPart)
	{
		QueryPerformanceCounter (&lCurTime);
		
		t_sec = lCurTime.QuadPart / lFrequency.QuadPart;
		t_nsec = lCurTime.QuadPart*1000000000 / lFrequency.QuadPart;
						
		tp->tv_sec = (long)t_sec;
		tp->tv_nsec = (long)(t_nsec-t_sec);		
	}

	return 0;
}

#endif
