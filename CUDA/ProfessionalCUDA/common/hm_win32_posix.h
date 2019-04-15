#ifndef _WIN32_SYSTEM_RT
#define _WIN32_SYSTEM_RT

#ifdef _WIN32
#include <Winsock2.h>
#include <windows.h>
#include <time.h>
#include <sys/timeb.h>

// Needed for "struct timespec"
// #include "pthread.h"
#ifndef HAVE_STRUCT_TIMESPEC
#define HAVE_STRUCT_TIMESPEC 1
//struct timespec {
//        long tv_sec;
//        long tv_nsec;
//};
#endif /* HAVE_STRUCT_TIMESPEC */

typedef LONGLONG _uint64;
typedef int clockid_t;



// Overload of snprintf function which does not exist in Windows. _snprintf exists instead.
// GMA: Don't know why this is not working when linking...
//__DEFINE_CPP_OVERLOAD_STANDARD_NFUNC_0_2_ARGLIST_EX(int, __RETURN_POLICY_SAME, _CRTIMP, snprintf, _snprintf, __out_z char, __out_ecount(_Count) char, _Dest, __in size_t, _Count, __in_z __format_string const char *, _Format)

#ifdef WIN32
#define SNPRINTF _snprintf
#else
#define SNPRINTF snprintf
#endif


extern void usleep (long usec);

extern int gettimeofday (struct timeval *tp, void *tz);

extern _uint64 timespec2nsec( const struct timespec* ts );


#define CLOCK_REALTIME 0

extern int clock_gettime(clockid_t clock_id, struct timespec *tp);

#endif

#endif // _WIN32_SYSTEM_RT