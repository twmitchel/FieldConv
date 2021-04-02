/*
Copyright (c) 2017, Michael Kazhdan
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef MISCELLANY_INCLUDED
#define MISCELLANY_INCLUDED

#define VERBOSE_MESSAGING

//////////////////
// OpenMP Stuff //
//////////////////
#ifdef _OPENMP
#include <omp.h>
#else // !_OPENMP
inline int omp_get_num_procs  ( void ){ return 1; }
inline int omp_get_max_threads( void ){ return 1; }
inline int omp_get_thread_num ( void ){ return 0; }
inline void omp_set_num_threads( int ){}
inline void omp_set_nested( int ){}
struct omp_lock_t{};
inline void omp_init_lock( omp_lock_t* ){}
inline void omp_set_lock( omp_lock_t* ){}
inline void omp_unset_lock( omp_lock_t* ){}
inline void omp_destroy_lock( omp_lock_t* ){}
#endif // _OPENMP

namespace Miscellany
{
	//////////////
	// HSVtoRGB //
	//////////////
	// hsv \in [0,2\pi)] x [0,1] x [0,1]
	void HSVtoRGB( const double hsv[3] , double rgb[3] )
	{
		// From FvD
		if( hsv[1]<=0.0 ) rgb[0] = rgb[1] = rgb[2] = hsv[2];
		else
		{
			double hsv0 = hsv[0];

			hsv0 = fmod( hsv0 , 2.0 * M_PI );
			if( hsv0<0.0 ) hsv0 += 2.0 * M_PI;
			hsv0 /= M_PI / 3.0;
			int i = (int)floor(hsv0);
			double f = hsv0 - i;
			double p = hsv[2] * (1. - hsv[1]);
			double q = hsv[2] * (1. - (hsv[1]*f));
			double t = hsv[2] * (1. - (hsv[1]*(1.-f)));
			switch(i) 
			{
			case 0:  rgb[0] = hsv[2] ; rgb[1] = t      ; rgb[2] = p      ; break;
			case 1:  rgb[0] = q      ; rgb[1] = hsv[2] ; rgb[2] = p      ; break;
			case 2:  rgb[0] = p      ; rgb[1] = hsv[2] ; rgb[2] = t      ; break;
			case 3:  rgb[0] = p      ; rgb[1] = q      ; rgb[2] = hsv[2] ; break;
			case 4:  rgb[0] = t      ; rgb[1] = p      ; rgb[2] = hsv[2] ; break;
			default: rgb[0] = hsv[2] ; rgb[1] = p      ; rgb[2] = q      ; break;
			}
		}
	}

	////////////////
	// Time Stuff //
	////////////////
#include <string.h>
#include <sys/timeb.h>
#if defined( _WIN32 ) || defined( _WIN64 )
#else // !_WIN32 && !_WIN64
#include <sys/time.h>
#endif // _WIN32 || _WIN64

	inline double Time( void )
	{
#if defined( _WIN32 ) || defined( _WIN64 )
		struct _timeb t;
		_ftime( &t );
		return double( t.time ) + double( t.millitm ) / 1000.0;
#else // !_WIN32 && !_WIN64
		struct timeval t;
		gettimeofday( &t , NULL );
		return t.tv_sec + double( t.tv_usec ) / 1000000;
#endif // _WIN32 || _WIN64
	}

#if 1
	struct Timer
	{
		Timer( void ){ _start = Time(); }
		double elapsed( void ) const { return Time()-_start; };
		void reset( void ){ _start = Time(); }
	protected:
		double _start;
	};
#else
#include <cstdio>
#include <ctime>
#include <chrono>
	struct Timer
	{
		Timer( void ){ _startCPUClock = std::clock() , _startWallClock = std::chrono::system_clock::now(); }
		double cpuTime( void ) const{ return (std::clock() - _startCPUClock) / (double)CLOCKS_PER_SEC; };
		double wallTime( void ) const{  std::chrono::duration<double> diff = (std::chrono::system_clock::now() - _startWallClock) ; return diff.count(); }
	protected:
		std::clock_t _startCPUClock;
		std::chrono::time_point< std::chrono::system_clock > _startWallClock;
	};
#endif

	///////////////
	// I/O Stuff //
	///////////////
#if defined( _WIN32 ) || defined( _WIN64 )
	const char FileSeparator = '\\';
#else // !_WIN
	const char FileSeparator = '/';
#endif // _WIN

#ifndef SetTempDirectory
#if defined( _WIN32 ) || defined( _WIN64 )
#define SetTempDirectory( tempDir , sz ) GetTempPath( (sz) , (tempDir) )
#else // !_WIN32 && !_WIN64
#define SetTempDirectory( tempDir , sz ) if( std::getenv( "TMPDIR" ) ) strcpy( tempDir , std::getenv( "TMPDIR" ) );
#endif // _WIN32 || _WIN64
#endif // !SetTempDirectory

#include <stdarg.h>
#include <vector>
	struct MessageWriter
	{
		char* outputFile;
		bool echoSTDOUT;
		MessageWriter( void ){ outputFile = NULL , echoSTDOUT = true; }
		void operator() ( const char* format , ... )
		{
			if( outputFile )
			{
				FILE* fp = fopen( outputFile , "a" );
				va_list args;
				va_start( args , format );
				vfprintf( fp , format , args );
				fclose( fp );
				va_end( args );
			}
			if( echoSTDOUT )
			{
				va_list args;
				va_start( args , format );
				vprintf( format , args );
				va_end( args );
			}
		}
		void operator() ( std::vector< char* >& messages  , const char* format , ... )
		{
			if( outputFile )
			{
				FILE* fp = fopen( outputFile , "a" );
				va_list args;
				va_start( args , format );
				vfprintf( fp , format , args );
				fclose( fp );
				va_end( args );
			}
			if( echoSTDOUT )
			{
				va_list args;
				va_start( args , format );
				vprintf( format , args );
				va_end( args );
			}
			// [WARNING] We are not checking the string is small enough to fit in 1024 characters
			messages.push_back( new char[1024] );
			char* str = messages.back();
			va_list args;
			va_start( args , format );
			vsprintf( str , format , args );
			va_end( args );
			if( str[strlen(str)-1]=='\n' ) str[strlen(str)-1] = 0;
		}
	};

#if 0
	///////////////
	// Exception //
	///////////////
#include <exception>
#include <string>
#ifdef VERBOSE_MESSAGING
	inline char *_MakeMessageString( const char *header , const char *fileName , int line , const char *functionName , const char *format , ... )
	{
		va_list args;
		va_start( args , format );

		// Formatting is:
		// <header> <filename> (Line <line>)
		// <header size> <function name>
		// <header size> <format message>
		char lineBuffer[25];
		sprintf( lineBuffer , "(Line %d)" , line );
		size_t _size , size=0;

		// Line 1
		size += strlen(header)+1;
		size += strlen(fileName)+1;
		size += strlen(lineBuffer)+1;

		// Line 2
		size += strlen(header)+1;
		size += strlen(functionName)+1;

		// Line 3
		size += strlen(header)+1;
		size += vsnprintf( NULL , 0 , format , args );

		char *_buffer , *buffer = new char[ size+1 ];
		_size = size , _buffer = buffer;

		// Line 1
		sprintf( _buffer , "%s " , header );
		_buffer += strlen(header)+1;
		_size -= strlen(header)+1;

		sprintf( _buffer , "%s " , fileName );
		_buffer += strlen(fileName)+1;
		_size -= strlen(fileName)+1;

		sprintf( _buffer , "%s\n" , lineBuffer );
		_buffer += strlen(lineBuffer)+1;
		_size -= strlen(lineBuffer)+1;

		// Line 2
		for( int i=0 ; i<strlen(header)+1 ; i++ ) _buffer[i] = ' ';
		_buffer += strlen(header)+1;
		_size -= strlen(header)+1;

		sprintf( _buffer , "%s\n" , functionName );
		_buffer += strlen(functionName)+1;
		_size -= strlen(functionName)+1;

		// Line 3
		for( int i=0 ; i<strlen(header)+1 ; i++ ) _buffer[i] = ' ';
		_buffer += strlen(header)+1;
		_size -= strlen(header)+1;

		vsnprintf( _buffer , _size+1 , format , args );

		return buffer;
	}

	struct Exception : public std::exception
	{
		const char *what( void ) const noexcept { return _message.c_str(); }
		template< typename ... Args >
		Exception( const char *fileName , int line , const char *functionName , const char *format , Args ... args )
		{
			char *buffer = _MakeMessageString( "[EXCEPTION]" , fileName , line , functionName , format , args ... );
			_message = std::string( buffer );
			delete[] buffer;
		}
	protected:
		std::string _message;
	};

	template< typename ... Args > void _Throw( const char *fileName , int line , const char *functionName , const char *format , Args ... args ){ throw Exception( fileName , line , functionName , format , args ... ); }
	template< typename ... Args >
	void _Warn( const char *fileName , int line , const char *functionName , const char *format , Args ... args )
	{
		char *buffer = _MakeMessageString( "[WARNING]" , fileName , line , functionName , format , args ... );
		fprintf( stderr , "%s\n" , buffer );
		delete[] buffer;
	}
	template< typename ... Args >
	void _ErrorOut( const char *fileName , int line , const char *functionName , const char *format , Args ... args )
	{
		char *buffer = _MakeMessageString( "[ERROR]" , fileName , line , functionName , format , args ... );
		fprintf( stderr , "%s\n" , buffer );
		delete[] buffer;
	}
#ifndef Warn
#define Warn( ... ) _Warn( __FILE__ , __LINE__ , __FUNCTION__ , __VA_ARGS__ )
#endif // Warn
#ifndef Throw
#define Throw( ... ) _Throw( __FILE__ , __LINE__ , __FUNCTION__ , __VA_ARGS__ )
#endif // Throw
#ifndef ErrorOut
#define ErrorOut( ... ) _ErrorOut( __FILE__ , __LINE__ , __FUNCTION__ , __VA_ARGS__ )
#endif // ErrorOut

#else // !VERBOSE_MESSAGING
	inline char *_MakeMessageString( const char *header , const char *functionName , const char *format , ... )
	{
		va_list args;
		va_start( args , format );

		size_t _size , size = vsnprintf( NULL , 0 , format , args );
		size += strlen(header)+1;
		size += strlen(functionName)+2;

		char *_buffer , *buffer = new char[ size+1 ];
		_size = size , _buffer = buffer;

		sprintf( _buffer , "%s " , header );
		_buffer += strlen(header)+1;
		_size -= strlen(header)+1;

		sprintf( _buffer , "%s: " , functionName );
		_buffer += strlen(functionName)+2;
		_size -= strlen(functionName)+2;

		vsnprintf( _buffer , _size+1 , format , args );

		return buffer;
	}
	struct Exception : public std::exception
	{
		const char *what( void ) const noexcept { return _message.c_str(); }
		template< typename ... Args >
		Exception( const char *functionName , const char *format , Args ... args )
		{
			char *buffer = _MakeMessageString( "[EXCEPTION]" , functionName , format , args ... );
			_message = std::string( buffer );
			delete[] buffer;
		}
	protected:
		std::string _message;
	};
	template< typename ... Args > void _Throw( const char *functionName , const char *format , Args ... args ){ throw Exception( functionName , format , args ... ); }
	template< typename ... Args >
	void _Warn( const char *functionName , const char *format , Args ... args )
	{
		char *buffer = _MakeMessageString( "[WARNING]" , functionName , format , args ... );
		fprintf( stderr , "%s\n" , buffer );
		delete[] buffer;
	}
	template< typename ... Args >
	void _ErrorOut( const char *functionName , const char *format , Args ... args )
	{
		char *buffer = _MakeMessageString( "[ERROR]" , functionName , format , args ... );
		fprintf( stderr , "%s\n" , buffer );
		delete[] buffer;
	}
#ifndef Warn
#define Warn( ... ) _Warn( __FUNCTION__ , __VA_ARGS__ )
#endif // Warn
#ifndef Throw
#define Throw( ... ) _Throw( __FUNCTION__ , __VA_ARGS__ )
#endif // Throw
#ifndef ErrorOut
#define ErrorOut( ... ) _ErrorOut( __FUNCTION__ , __VA_ARGS__ )
#endif // ErrorOut
#endif // VERBOSE_MESSAGING
#endif

	//////////////////
	// Memory Stuff //
	//////////////////
	size_t getPeakRSS( void );
	size_t getCurrentRSS( void );

	struct MemoryInfo
	{
		static size_t Usage( void ){ return getCurrentRSS(); }
		static int PeakMemoryUsageMB( void ){ return (int)( getPeakRSS()>>20 ); }
	};
#if defined( _WIN32 ) || defined( _WIN64 )
#include <Windows.h>
#include <Psapi.h>
	inline void SetPeakMemoryMB( size_t sz )
	{
		sz <<= 20;
		SIZE_T peakMemory = sz;
		HANDLE h = CreateJobObject( NULL , NULL );
		AssignProcessToJobObject( h , GetCurrentProcess() );

		JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = { 0 };
		jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
		jeli.JobMemoryLimit = peakMemory;
		if( !SetInformationJobObject( h , JobObjectExtendedLimitInformation , &jeli , sizeof( jeli ) ) ) fprintf( stderr , "Failed to set memory limit\n" );
	}
#else // !_WIN32 && !_WIN64
#include <sys/time.h> 
#include <sys/resource.h> 
	inline void SetPeakMemoryMB( size_t sz )
	{
		sz <<= 20;
		struct rlimit rl;
		getrlimit( RLIMIT_AS , &rl );
		rl.rlim_cur = sz;
		setrlimit( RLIMIT_AS , &rl );
	}
#endif // _WIN32 || _WIN64

	/*
	* Author:  David Robert Nadeau
	* Site:    http://NadeauSoftware.com/
	* License: Creative Commons Attribution 3.0 Unported License
	*          http://creativecommons.org/licenses/by/3.0/deed.en_US
	*/

#if defined(_WIN32) || defined( _WIN64 )
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif





	/**
	* Returns the peak (maximum so far) resident set size (physical
	* memory use) measured in bytes, or zero if the value cannot be
	* determined on this OS.
	*/
	inline size_t getPeakRSS( )
	{
#if defined(_WIN32)
		/* Windows -------------------------------------------------- */
		PROCESS_MEMORY_COUNTERS info;
		GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
		return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
		/* AIX and Solaris ------------------------------------------ */
		struct psinfo psinfo;
		int fd = -1;
		if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
			return (size_t)0L;      /* Can't open? */
		if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
		{
			close( fd );
			return (size_t)0L;      /* Can't read? */
		}
		close( fd );
		return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
		/* BSD, Linux, and OSX -------------------------------------- */
		struct rusage rusage;
		getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
		return (size_t)rusage.ru_maxrss;
#else
		return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
		/* Unknown OS ----------------------------------------------- */
		return (size_t)0L;          /* Unsupported. */
#endif
	}





	/**
	* Returns the current resident set size (physical memory use) measured
	* in bytes, or zero if the value cannot be determined on this OS.
	*/
	inline size_t getCurrentRSS( )
	{
#if defined(_WIN32) || defined( _WIN64 )
		/* Windows -------------------------------------------------- */
		PROCESS_MEMORY_COUNTERS info;
		GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
		return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
		/* OSX ------------------------------------------------------ */
		struct mach_task_basic_info info;
		mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
		if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
			(task_info_t)&info, &infoCount ) != KERN_SUCCESS )
			return (size_t)0L;      /* Can't access? */
		return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
		/* Linux ---------------------------------------------------- */
		long rss = 0L;
		FILE* fp = NULL;
		if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
			return (size_t)0L;      /* Can't open? */
		if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
		{
			fclose( fp );
			return (size_t)0L;      /* Can't read? */
		}
		fclose( fp );
		return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
		/* AIX, BSD, Solaris, and Unknown OS ------------------------ */
		return (size_t)0L;          /* Unsupported. */
#endif
	}
}
#endif // MISCELLANY_INCLUDED
