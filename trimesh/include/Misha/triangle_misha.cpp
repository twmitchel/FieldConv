/*****************************************************************************/
/*                                                                           */
/*  (tricall.c)                                                              */
/*                                                                           */
/*  Example program that demonstrates how to call Triangle.                  */
/*                                                                           */
/*  Accompanies Triangle Version 1.6                                         */
/*  July 19, 1996                                                            */
/*                                                                           */
/*  This file is placed in the public domain (but the file that it calls     */
/*  is still copyrighted!) by                                                */
/*  Jonathan Richard Shewchuk                                                */
/*  2360 Woolsey #H                                                          */
/*  Berkeley, California  94705-1927                                         */
/*  jrs@cs.berkeley.edu                                                      */
/*                                                                           */
/*****************************************************************************/

/* If SINGLE is defined when triangle.o is compiled, it should also be       */
/*   defined here.  If not, it should not be defined here.                   */

/* #define SINGLE */

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#define ANSI_DECLARATORS
#include "triangle.h"


void triangle_misha( double* pcoordinates , int pcount , std::vector< std::vector< int > >& triangles , bool quiet )
{
	struct triangulateio in, out;
	in.numberofpoints = pcount;
	in.numberofpointattributes = 0;
	in.pointlist = pcoordinates;
	in.pointattributelist = NULL;
	in.pointmarkerlist = NULL;
	in.numberofsegments = 0;
	in.numberofholes = 0;
	in.numberofregions = 0;

	out.pointlist = NULL;
	out.pointattributelist = NULL;
	out.pointmarkerlist = NULL;
	out.trianglelist = NULL;
	out.triangleattributelist = NULL;
	out.neighborlist = NULL;
	out.segmentlist = NULL;
	out.segmentmarkerlist = NULL;
	out.edgelist = NULL;
	out.edgemarkerlist = NULL;

	if( quiet ) triangulate( "zQ" , &in , &out , NULL );
	else        triangulate( "z"  , &in , &out , NULL );
	triangles.resize( out.numberoftriangles );
	for( int i=0 ; i<out.numberoftriangles ; i++ )
	{
		triangles[i].resize(out.numberofcorners);
		for( int j=0 ; j<out.numberofcorners ; j++ ) triangles[i][j] = out.trianglelist[out.numberofcorners*i+j];
	}

  free( out.trianglelist );
}
void triangle_misha( float* pcoordinates , int pcount , std::vector< std::vector< int > >& triangles , bool quiet )
{
	double *_pcoordinates = new double [pcount*2];
	for( int i=0 ; i<pcount*2 ; i++ ) _pcoordinates[i] = pcoordinates[i];
	triangle_misha( _pcoordinates , pcount , triangles , quiet );
	delete[] _pcoordinates;
}
