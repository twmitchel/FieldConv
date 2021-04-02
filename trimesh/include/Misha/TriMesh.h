/*
Copyright (c) 2020, Michael Kazhdan and Thomas Mitchel
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

#ifndef TRIMESH_INCLUDED
#define TRIMESH_INCLUDED

#include <vector>
#include <omp.h>
#include <limits.h>
#include <Misha/Geometry.h>
#include <Misha/FEM.h>
#include <Eigen/Eigen>
#include <Misha/Spectrum.h>


#ifndef M_PI
#define M_PI 3.1415926535897932384
#endif

/*
// === Geodesics ===
#include "DGPC/Generator.h"

typedef DGPC::Vector3< double > PointOM;
typedef DGPC::MeshOM< PointOM > repOM;
typedef DGPC::Generator< repOM > DGPCgenerator;
*/

template< class Real=float >
class TriMesh
{
protected:
	const std::vector< Point3D< Real > > &_vertices;
	const std::vector< TriangleIndex > &_triangles;

	std::vector< Real > _triangleAreas;

	std::vector< std::vector< TriangleIndex > > _vertexStars;
	std::vector< std::vector< int > > _starTriangles;
	std::vector< std::vector< int > > _starVertices;

   std::vector< Point3D< double > > _triangleMetrics; // g = [a, b; b, c], (a, b, c)

	// === Geodesics ===
	//repOM _surfaceMesh;
	//double _logEpsilon = 1.0e-12;

public:
	TriMesh( const std::vector< Point3D< Real > > &vertices , const std::vector< TriangleIndex > &triangles );

	// Return vertex and triangle vectors respectively
	const std::vector< Point3D< Real > > &vertices( void ) const { return _vertices; }
	const std::vector< TriangleIndex> &triangles( void ) const { return _triangles; }

	// Returns indexed area
	Real triangleArea( int l ) const;
	Real    totalArea( void )  const;

	// Returns indexed vertex star
	const std::vector< TriangleIndex > &vertexStar( int l ) const;
	const std::vector< int > &vertexStarList ( int l ) const;

	// The functor takes in a TriangleIndex and outputs a Point3D< double > giving the square lengths of the edges opposite the indexed vertices
	template< typename TriangleSquareEdgeLengthFunctor >
	void initMetricsFromSquareEdgeLengths( TriangleSquareEdgeLengthFunctor F );
	SquareMatrix< double , 2 > triangleMetric( int l ) const;

   // Gets value of gradient of implicit function in tangent space at triangles w.r.t. to defined metric
    Point2D< double > getMetricGradient( int l, const double phi1, const double phi2, const double phi3) const;
	Point2D< double > getMetricGradient( int l, const std::vector< double >& Implicit) const;

	// Computes gradient of implicit function at all triangles
   void metricGradient( const std::vector< double >& Implciit , std::vector<Point2D< double > > &triGrads ) const;

   // Tangent space operations
   double metricDot( int l, const Point2D< double >& w1, const Point2D< double >& w2 ) const;

   double metricSquareNorm( int l , const Point2D< double >& w ) const;

   Point2D< double > metricRotate90 ( int l, const Point2D< double >& w) const;

	// Smooth vertex signal
	void smoothVertexSignal( std::vector< double >& Implicit , Real diffTime );


	// =================
	// === Geodesics ===
	// =================
   
   /*
	void setGeodesicEpsilon( double epsilon );

	void initGeodesicCalc( void ); // Initalizes mesh for geodesic calculations

	std::vector< double > computeGeodesicsAbout( int nodeIndex, float rho ) const;

	std::vector< double > computeGeodesicsAbout( std::pair< int , Point3D< double > > P , float rho ) const;

	std::pair<std::vector< double >, std::vector< double >> computeLogarithmAbout( int nodeIndex , float rho ) const;

	std::pair<std::vector< double >, std::vector< double >> computeLogarithmAbout( std::pair<int, Point3D< double >> P, float rho ) const;
*/

	// ===============================================
	// === Laplace-Beltrami spectral decomposition ===
	// ===============================================
	// Spectral distance
	double spectralDist( const Spectrum< double > &spectrum , int nodeX , int nodeY ) const;
	double spectralDist( const Spectrum< double > &spectrum , std::pair< int , Point3D< double > > P , int node ) const;

	std::vector< double > computeSpectralDistancesAbout( const Spectrum< double > &spectrum , int nodeIndex, double rho = std::numeric_limits< double >::max () ) const;
	std::vector< double > computeSpectralDistancesAbout( const Spectrum< double > &spectrum , std::pair<int, Point3D< double >> & P, double rho = std::numeric_limits< double >::max () ) const;
};

#include "TriMesh.inl"
#endif // TRIMESH_INCLUDED
