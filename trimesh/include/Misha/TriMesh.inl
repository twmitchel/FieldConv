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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <Misha/Solver.h>


template< class Real >
TriMesh< Real >::TriMesh( const std::vector< Point3D< Real > > &vertices , const std::vector< TriangleIndex > &triangles ) : _vertices(vertices) , _triangles(triangles)
{  
    // Compute stars
    _vertexStars.resize( _vertices.size() );
    _starTriangles.resize ( _vertices.size() );

    for( int l=0 ; l<_triangles.size() ; l++ )
    {
        for( int i=0 ; i<3 ; i++ )
        {
            _vertexStars[ _triangles[l][i] ].push_back( _triangles[l] );
            _starTriangles[ _triangles[l][i] ].push_back (l);
        }
    }

    _starVertices.resize( _vertices.size() );

    for( int l=0 ; l<_vertices.size() ; l++ )
        for( int i=0 ; i<_starTriangles[l].size() ; i++ )
            for( int j=0 ; j<3 ; j++ )
                if( _triangles[ _starTriangles[l][i] ][j]!=l )
                    if ( std::find ( _starVertices[l].begin() , _starVertices[l].end() , _triangles[ _starTriangles[l][i] ][j] )==_starVertices[l].end() )
                        _starVertices[l].push_back( _triangles[ _starTriangles[l][i] ][j] );
}

template< class Real >
Real TriMesh< Real >::triangleArea( int l ) const
{
    return _triangleAreas[l];
}

template< class Real >
Real TriMesh< Real >::totalArea( void ) const
{
    Real A = 0.0;

    for( int l=0 ; l<_triangleAreas.size() ; l++ ) A += _triangleAreas[l];

    return A;
}

template< class Real >
const std::vector< TriangleIndex > &TriMesh< Real >::vertexStar ( int l ) const
{
    return _vertexStars[l];
}

template< class Real >
const std::vector< int > &TriMesh< Real >::vertexStarList ( int l ) const
{
    return _starTriangles[l];
}

template< typename Real >
template< typename TriangleSquareEdgeLengthFunctor >
void TriMesh< Real >::initMetricsFromSquareEdgeLengths( TriangleSquareEdgeLengthFunctor F )
{
    _triangleMetrics.resize( _triangles.size() );
    _triangleAreas.resize( _triangles.size() );

#pragma omp parallel for
    for( int i=0 ; i<_triangles.size() ; i++ )
    {
        Point3D< Real > squareEdgeLengths = F( _triangles[i] );

        _triangleMetrics[i][0] = squareEdgeLengths[2];
        _triangleMetrics[i][1] = ( squareEdgeLengths[2] + squareEdgeLengths[1] - squareEdgeLengths[0] ) / 2;
        _triangleMetrics[i][2] = squareEdgeLengths[1];

        _triangleAreas[i] = (Real)( std::sqrt( std::fabs( _triangleMetrics[i][0] * _triangleMetrics[i][2] - _triangleMetrics[i][1] * _triangleMetrics[i][1] ) ) / 2. );
    }
}

// Gradients

template< class Real >
Point2D< double > TriMesh< Real >::getMetricGradient( int l , const double phi1 , const double phi2 , const double phi3 ) const
{
    
    Point2D< double > grad(0, 0);

    double a = _triangleMetrics[l][0];
    double b = _triangleMetrics[l][1];
    double c = _triangleMetrics[l][2];

    
    double mDet = a * c - b * b;

    if (mDet > 0)
    {
       double v1 = phi2 - phi1;
       double v2 = phi3 - phi1;

       grad[0] = ( c * v1 - b * v2 ) / mDet;
       grad[1] = ( a * v2 - b * v1 ) / mDet;

    }
 
    return grad;
    
}

template< class Real >
Point2D< double > TriMesh< Real >::getMetricGradient( int l , const std::vector< double > &Implicit ) const
{
    
    Point2D< double > grad(0, 0);

    double a = _triangleMetrics[l][0];
    double b = _triangleMetrics[l][1];
    double c = _triangleMetrics[l][2];

    
    double mDet = a * c - b * b;

    if (mDet > 0)
    {
        double v1 = Implicit[ _triangles[l][1] ] - Implicit[ _triangles[l][0] ];
        double v2 = Implicit[ _triangles[l][2] ] - Implicit[ _triangles[l][0] ];

       grad[0] = ( c * v1 - b * v2 ) / mDet;
       grad[1] = ( a * v2 - b * v1 ) / mDet;

    }
 
    return grad;
}

template< class Real >
void TriMesh< Real >::metricGradient ( const std::vector< double >& Implicit , std::vector< Point2D< double > > & triGrads ) const
{
    triGrads.resize( _triangles.size() );

#pragma omp parallel for
    for( int l=0 ; l<_triangles.size() ; l++ ) triGrads[l] = getMetricGradient( l , Implicit );
}


// Metric dot products
template< typename Real >
SquareMatrix< double , 2 > TriMesh< Real >::triangleMetric( int l ) const
{
    SquareMatrix< double , 2 > m;
    m(0,0) = _triangleMetrics[l][0];
    m(1,1) = _triangleMetrics[l][2];
    m(0,1) = m(1,0) = _triangleMetrics[l][1];
    return m;
}
template< class Real >
double TriMesh< Real >::metricDot( int l , const Point2D< double >& w1, const Point2D< double >& w2) const
{
    double a = _triangleMetrics[l][0];
    double b = _triangleMetrics[l][1];
    double c = _triangleMetrics[l][2];

    return (a * w1[0] + b * w1[1]) * w2[0] + (b * w1[0] + c * w1[1]) * w2[1];  
}

template< class Real >
double TriMesh< Real >::metricSquareNorm( int l, const Point2D< double >& w) const
{
   return metricDot (l, w, w);
}

template< class Real >
Point2D< double > TriMesh< Real >::metricRotate90( int l , const Point2D< double > &w ) const
{

    double a = _triangleMetrics[l][0];
    double b = _triangleMetrics[l][1];
    double c = _triangleMetrics[l][2];

    double detRoot = std::sqrt(a * c - b * b);

    if( detRoot>0 )
    {

       double alphaR = -(b * w[0] + c * w[1]) / detRoot;

       double betaR = (a * w[0] + b * w[1]) / detRoot;

       return Point2D< double >( alphaR , betaR );
    }
    else
    {
       return Point2D< double >(0, 0);
    }
}

template< class Real >
void TriMesh< Real >::smoothVertexSignal( std::vector< double > &x , Real diffTime )
{
    // Normalize mesh to have unit area
    FEM::RiemannianMesh< double > mesh( GetPointer( _triangles ) , _triangles.size() );
    mesh.template setMetricFromEmbedding< 3 >( [&]( unsigned int idx ){ return Point3D< double >( _vertices[idx] ); } );
    mesh.makeUnitArea();

    // Compute and solve the system

    SparseMatrix< double , int > M = mesh.template massMatrix< FEM::BASIS_0_WHITNEY >() * diffTime;
    SparseMatrix< double , int > S = mesh.template stiffnessMatrix< FEM::BASIS_0_WHITNEY >();

#pragma omp parallel for
    for( int i=0 ; i<S.rows ; i++ ) for( int j=0 ; j<S.rowSizes[i] ; j++ ) S[i][j].Value += M[i][j].Value;

    EigenSolverCholeskyLLt< double > llt( S );

    std::vector< double > b( _vertices.size() );

    M.Multiply( ( ConstPointer( double ) )GetPointer( x ) , GetPointer( b ) );
    llt.solve( ( ConstPointer( double ) )GetPointer( b ) , GetPointer( x ) );
}

/*
// ==============================
// ======== Geodesics ===========
// ==============================

template< class Real >
void TriMesh< Real >::setGeodesicEpsilon( double epsilon )
{
    _logEpsilon = epsilon;
}

template< class Real >
void TriMesh< Real >::initGeodesicCalc( void )
{
    _surfaceMesh.fromTriangles( _vertices , _triangles );
}

template< class Real >
std::vector< double > TriMesh< Real >::computeGeodesicsAbout( int nodeIndex , float rho ) const
{
    DGPCgenerator dgpc( _surfaceMesh );
    dgpc.setEps( _logEpsilon );
    dgpc.setStopDist( rho );
    dgpc.setNodeSource( nodeIndex );
    dgpc.run();

    return dgpc.getDistances ();
}

template< class Real >
std::vector< double > TriMesh< Real >::computeGeodesicsAbout( std::pair< int , Point3D< double > > P , float rho ) const
{
    DGPCgenerator dgpc( _surfaceMesh );
    dgpc.setEps( _logEpsilon );
    dgpc.setStopDist( rho );
    Point3D< Real > pt = _vertices[ _triangles[P.first][0] ] * (Real)P.second[0] + _vertices[ _triangles[P.first][1] ] * (Real)P.second[1] + _vertices[ _triangles[P.first][2] ] * (Real)P.second[2];
    dgpc.setSource( PointOM( pt[0] , pt[1] , pt[2] ) , P.first );
    dgpc.run();

    return dgpc.getDistances ();
}


template< class Real >
std::pair< std::vector< double > , std::vector< double > > TriMesh< Real >::computeLogarithmAbout( int nodeIndex , float rho ) const
{
    DGPCgenerator dgpc( _surfaceMesh );
    dgpc.setEps( _logEpsilon );
    dgpc.setStopDist( rho );
    dgpc.setNodeSource( nodeIndex );
    dgpc.run();

    return std::pair< std::vector< double >, std::vector< double > >( dgpc.getDistances () , dgpc.getAngles () );

}

template< class Real >
std::pair< std::vector< double > , std::vector< double > > TriMesh< Real >::computeLogarithmAbout( std::pair< int , Point3D< double > > P , float rho ) const
{
    DGPCgenerator dgpc( _surfaceMesh );
    dgpc.setEps( _logEpsilon );
    dgpc.setStopDist( rho );
    Point3D<Real> pt = vertices[triangles[P.first][0]] * P.second[0] +  vertices[triangles[P.first][1]] * P.second[1] + vertices[triangles[P.first][2]] * P.second[2];
    dgpc.setSource( PointOM( pt[0] , pt[1] , pt[2] ) , P.first );
    dgpc.run();

    return std::pair< std::vector< double >, std::vector< double > > ( dgpc.getDistances (), dgpc.getAngles () );
}
*/

// ==========================
// === Spectral Distances ===
// ==========================
template< class Real >
double TriMesh< Real >::spectralDist( const Spectrum< double > &spectrum , int nodeX , int nodeY ) const
{
    return spectrum.spectralDistance( nodeX , nodeY , 1 , std::numeric_limits< int >::max() );
}

template< class Real >
double TriMesh< Real >::spectralDist( const Spectrum< double > &spectrum , std::pair< int , Point3D< double > > P , int node ) const
{
    return spectrum.spectralDistance( node , _triangles[ P.first ] , P.second , 1 , std::numeric_limits< int >::max() );
}

template< class Real >
std::vector< double > TriMesh< Real >::computeSpectralDistancesAbout( const Spectrum< double > &spectrum , int nodeIndex, double rho ) const
{
    std::vector< double > sDistances; 
    sDistances.resize( _vertices.size () , std::numeric_limits< double >::max() );

    std::vector< bool > processed;
    processed.resize( _vertices.size() , false );

    std::vector< int > Q { nodeIndex };
    processed[ nodeIndex ] = true;
    sDistances[ nodeIndex ] = 0.0;

    while( Q.size() )
    {
        int q = Q.back();
        Q.pop_back();

        for( int l=0 ; l<_starVertices[q].size() ; l++ )
        {
            int r = _starVertices[q][l];
            if( !processed[r] )
            {
                sDistances[r] = spectralDist( spectrum , nodeIndex , r );
                processed[r] = true;
                if( sDistances[r]<=rho ) Q.push_back ( r );
            }
        }
    }

    return sDistances;
}

template< class Real >
std::vector< double > TriMesh< Real >::computeSpectralDistancesAbout( const Spectrum< double > &spectrum , std::pair<int, Point3D< double >> & P, double rho ) const
{
    std::vector< double > sDistances; 
    sDistances.resize( _vertices.size() , std::numeric_limits< double >::max() );

    std::vector<bool> processed;
    processed.resize( _vertices.size () , false );

    std::vector< int > Q { (int)_triangles[P.first][0] , (int)_triangles[P.first][1] , (int)_triangles[P.first][2] };
    processed[ _triangles[P.first][0] ] = processed[ _triangles[P.first][1] ] = processed[ _triangles[P.first][2] ] = true;

    sDistances[ _triangles[P.first][0] ] = spectralDist( spectrum , P , _triangles[P.first][0] );
    sDistances[ _triangles[P.first][1] ] = spectralDist( spectrum , P , _triangles[P.first][1] );
    sDistances[ _triangles[P.first][2] ] = spectralDist( spectrum , P , _triangles[P.first][2] );

    while( Q.size () > 0 )
    {
        int q = Q.back ();
        Q.pop_back ();

        for (int l = 0; l < _starVertices[q].size (); l++)
        {
            int r = _starVertices[q][l];
            if ( !processed[r] )
            {
                sDistances[r] = spectralDist( spectrum , P , r );
                processed[r] = true;
                if( sDistances[r]<=rho ) Q.push_back ( r );
            }
        }
    }

    return sDistances;
}
