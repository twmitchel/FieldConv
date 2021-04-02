/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
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

#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED

//#define NEW_SIMPLEX

#include <cmath>
#include <cassert>
#include <complex>
#include <vector>
#include <unordered_map>
#include <initializer_list>
#include <string.h>
#include <algorithm>
#include "Algebra.h"

#define PAN_FIX 1
#define FAST_POINT 1

// An empty type
template< typename Real >
struct EmptyVectorType
{
	EmptyVectorType& operator += ( const EmptyVectorType& p ){ return *this; }
	EmptyVectorType& operator -= ( const EmptyVectorType& p ){ return *this; }
	EmptyVectorType& operator *= ( Real s )                  { return *this; }
	EmptyVectorType& operator /= ( Real s )                  { return *this; }
	EmptyVectorType  operator +  ( const EmptyVectorType& p ) const { EmptyVectorType _p = *this ; _p += p ; return _p; }
	EmptyVectorType  operator -  ( const EmptyVectorType& p ) const { EmptyVectorType _p = *this ; _p -= p ; return _p; }
	EmptyVectorType  operator *  ( Real s )                   const { EmptyVectorType _p = *this ; _p *= s ; return _p; }
	EmptyVectorType  operator /  ( Real s )                   const { EmptyVectorType _p = *this ; _p /= s ; return _p; }

	friend std::ostream &operator << ( std::ostream &os , const EmptyVectorType &v ){ return os; }
};
template< typename Real > EmptyVectorType< Real > operator * ( Real s , EmptyVectorType< Real > v ){ return v*s; }

template< typename _Real , typename ... VectorTypes >
struct VectorTypeUnion
{
protected:
	typedef std::tuple< VectorTypes ... > _VectorTuple;
public:
	typedef _Real Real;
	template< unsigned int I > using VectorType = typename std::tuple_element< I , _VectorTuple >::type;
	template< unsigned int I >       VectorType< I >& get( void )       { return std::get< I >( _vectorTypeTuple ); }
	template< unsigned int I > const VectorType< I >& get( void ) const { return std::get< I >( _vectorTypeTuple ); }

	VectorTypeUnion& operator += ( const VectorTypeUnion& p ){ _add<0>( p ) ; return *this; }
	VectorTypeUnion& operator -= ( const VectorTypeUnion& p ){ _sub<0>( p ) ; return *this; }
	VectorTypeUnion& operator *= ( Real s )                  { _mul<0>( s ) ; return *this; }
	VectorTypeUnion& operator /= ( Real s )                  { _div<0>( s ) ; return *this; }
	VectorTypeUnion  operator +  ( const VectorTypeUnion& p ) const { VectorTypeUnion _p = *this ; _p += p ; return _p; }
	VectorTypeUnion  operator -  ( const VectorTypeUnion& p ) const { VectorTypeUnion _p = *this ; _p -= p ; return _p; }
	VectorTypeUnion  operator *  ( Real s )                   const { VectorTypeUnion _p = *this ; _p *= s ; return _p; }
	VectorTypeUnion  operator /  ( Real s )                   const { VectorTypeUnion _p = *this ; _p /= s ; return _p; }

	VectorTypeUnion( void ){}
	VectorTypeUnion( const VectorTypes & ... vectors ){ _set< 0 >( vectors ... ); }

	friend std::ostream &operator << ( std::ostream &os , const VectorTypeUnion &v )
	{
		os << "{ ";
		v._streamOut< 0 >( os );
		os << " }";
		return os;
	}
protected:
	std::tuple< VectorTypes ... > _vectorTypeTuple;
	template< unsigned int I , typename _Vector , typename ... _Vectors > void _set( const _Vector &vector , const _Vectors & ... vectors ){ get< I >() = vector ; _set< I+1 >( vectors ... ); }
	template< unsigned int I , typename _Vector                         > void _set( const _Vector &vector                                ){ get< I >() = vector ;                             }
	template< unsigned int I > typename std::enable_if< I!=sizeof...(VectorTypes) >::type _add( const VectorTypeUnion& p ){ get<I>() += p.get<I>() ; _add< I+1 >( p ); }
	template< unsigned int I > typename std::enable_if< I==sizeof...(VectorTypes) >::type _add( const VectorTypeUnion& p ){ }
	template< unsigned int I > typename std::enable_if< I!=sizeof...(VectorTypes) >::type _sub( const VectorTypeUnion& p ){ get<I>() -= p.get<I>() ; _sub< I+1 >( p ); }
	template< unsigned int I > typename std::enable_if< I==sizeof...(VectorTypes) >::type _sub( const VectorTypeUnion& p ){ }
	template< unsigned int I > typename std::enable_if< I!=sizeof...(VectorTypes) >::type _mul( Real s ){ get<I>() *= s ; _mul< I+1 >( s ); }
	template< unsigned int I > typename std::enable_if< I==sizeof...(VectorTypes) >::type _mul( Real s ){ }
	template< unsigned int I > typename std::enable_if< I!=sizeof...(VectorTypes) >::type _div( Real s ){ get<I>() /= s ; _div< I+1 >( s ); }
	template< unsigned int I > typename std::enable_if< I==sizeof...(VectorTypes) >::type _div( Real s ){ }
	template< unsigned int I > typename std::enable_if< I!=sizeof...(VectorTypes) >::type _streamOut( std::ostream &os ) const { os << get<I>() ; if( I!=sizeof...(VectorTypes)-1 ) os << " , "; _streamOut< I+1 >( os ); }
	template< unsigned int I > typename std::enable_if< I==sizeof...(VectorTypes) >::type _streamOut( std::ostream &os ) const { }
};
template< typename Real , typename ... Vectors >
VectorTypeUnion< Real , Vectors ... > operator * ( Real s , VectorTypeUnion< Real , Vectors ... > vu ){ return vu * s; }

template< class Real > Real Random( void );

template< class Real , int Dim > class SquareMatrix;

template< typename Real , unsigned int Dim > class XForm;

template< typename Real , int ... Dims > struct Point;

template< class Real , int Dim >
class Point< Real , Dim > : public InnerProductSpace< Real , Point< Real , Dim > >
{
	template< class ... Points >
	static void _AddColumnVector( SquareMatrix< Real , Dim >& x , int c , Point point , Points ... points )
	{
		for( int r=0 ; r<Dim ; r++ ) x( c , r ) = point[r];
		_AddColumnVector( x , c+1 , points ... );
	}
	static void _AddColumnVector( SquareMatrix< Real , Dim >& x , int c ){ ; }
public:
	/////////////////////////////////
	// Inner product space methods //
	void Add            ( const Point& p );
	void Scale          ( Real s );
	Real InnerProduct   ( const Point< Real , Dim >& p ) const;
	/////////////////////////////////

	Real coords[Dim];
	Point( void ) { memset( coords , 0 , sizeof(Real)*Dim ); }
	template< typename Real2 >
	Point( Point< Real2 , Dim > p ){ for( int i=0 ; i<Dim ; i++ ) coords[i] = (Real)p.coords[i]; }
	Point( std::initializer_list< Real > l ){ memset( coords , 0 , sizeof(Real)*Dim ) ; for( int i=0 ; i<Dim && l.size() ; i++ ) coords[i] = l.begin()[i]; }
	template< class Real2 >
	operator Point< Real2 , Dim > ( void ) const
	{
		Point< Real2, Dim > p;
		for( int d=0 ; d<Dim ; d++ ) p.coords[d] = Real2( coords[d] ); 
		return p;
	}
	Real& operator [] (int idx) { return coords[idx]; }
	const Real& operator [] (int idx) const { return coords[idx]; }

	template< class ... Points > static Point CrossProduct( Points ... points )
	{
		static_assert( sizeof ... ( points )==Dim-1 , "Number of points in cross-product must be one less than the dimension" );
		SquareMatrix< Real , Dim > x;
		_AddColumnVector( x , 0 , points ... );
		Point p;
		for( int d=0 ; d<Dim ; d++ ) p[d] = x.subDeterminant( Dim-1 , d );
		return p;
	}
	static Point CrossProduct( const Point *points )
	{
		XForm< Real , Dim > x;
		for( unsigned int d=0 ; d<Dim-1 ; d++ ) for( unsigned int c=0 ; c<Dim ; c++ ) x(d,c) = points[d][c];
		Point p;
		for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( d&1 ) ? -x.subDeterminant( Dim-1 , d ) : x.subDeterminant( Dim-1 , d );
		return p;
	}
	static Point CrossProduct( Point *points ){ return CrossProduct( ( const Point * )points ); }


	static Point< Real , Dim > Min( Point< Real , Dim > p , Point< Real , Dim > q ){ Point< Real , Dim > m ; for( int d=0 ; d<Dim ; d++ ) m[d] = std::min< Real >( p[d] , q[d] ) ; return m; }
	static Point< Real , Dim > Max( Point< Real , Dim > p , Point< Real , Dim > q ){ Point< Real , Dim > m ; for( int d=0 ; d<Dim ; d++ ) m[d] = std::max< Real >( p[d] , q[d] ) ; return m; }

	friend std::ostream &operator << ( std::ostream &os , const Point &p )
	{
		os << "( ";
		for( int d=0 ; d<Dim ; d++ )
		{
			if( d ) os << " , ";
			os << p[d];
		}
		return os << " )";
	}
};
template< class Real , int Dim > Point< Real , Dim > operator * ( Real s , Point< Real , Dim > p ){ return p*s; }

template< class Real >
class Point< Real , -1 > : public InnerProductSpace< Real , Point< Real , -1 > >
{
public:
	/////////////////////////////////
	// Inner product space methods //
	void Add( const Point& p )
	{
		if( !_dim ){ _resize( p._dim ) ; for( unsigned int i=0 ; i<_dim ; i++ ) _coords[i] = p._coords[i]; }
		else if( _dim==p._dim ) for( unsigned int i=0 ; i<_dim ; i++ ) _coords[i] += p._coords[i];
		else ERROR_OUT( "Dimensions don't match: " , _dim , " != " , p._dim );
	}
	void Scale( Real s ){ for( unsigned int i=0 ; i<_dim ; i++ ) (*this)[i] *= s; }
	Real InnerProduct( const Point &p ) const
	{
		Real dot;
		if( _dim!=p._dim ) ERROR_OUT( "Dimensions differ: " , _dim , " != " , p._dim );
		for( size_t d=0 ; d<_dim ; d++ ) dot += _coords[d] * p._coords[d];
		return dot;
	}
	/////////////////////////////////

	Point( void ) : _coords(NULL) , _dim(0){}
	Point( size_t dim ) : _coords(NULL) , _dim(0) { if( dim ){ _resize( (unsigned int)dim ) ; memset( _coords , 0 , sizeof(Real)*_dim ); } }
	Point( const Point &p ) : _coords(NULL) , _dim(0) { if( p._dim ){ _resize( p._dim ) ; memcpy( _coords , p._coords , sizeof(Real)*_dim ); } }
	~Point( void ){ delete[] _coords ; _coords = NULL; }

	Point &operator = ( const Point &p )
	{
		if( !_dim ){ _resize( p._dim ) ; memcpy( _coords , p._coords , sizeof(Real)*_dim ); }
		else if( _dim==p._dim ) memcpy( _coords , p._coords , sizeof(Real)*_dim );
		else ERROR_OUT( "Dimensions don't match: " , _dim , " != " , p._dim );
		return *this;
	}

	unsigned int dim( void ) const { return _dim; }
	Real &operator[]( size_t idx ){ return _coords[idx]; }
	const Real &operator[]( size_t idx ) const { return _coords[idx]; }

	friend std::ostream &operator << ( std::ostream &os , const Point &p )
	{
		os << "( ";
		for( size_t d=0 ; d<dim() ; d++ )
		{
			if( d ) os << " , ";
			os << p[d];
		}
		return os << " )";
	}
protected:
	Real *_coords;
	unsigned int _dim;
	void _resize( unsigned int dim ){ if( dim ){ _coords = new Real[dim] ; _dim = dim; } }
};
template< class Real > Point< Real , -1 > operator * ( Real s , Point< Real , -1 > p ){ return p*s; }

////////////////////////////////////////////////////////////////////////////////
/*! Convenience function to create Point<Real, 2>s
//  @param[in]  x   the first component
//  @param[in]  x   the second component
//  @return     Point<Real, 2> initialized with the two components
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
Point<Real, 2> MakePoint2(Real x, Real y)
{
	Point<Real, 2> result;
	result[0] = x;
	result[1] = y;
	return result;
}

////////////////////////////////////////////////////////////////////////////////
/*! Complex-real vector dot product
//  @param[in]  a   complex vector lhs
//  @param[in]  b   real vector rhs
//  @return     (complex result) a dot b
*///////////////////////////////////////////////////////////////////////////////
template <class Real, int Dim>
inline std::complex<Real> operator*(const Point<std::complex<Real>, Dim> &a
	, const Point<Real              , Dim> &b)
{
	std::complex<Real> dot = 0;
	for (int i = 0; i < Dim; ++i)   {
		dot += a[i] * b[i];
	}
	return dot;
}

template<class Real,int Cols,int Rows>
class Matrix : public InnerProductSpace<Real,Matrix<Real,Cols,Rows> >
{
public:
	//////////////////////////
	// Vector space methods //
	void Add            ( const Matrix& m );
	void Scale          ( Real s );
	Real InnerProduct   ( const Matrix& m ) const;
	//////////////////////////

	Real coords[Cols][Rows];
	Matrix ( void ) { memset( coords , 0 , sizeof( Real ) * Cols * Rows ); }
	template<class Real2>
	operator Matrix< Real2 , Cols , Rows > ( void ) const
	{
		Matrix< Real2, Cols , Rows > m;
		for( int c=0 ; c<Cols ; c++ ) for ( int r=0 ; r<Rows ; r++ ) m.coords[c][r] = Real2( coords[c][r] ); 
		return m;
	}
	template<int C,int R>
	Matrix(const Matrix<Real,C,R>& m)
	{
		for(int i=0;i<Cols && i<C;i++)
			for(int j=0;j<Rows && j<R;j++)
				coords[i][j]=m.coords[i][j];
	}
	Real& operator () (int c,int r) { return coords[c][r]; }
	const Real& operator () (int c,int r) const { return coords[c][r]; }

	template<int Cols1>
	Matrix<Real,Cols1,Rows> operator * ( const Matrix< Real , Cols1 , Cols >& m ) const;

	Matrix<Real,Rows,Cols> transpose( void ) const;

	template<class Real2>
	Point<Real2,Rows> operator * ( const Point< Real2 , Cols >& v ) const;
	template<class Real2>
	Point<Real2,Rows> operator () ( const Point< Real2 , Cols >& v ) const;

	friend std::ostream &operator << ( std::ostream &os , const Matrix &m )
	{
		os << "{ ";
		for( int r=0 ; r<Rows ; r++ )
		{
			if( r ) os << " , ";
			os << "{ ";
			for( int c=0 ; c<Cols ; c++ )
			{
				if( c ) os << " , ";
				os << m.coords[c][r];
			}
			os << " }";
		}
		return os << " }";
	}
};

template< class Real , int Dim >
class SquareMatrix : public Algebra< Real , SquareMatrix< Real , Dim > > , public Matrix< Real , Dim , Dim >
{
public:
	using Matrix< Real , Dim , Dim >::coords;
	////////////////////////////////
	// Additional algebra methods //
	void Multiply ( const SquareMatrix& m );
	void SetIdentity( void );
	////////////////////////////////

	SquareMatrix( const Matrix< Real , Dim , Dim >& m ){ memcpy( coords , m.coords , sizeof(Real)*Dim*Dim ); }
	SquareMatrix( void )                               { memset( coords , 0 ,        sizeof(Real)*Dim*Dim ); }
	static SquareMatrix Identity( void ){ SquareMatrix m ; for( int i=0 ; i<Dim ; i++ ) m(i,i) = (Real)1 ; return m; }
	Real subDeterminant( int c , int r ) const;
	Real determinant( void ) const;
	Real trace( void ) const;
	SquareMatrix inverse( bool& success ) const;
	SquareMatrix inverse( void ) const;
	SquareMatrix transpose( void ) const { return Matrix< Real , Dim , Dim >::transpose(); }

	// [NOTE] Disabling because we would like the product to be returned as a square matrix
//	using Matrix< Real , Dim , Dim >::operator *;
	using Matrix< Real , Dim , Dim >::operator ();
	template< class Real2 > Point< Real2 , Dim-1 > operator () ( const Point< Real2 , Dim-1 >& v ) const;
};
template< class Real , int Dim1 , int Dim2 > Matrix< Real , Dim2 , Dim1 > operator * ( const SquareMatrix< Real , Dim1 >& m1 , const Matrix< Real , Dim2 , Dim1 >& m2 ){ return ( Matrix< Real , Dim1 , Dim1 > )m1 * m2; }
template< class Real , int Dim1 , int Dim2 > Matrix< Real , Dim1 , Dim2 > operator * ( const Matrix< Real , Dim1 , Dim2 >& m1 , const SquareMatrix< Real , Dim1 >& m2 ){ return m1 * ( Matrix< Real , Dim1 , Dim1 > )m2; }

//const Matrix< Real , 1Dim , Dim >& m ) const { return Matrix< Real , Dim , Dim >::operator * ( m ); }
//template< int _Dim > friend Matrix< Real , Dim , _Dim > operator * ( const Matrix< Real , Dim , _Dim >& m1 , const SquareMatrix< Real , Dim >& m2 ){ return m1*m2; }


template< class V , int Dim , class _R = typename V::R >
class Gradient : public VectorSpace< _R , Gradient< V , Dim , _R > >
{
public:
	//////////////////////////
	// Vector space methods //
	void Add            ( const Gradient& g ) { for( int c=0  ; c<Dim ; c++ ) gradients[c] += g.gradients[c]; }
	void Scale          ( _R s ) { for( int c=0 ; c<Dim ; c++ ) gradients[c] *= s; }
	//                      //
	//////////////////////////

	V gradients[Dim];
	Gradient( void ) { for( int d=0 ; d<Dim ;  d++ ) gradients[d] *= 0; }
	V& operator[] ( int idx ) { return gradients[idx]; }
	const V& operator[] ( int idx ) const { return gradients[idx]; }

	template< class V2 , class _R2>
	operator Gradient< V2, Dim , _R2 > ( void ) const
	{
		Gradient< V2 , Dim , _R2 > g;
		for( int d=0 ; d<Dim ; d++ ) g.gradients[d] = V2( gradients[d] ); 
		return g;
	}

	template< class Real >
	Gradient Project( const Point< Real , Dim >& dir ) const
	{
		V dot;
		Gradient g;
		g *= 0;
		dot *= 0;
		Real len = Real( sqrt( Point< Real , Dim >::SquareNorm( dir ) ) );
		if( !len ) return g;
		Point< Real , Dim > _dir = dir / len;
		for( int d=0 ; d<Dim ; d++ ) dot += gradients[d] * _dir[d];
		for( int d=0 ; d<Dim ; d++ ) g.gradients[d] = dot * _dir[d];
		return g;
	}
};

template< class V , int Dim , class _R = typename V::R >
class ConstantFunction : public VectorSpace< _R , ConstantFunction< V , Dim , _R > >
{
public:
	V value;
	Gradient< V , Dim , _R > gradients;
	ConstantFunction( void ) { value *= 0 , gradients *= 0;}

	template< class Real > V operator( ) ( const Point< Real , Dim >& p ) const { return value; }
	template< class Real > Gradient< V , Dim , _R > gradient( const Point< Real , Dim >& p ) const { return gradients; }

	//////////////////////////
	// Vector space methods //
	void Add            ( const ConstantFunction& cf ) { value += cf.value; }
	void Scale          ( _R s ) { value *= s , this->offset *= s; }
	//////////////////////////
};

template< class V , int Dim , class _R = typename V::R >
class LinearFunction : public VectorSpace< _R , LinearFunction< V , Dim , _R > >
{
public:
	Gradient< V , Dim , _R > gradients;
	V offset;
	LinearFunction( void ) { offset *= 0 ; }
	template< class Real >
	V operator( ) ( const Point< Real , Dim >& p ) const
	{
		V v;
		v *= 0;
		for( int d=0 ; d<Dim ; d++ ) v += gradients[d] * p[d];
		v -= offset;
		return v;
	}
	template< class Real >
	LinearFunction fitToHyperplane( const Point< Real , Dim >& p , const Point< Real , Dim >& n ) const
	{
		LinearFunction f;
		Real len = Point< Real , Dim >::SquareNorm( n );
		if( !len )
		{
			f.gradients *= 0;
			f.offset = -(*this)( p );
		}
		else
		{
			Point< Real , Dim > normal = n / Real( sqrt( double( len ) ) );
			V dot;
			dot *= 0;
			for( int d=0 ; d<Dim ; d++ ) dot += gradients[d] * normal[d];
			for( int d=0 ; d<Dim ; d++ ) f.gradients[d] = gradients[d] - dot * normal[d];
			f.offset *= 0;
			f.offset = -(*this)( p ) + f( p );
		}
		return f;
	}
	template< class V2 , class _R2 >
	operator LinearFunction< V2 , Dim , _R2 > ( void ) const
	{
		LinearFunction< V2 , Dim , _R2 > lf;
		lf.offset = V2 ( offset );
		lf.gradients = Gradient< V2 , Dim , _R2 >( gradients );
		return lf;
	}
	template< class Real >
	Gradient< V , Dim , _R > gradient( const Point< Real , Dim >& p ) const { return gradients; }

	// Warning, this function requires the taking of an inverse, which may fail...
	template< class Real >
	static LinearFunction BestFit( const Point< Real , Dim >* points , const V* values , int count )
	{
		LinearFunction lf;
		V constraint[Dim];
		SquareMatrix< Real , Dim > M , Minv;
		M *= 0;
		for( int d=0 ; d<Dim ; d++ ) constraint[d] *= 0;
		for( int i=0 ; i<count ; i++ )
		{
			for( int k=0 ; k<Dim ; k++ ) for( int l=0 ; l<Dim ; l++ ) M( k , l ) += points[i][k] * points[i][l];
			for( int j=0 ; j<count ; j++ ) for( int k=0 ; k<Dim ; k++ ) for( int l=0 ; l<Dim ; l++ ) M( k , l ) -= points[i][k] * points[j][l] / Real( count ); 

			for( int d=0 ; d<Dim ; d++ ) constraint[d] += values[i] * points[i][d];
			for( int j=0 ; j<count ; j++ ) for( int d=0 ; d<Dim ; d++ ) constraint[d] -= values[j] * points[i][d] / Real( count );
		}
		Minv = M.inverse();

		lf *= 0;
		for( int c=0 ; c<Dim ; c++ ) for( int r=0 ; r<Dim ; r++ ) lf.gradients[r] += constraint[c] * Minv( c , r );
		for( int i=0 ; i<count ; i++ )
		{
			for( int d=0 ; d<Dim ; d++ ) lf.offset += lf.gradients[d] * points[i][d];
			lf.offset -= values[i];
		}
		lf.offset /= Real( count );
		return lf;
	}


	//////////////////////////
	// Vector space methods //
	void Add            ( const LinearFunction& lf ) { this->gradient += lf.gradient , offset += lf.offset; }
	void Scale          ( _R s ) { gradients *= s , offset *= s; }
	//////////////////////////
};

template< class Real , int Dim >
struct OrientedPoint
{
	Point< Real , Dim > position , normal;
	template< class Real2 > operator Point< Real2, Dim > ( void ) const { return Point< Real2 , Dim >( position ); }
};

template< typename Real , typename Data >
struct ProjectiveData
{
	Data data;
	Real weight;
	ProjectiveData( Data d=Data() , Real w=(Real)0 ) : data(d) , weight(w) { ; }
	operator Data (){ return weight!=0 ? data/weight : data*weight; }
	Data value( void ) const { return weight!=0 ? data/weight : data*weight; }
	ProjectiveData& operator += ( const ProjectiveData& p ){ data += p.data , weight += p.weight ; return *this; }
	ProjectiveData& operator -= ( const ProjectiveData& p ){ data -= p.data , weight -= p.weight ; return *this; }
	ProjectiveData& operator *= ( Real s ){ data *= s , weight *= s ; return *this; }
	ProjectiveData& operator /= ( Real s ){ data /= s , weight /= s ; return *this; }
	ProjectiveData  operator +  ( const ProjectiveData& p ) const { return ProjectiveData( data+p.data , weight+p.weight ); }
	ProjectiveData  operator -  ( const ProjectiveData& p ) const { return ProjectiveData( data-p.data , weight-p.weight ); }
	ProjectiveData  operator *  ( Real s ) const { return ProjectiveData( data*s , weight*s ); }
	ProjectiveData  operator /  ( Real s ) const { return ProjectiveData( data/s , weight/s ); }
};

/** This templated class represents a Ray.*/
template< typename Real , unsigned int Dim >
class Ray
{
public:
	/** The starting point of the ray */
	Point< Real , Dim > position;

	/** The direction of the ray */
	Point< Real , Dim > direction;

	/** The default constructor */
	Ray( void );

	/** The constructor settign the the position and direction of the ray */
	Ray( const Point< Real , Dim > &position , const Point< Real , Dim > &direction );

	/** This method computes the translation of the ray by p and returns the translated ray.*/
	Ray  operator +  ( const Point< Real , Dim > &p ) const;

	/** This method translates the current ray by p.*/
	Ray &operator += ( const Point< Real , Dim > &p );

	/** This method computes the translation of the ray by -p and returns the translated ray.*/
	Ray  operator -  ( const Point< Real , Dim > &p ) const;

	/** This method translates the current ray by -p.*/
	Ray &operator -= ( const Point< Real , Dim > &p );

	/** This method returns the point at a distance of t along the ray. */
	Point< Real , Dim > operator() ( double t ) const;
};

/** This method applies a transformation to a ray.*/
template< typename Real , unsigned int Dim >
Ray< Real , Dim > operator * ( const SquareMatrix< Real , Dim+1 > &m , const Ray< Real , Dim > &ray );

/** This function prints out the ray.*/
template< typename Real , unsigned int Dim >
std::ostream &operator << ( std::ostream &stream , const Ray< Real , Dim > &ray )
{
	stream << "[ " << ray.position << " ] [ " << ray.direction << " ]";
	return stream;
}

#if FAST_POINT
// Specialization of the 2D point.
template< class Real >
struct Point< Real , 2 >
{
	Point( void ){ coords[0] = coords[1] = (Real)0.; }
	Point( Real x , Real y ){ coords[0] = x , coords[1] = y; }
	template< class Real2 > Point( const Point< Real2 , 2 >& p ){ coords[0] = (Real)p.coords[0] , coords[1] = (Real)p.coords[1]; }
	template< class Real2 , int Dim > Point( const Point< Real2 , Dim >& p )
	{
		for( int d=0; d<2 && d<Dim ; d++ ) coords[d] = (Real)p.coords[d];
		for( int d=Dim ; d<2 ; d++ ) coords[d] = 0;
	}
	template< class Real2 > Point& operator = ( const Point< Real2 , 2 >& p ){ coords[0] = (Real)p.coords[0] , coords[1] = (Real)p.coords[1] ; return *this; }
	Real& operator[] ( int i ) { return coords[i]; }
	const Real& operator[] ( int i ) const { return coords[i]; }

	Point& operator += ( const Point& p ){ coords[0] += p.coords[0] , coords[1] += p.coords[1] ; return *this; }
	Point& operator -= ( const Point& p ){ coords[0] -= p.coords[0] , coords[1] -= p.coords[1] ; return *this; }
	Point& operator *= ( Real s ){ coords[0] *= s , coords[1] *= s ; return *this; }
	Point& operator /= ( Real s ){ coords[0] /= s , coords[1] /= s ; return *this; }
	Point& operator *= ( const Point &p ){ coords[0] *= p.cords[0] , coords[1] *= p.cords[1] ; return *this; }
	Point& operator /= ( const Point &p ){ coords[0] /= p.cords[0] , coords[1] /= p.cords[1] ; return *this; }

	Point operator - ( void ) const { return Point( -coords[0] , -coords[1] ); }

	Point operator + ( const Point& p ) const { return Point( coords[0] + p.coords[0] , coords[1] + p.coords[1] ); }
	Point operator - ( const Point& p ) const { return Point( coords[0] - p.coords[0] , coords[1] - p.coords[1] ); }
	Point operator * ( Real s ) const { return Point( coords[0] * s , coords[1] * s ); }
	Point operator / ( Real s ) const { return Point( coords[0] / s , coords[1] / s ); }
	Point operator * ( const Point& p ) const { return Point( coords[0] * p.coords[0] , coords[1] * p.coords[1] ); }
	Point operator / ( const Point& p ) const { return Point( coords[0] / p.coords[0] , coords[1] / p.coords[1] ); }

	Real squareNorm( void ) const { return coords[0]*coords[0] + coords[1]*coords[1]; }
	static Real Dot( const Point& p1 , const Point& p2 ){ return p1.coords[0]*p2.coords[0] + p1.coords[1]*p2.coords[1]; }
	static Real SquareNorm( const Point& p ){ return p.coords[0]*p.coords[0] + p.coords[1]*p.coords[1]; }
	static Real Length( const Point& p ){ return (Real)sqrt( SquareNorm(p) ); }
	Real coords[2];

	static Point< Real , 2 > Min( Point< Real , 2 > p , Point< Real , 2 > q ){ Point< Real , 2 > m ; for( int d=0 ; d<2 ; d++ ) m[d] = std::min< Real >( p[d] , q[d] ) ; return m; }
	static Point< Real , 2 > Max( Point< Real , 2 > p , Point< Real , 2 > q ){ Point< Real , 2 > m ; for( int d=0 ; d<2 ; d++ ) m[d] = std::max< Real >( p[d] , q[d] ) ; return m; }

	friend std::ostream& operator << ( std::ostream& os , const Point< Real , 2 > &p )
	{
		os << "( ";
		for( int d=0 ; d<2 ; d++ )
		{
			if( d ) os << " , ";
			os << p[d];
		}
		return os << " )";
	}
};
template< class Real > using Point2D = Point< Real , 2 >;
template< class Real > Point2D< Real > operator * ( Real s , const Point2D< Real >& v ){ return v * s; }
template< class Real > Point2D< Real > operator / ( Real s , const Point2D< Real >& v ){ return v / s; }
template< class Real > Point2D< Real > operator * ( const SquareMatrix< Real , 2 >& M , const Point2D< Real >& v )
{
	return Point2D< Real >( M.coords[0][0] * v[0] + M.coords[1][0] * v[1] , M.coords[0][1] * v[0] + M.coords[1][1] * v[1] );
}
// Specialization of the 3D point.
template< class Real >
struct Point< Real , 3 >
{
	Point( void ){ coords[0] = coords[1] = coords[2] = (Real)0.; }
	Point( Real x , Real y , Real z ){ coords[0] = x , coords[1] = y , coords[2] = z; }
	template< class Real2 > Point( const Point< Real2 , 3 >& p ){ coords[0] = (Real)p.coords[0] , coords[1] = (Real)p.coords[1] , coords[2] = (Real)p.coords[2]; }
#if 0
	template< class Real2 , int Dim > Point( const Point< Real2 , Dim >& p )
	{
		for( int d=0; d<3 && d<Dim ; d++ ) coords[d] = (Real)p.coords[d];
		for( int d=Dim ; d<3 ; d++ ) coords[d] = 0;
	}
#endif
	template< class Real2 > Point& operator = ( const Point< Real2 , 3 >& p ){ coords[0] = (Real)p.coords[0] , coords[1] = (Real)p.coords[1] , coords[2] = (Real)p.coords[2] ; return *this; }
	Real& operator[] ( int i ) { return coords[i]; }
	const Real& operator[] ( int i ) const { return coords[i]; }

	Point& operator += ( const Point& p ){ coords[0] += p.coords[0] , coords[1] += p.coords[1] , coords[2] += p.coords[2] ; return *this; }
	Point& operator -= ( const Point& p ){ coords[0] -= p.coords[0] , coords[1] -= p.coords[1] , coords[2] -= p.coords[2] ; return *this; }
	Point& operator *= ( Real s ){ coords[0] *= s , coords[1] *= s , coords[2] *= s ; return *this; }
	Point& operator /= ( Real s ){ coords[0] /= s , coords[1] /= s , coords[2] /= s ; return *this; }
	Point& operator *= ( const Point& p ){ coords[0] *= p.coords[0] , coords[1] *= p.coords[1] , coords[2] *= p.coords[2] ; return *this; }
	Point& operator /= ( const Point& p ){ coords[0] /= p.coords[0] , coords[1] /= p.coords[1] , coords[2] /= p.coords[2] ; return *this; }

	Point operator - ( void ) const { return Point( -coords[0] , -coords[1] , -coords[2] ); }

	Point operator + ( const Point& p ) const { return Point( coords[0] + p.coords[0] , coords[1] + p.coords[1] , coords[2] + p.coords[2] ); }
	Point operator - ( const Point& p ) const { return Point( coords[0] - p.coords[0] , coords[1] - p.coords[1] , coords[2] - p.coords[2] ); }
	Point operator * ( Real s ) const { return Point( coords[0] * s , coords[1] * s , coords[2] * s ); }
	Point operator / ( Real s ) const { return Point( coords[0] / s , coords[1] / s , coords[2] / s ); }
	Point operator * ( const Point& p ) const { return Point( coords[0] * p.coords[0] , coords[1] * p.coords[1] , coords[2] * p.coords[2] ); }
	Point operator / ( const Point& p ) const { return Point( coords[0] / p.coords[0] , coords[1] / p.coords[1] , coords[2] / p.coords[2] ); }

	Real squareNorm( void ) const { return coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2]; }
	static Real Dot( const Point& p1 , const Point& p2 ){ return p1.coords[0]*p2.coords[0] + p1.coords[1]*p2.coords[1] + p1.coords[2]*p2.coords[2]; }
	static Real SquareNorm( const Point& p ){ return p.coords[0]*p.coords[0] + p.coords[1]*p.coords[1] + p.coords[2]*p.coords[2]; }
	static Real Length( const Point& p ){ return (Real)sqrt( SquareNorm(p) ); }
	static Point CrossProduct( const Point& p1 , const Point& p2 )
	{
		return Point( p1.coords[1]*p2.coords[2] - p1.coords[2]*p2.coords[1], - p1.coords[0]*p2.coords[2] + p1.coords[2]*p2.coords[0] , p1.coords[0]*p2.coords[1] - p1.coords[1]*p2.coords[0] );
	}
	static Point CrossProduct( const Point *p ){ return CrossProduct( p[0] , p[1] ); }
	static Point CrossProduct( Point *p ){ return CrossProduct( p[0] , p[1] ); }

	Real coords[3];
	static Point< Real , 3 > Min( Point< Real , 3 > p , Point< Real , 3 > q ){ Point< Real , 3 > m ; for( int d=0 ; d<3 ; d++ ) m[d] = std::min< Real >( p[d] , q[d] ) ; return m; }
	static Point< Real , 3 > Max( Point< Real , 3 > p , Point< Real , 3 > q ){ Point< Real , 3 > m ; for( int d=0 ; d<3 ; d++ ) m[d] = std::max< Real >( p[d] , q[d] ) ; return m; }

	friend std::ostream &operator << ( std::ostream &os , const Point< Real , 3 > &p )
	{
		os << "( ";
		for( int d=0 ; d<3 ; d++ )
		{
			if( d ) os << " , ";
			os << p[d];
		}
		return os << " )";
	}
};
template< class Real > using Point3D = Point< Real , 3 >;
template< class Real > Point3D< Real > operator * ( Real s , const Point3D< Real >& v ){ return v * s; }
template< class Real > Point3D< Real > operator * ( const SquareMatrix< Real , 3 >& M , const Point3D< Real >& v )
{
	return Point3D< Real >( M.coords[0][0] * v[0] + M.coords[1][0] * v[1] + M.coords[2][0] * v[2] , M.coords[0][1] * v[0] + M.coords[1][1] * v[1] + M.coords[2][1] * v[2] , M.coords[0][2] * v[0] + M.coords[1][2] * v[1] + M.coords[2][2] * v[2] );
}

#else // !FAST_POINT
template< class Real >
class Point2D : public Point<Real,2>
{
public:
	Point2D(void) : Point<Real,2>(){;}
	Point2D(const Point<Real,2>& p) {   memcpy(this->coords,p.coords,sizeof(Real)*2);   }
	Point2D( Real v1 , Real v2 ) { coords[0] = v1 , coords[1] = v2; }
};

template< class Real >
class Point3D : public Point<Real,3>
{
public:
	using Point<Real, 3>::coords;
	Point3D(void) : Point<Real,3>(){;}
	Point3D(const Point<Real,3>& p) { *this = p; }
	Point3D( Real v1 , Real v2 , Real v3 ) { coords[0] = v1 , coords[1] = v2 , coords[2] = v3; }
	Point3D(const Real *p)  { *this = p; }

	static Point3D CrossProduct( const Point3D& p1 , const Point3D & p2 );

	////////////////////////////////////////////////////////////////////////////
	/*! Allow assignment from another point of lower or equal dimension. If the
	//  RHS dimension is lower, copy only the number of elements in RHS.
	//  @param[in]  p   vector from which to copy
	//  @return     reference to the LHS (this) to allow chaining
	*///////////////////////////////////////////////////////////////////////////
	template<int RHSDim>
	Point3D &operator= (const Point<Real, RHSDim> &p)
	{
		assert(RHSDim <= 3);
		for (int d = 0; d < RHSDim; d++)
			coords[d] = p.coords[d];

		return *this;
	}

	////////////////////////////////////////////////////////////////////////////
	/*! Allow assignment from a flat array
	//  @param[in]  p   the value array to copy
	//  @return     reference to the LHS (this) to allow chaining
	*///////////////////////////////////////////////////////////////////////////
	Point3D &operator=(const Real *p)
	{
		memcpy(coords, p, sizeof(Real) * 3);
		return *this;
	}

	////////////////////////////////////////////////////////////////////////////
	/*! Allow assignment from a point of a different real type
	//  @return     reference to the LHS (this) to allow chaining
	*///////////////////////////////////////////////////////////////////////////
	template<typename Real2>
	Point3D &operator=(const Point3D<Real2> &p)
	{
		Point3D<Real2> p2;
		for (int d = 0; d < 3; d++)
			coords[d] = (Real) p.coords[d];

		return *this;
	}

	////////////////////////////////////////////////////////////////////////////
	/*! Allow addition with a flat array
	//  @param[in]  p   the value array to accumulate
	//  @return     copy of result
	*///////////////////////////////////////////////////////////////////////////
	Point3D operator+(const Real *p)
	{
		Point3D result(*this);
		for (int i = 0; i < 3; ++i)
			result.coords[i] += p[i];
		return result;
	}
};
#endif // FAST_POINT

////////////////////////////////////////////////////////////////////////////////
/*! Creates a complex 3d point: p1 + i p2
//  @param[in]  p1      real part
//  @param[in]  p1      imaginary part
//  @return     complex 3d point
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
inline Point3D<std::complex<Real> > ComplexPoint3D(const Point3D<Real> &p1
	, const Point3D<Real> &p2)
{
	return Point3D<std::complex<Real> >(std::complex<Real>(p1[0], p2[0])
		, std::complex<Real>(p1[1], p2[1])
		, std::complex<Real>(p1[2], p2[2]));
}

////////////////////////////////////////////////////////////////////////////////
/*! Extracts the real vector part from a complex vector
//  @param[in]  vector to process
//  @return     real part
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
inline Point3D<Real> real(const Point3D<std::complex<Real> > &p)
{
	return Point3D<Real>(real(p[0]), real(p[1]), real(p[2]));
}

////////////////////////////////////////////////////////////////////////////////
/*! Extracts the imaginary vector part from a complex vector
//  @param[in]  vector to process
//  @return     imaginary part
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
inline Point3D<Real> imag(const Point3D<std::complex<Real> > &p)
{
	return Point3D<Real>(imag(p[0]), imag(p[1]), imag(p[2]));
}

////////////////////////////////////////////////////////////////////////////////
/*! Separately normalizes both the real and the imaginary part of a complex 3D
//  vector
//  @param[inout]   p   the vector to normalize
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
inline void NormalizeParts(Point3D<std::complex<Real> > &p)
{
	Point3D<Real> realPart = real(p);
	Point3D<Real> imagPart = imag(p);
	realPart /= sqrt(SquareNorm(realPart));
	imagPart /= sqrt(SquareNorm(imagPart));
	p = ComplexPoint3D(realPart, imagPart);
}

////////////////////////////////////////////////////////////////////////////////
/*! Saves on a sqrt() operation when the vector is unit.
//  @param[inout]   p   the vector to normalize
*///////////////////////////////////////////////////////////////////////////////
template<typename Real>
void safe_normalize(Point3D<Real> &p)
{
	Real normSq = SquareNorm(p);
	if ((normSq != Real(0)) && (std::abs(1 - normSq) >= 1.0e-16))
		p *= (1 / sqrt(normSq));
}

////////////////////////////////////////////////////////////////////////////////
/*! Linearly interpolates between two points
//  @param[in]  alpha       blending factor (alpha = 0 ==> p1)
//  @param[in]  p1, p2      points to interpolate
//  @return     interpolation result
*///////////////////////////////////////////////////////////////////////////////
template<typename Num>
Point3D<Num> LinInterp(Num alpha, const Point3D<Num> &p1,
	const Point3D<Num> &p2)
{
	return (1 - alpha) * p1 + alpha * p2;
}

template< class V , class _R = typename V::R >
class Gradient3D : public Gradient< V , 3 , _R >
{
public:
	Gradient3D( void ) : Gradient< V , 3 , _R >( ){ ; }
	Gradient3D( const Gradient< V , 3 , _R >& lf ) { for( int d=0 ; d<3 ; d++ ) this->gradients[d] = lf.gradients[d]; }
};

////////////////////////////////////////////////////////////////////////////////
/*! Class to represent a linear function mapping R^3 to V
//  (as I understand it--JP)
//  @tparam  V  the range type
//  @tparam _R  the floating point representation
*///////////////////////////////////////////////////////////////////////////////
template< class V , class _R = typename V::R >
class LinearFunction3D : public LinearFunction< V , 3 , _R >
{
public:
	LinearFunction3D( void ) : LinearFunction< V , 3 , _R >( ){ ; }
	LinearFunction3D( const LinearFunction< V , 3 , _R >& lf )
	{
		for( int d=0 ; d<3 ; d++ ) this->gradients[d] = lf.gradients[d];
		this->offset = lf.offset;
	}
	template< class Real >
	static LinearFunction3D GetInterpolant( const Point3D< Real >* vertices , const V* values , Point3D< Real > n )
	{
		Point3D< Real > p1 = Point3D< Real >( vertices[1] - vertices[0] );
		Point3D< Real > p2 = Point3D< Real >( vertices[2] - vertices[0] );
		Point3D< Real > v1 = Point3D< Real >::CrossProduct( p2 , n  );
		Point3D< Real > v2 = Point3D< Real >::CrossProduct( p1 , n  );

		Real d1 = Point3D< Real >::Dot( v1 , p1 );
		Real d2 = Point3D< Real >::Dot( v2 , p2 );
		if( !d1 || !d2 )
		{
			LinearFunction3D< V , _R > lf;
			lf.gradients *= 0;
#if PAN_FIX
			lf.offset = -( values[0] + values[1] + values[2] ) / 3;
#else // !PAN_FIX
			lf.offset =  ( values[0] + values[1] + values[2] ) / 3;
#endif // PAN_FIX
			return lf;
		}
		else
		{
			v1 /= d1;
			v2 /= d2;
		}

		LinearFunction3D< V , _R > lf;
		lf.offset =
			- ( values[0] * ( 1.0 + Point3D< Real >::Dot( vertices[0] , v1 ) + Point3D< Real >::Dot( vertices[0] , v2 ) )
			- values[1] * Point3D< Real >::Dot( vertices[0] , v1 )
			- values[2] * Point3D< Real >::Dot( vertices[0] , v2 ) );
		for( int d=0 ; d<3 ; d++ ) lf.gradients[d] = - values[0]*v1[d] - values[0]*v2[d] + values[1]*v1[d] + values[2]*v2[d];
		return lf;
	}
	template< class Real >
	static LinearFunction3D GetInterpolant( const Point3D< Real >& v1 ,
		const Point3D< Real >& v2 ,
		const Point3D< Real >& v3 ,
		const V& s1 , const V& s2 , const V s3 ,
		Point3D< Real > n )
	{
		Point3D< Real > p1 = Point3D< Real >( v2 - v1 );
		Point3D< Real > p2 = Point3D< Real >( v3 - v1 );
		Point3D< Real > _v1 = Point3D< Real >::CrossProduct( p2 , n  );
		Point3D< Real > _v2 = Point3D< Real >::CrossProduct( p1 , n  );

		Real d1 = Point3D< Real >::Dot( _v1 , p1 );
		Real d2 = Point3D< Real >::Dot( _v2 , p2 );
		if( !d1 || !d2 )
		{
			LinearFunction3D< V , _R > lf;
			lf.gradients *= 0;
#if PAN_FIX
			lf.offset = -( s1 + s2 + s3 ) / _R(3);
#else // !PAN_FIX
			lf.offset = ( s1 + s2 + s3 ) / _R(3);
#endif // PAN_FIX
			return lf;
		}
		else
		{
			_v1 /= d1;
			_v2 /= d2;
		}

		LinearFunction3D< V , _R > lf;
		lf.offset = - ( s1 * ( Real(1.0)
			+ Point3D< Real >::Dot( v1 , _v1 )
			+ Point3D< Real >::Dot( v1 , _v2 ))
			- s2 * Point3D< Real >::Dot( v1 , _v1 )
			- s3 * Point3D< Real >::Dot( v1 , _v2 ) );
		for( int d=0 ; d<3 ; d++ ) lf.gradients[d] = - s1*_v1[d] - s1*_v2[d] + s2*_v1[d] + s3*_v2[d];
		return lf;
	}
	template< class Real >
	static LinearFunction3D GetInterpolant( const Point3D< Real >* vertices , const V* values )
	{
		Point3D< Real > p1 = Point3D< Real >( vertices[1] - vertices[0] );
		Point3D< Real > p2 = Point3D< Real >( vertices[2] - vertices[0] );
		Point3D< Real > n  = Point3D< Real >::CrossProduct( p1 , p2 );
		return GetInterpolant( vertices , values , n );
	}
	template< class Real >
	static LinearFunction3D GetInterpolant( const Point3D< Real >& v1 , const Point3D< Real >& v2 , const Point3D< Real >& v3 , const V& s1 , const V& s2 , const V& s3 )
	{
		Point3D< Real > p1 = Point3D< Real >( v2 - v1 );
		Point3D< Real > p2 = Point3D< Real >( v3 - v1 );
		Point3D< Real > n  = Point3D< Real >::CrossProduct( p1 , p2 );
		return GetInterpolant( v1 ,v2 , v3 , s1 , s2 , s3 , n );
	}
};

template<class Real>
class OrientedPoint2D : public OrientedPoint<Real,2>{;};
template<class Real>
class OrientedPoint3D : public OrientedPoint<Real,3>{;};

template< typename Real , unsigned int Dim >
class XForm : public SquareMatrix< Real , Dim >
{
	using SquareMatrix< Real , Dim >::coords;
public:
	using SquareMatrix< Real , Dim >::Identity;

	XForm( void ) : SquareMatrix< Real , Dim >(){;}
	XForm( const SquareMatrix< Real , Dim > &xForm ){ memcpy( coords , xForm.coords , sizeof(Real)*Dim*Dim ); };
	XForm( const SquareMatrix< Real , Dim-1 > &L , Point< Real , Dim-1 > t=Point< Real , Dim >() ) : XForm()
	{
		for( unsigned int i=0 ; i<Dim-1 ; i++ ) for( unsigned int j=0 ; j<Dim-1 ; j++ ) coords[i][j] = L.coords[i][j];
		for( unsigned int j=0 ; j<Dim-1 ; j++ ) coords[Dim-1][j] = t[j];
		coords[Dim-1][Dim-1] = (Real)1.;
	}
	Point< Real , Dim > operator * ( Point< Real , Dim > p ) const { return SquareMatrix< Real , Dim >::operator * ( p ); }
	Point< Real , Dim-1 > operator * ( Point< Real , Dim-1 > p ) const
	{
		Point< Real , Dim > _p;
		for( unsigned int d=0 ; d<Dim-1 ; d++ ) _p[d] = p[d];
		_p[Dim-1] = 1;
		_p = SquareMatrix< Real , Dim >::operator * ( _p );
		_p /= _p[Dim-1];
		for( int d=0 ; d<Dim-1 ; d++ ) p[d] = _p[d];
		return p;
	}

	static XForm Scale( Point< Real , Dim-1 > s )
	{
		XForm xForm = Identity();
		for( int d=0 ; d<Dim-1 ; d++ ) xForm(d,d) = s[d];
		return xForm;
	}

	static XForm Translate( Point< Real , Dim-1 > t )
	{
		XForm xForm = Identity();
		for( int d=0 ; d<Dim ; d++ ) xForm(Dim-1,d) = t[d];
		return xForm;
	}
};
template< typename Real > using XForm4x4 = XForm< Real , 4 >;
template< typename Real > using XForm3x3 = XForm< Real , 3 >;
template< typename Real > using XForm2x2 = XForm< Real , 2 >;

///////////////
// Simplices //
///////////////
template< unsigned int K > struct Factorial{ static const unsigned long long Value = Factorial< K-1 >::Value * K; };
template<> struct Factorial< 0 >{ static const unsigned long long Value = 1; };

template< class Real , unsigned int Dim , unsigned int K >
struct Simplex
{
	Point< Real , Dim > p[K+1];
	Simplex( void ){ static_assert( K<=Dim , "[ERROR] Bad simplex dimension" ); }
	Point< Real , Dim >& operator[]( unsigned int k ){ return p[k]; }
	const Point< Real , Dim >& operator[]( unsigned int k ) const { return p[k]; }
	Real measure( void ) const { return (Real)sqrt( squareMeasure() ); }
	Real squareMeasure( void ) const
	{
		XForm< Real , K > mass;
		for( unsigned int i=1 ; i<=K ; i++ ) for( unsigned int j=1 ; j<=K ; j++ ) mass(i-1,j-1) = Point< Real , Dim >::Dot( p[i]-p[0] , p[j]-p[0] );
		return mass.determinant() / ( Factorial< K >::Value * Factorial< K >::Value );
	}
	Point< Real , Dim > center( void ) const
	{
		Point< Real , Dim > c;
		for( unsigned int k=0 ; k<=K ; k++ ) c += p[k];
		return c / (K+1);
	}
	void split( const Real values[K+1] , std::vector< Simplex >& back , std::vector< Simplex >& front ) const;
	void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const;
	Point< Real , Dim > operator()( const Real weights[K+1] ) const
	{
		Real q;
		for( unsigned int k=0 ; k<=K ; k++ ) q += p[k] * weights[k];
		return q;
	}

	template< unsigned int _K=K >
	typename std::enable_if< _K==Dim-1 , Point< Real , Dim > >::type normal( void ) const
	{
		Point< Real , Dim > d[Dim-1];
		for( int k=1 ; k<Dim ; k++ ) d[k-1] = p[k] - p[0];
		return Point< Real , Dim >::CrossProduct( d );
	}

	template< unsigned int _K=K >
	typename std::enable_if< _K==Dim-1 , Real >::type volume( void ) const
	{
		// Goal:
		//		Compute \int_V 1 dv
		// Using the fact that 1 = div(V), with V = ( x , 0 , ... ) and Stokes' Theorem, we have:
		//		\int_V 1 = \int_dV < V , n >
		Point< Real , Dim > c = center() , n = normal();
		return ( Point< Real , Dim >::Dot( c , n ) / Point< Real , Dim >::Length( n ) * measure() ) / Dim;
	}

	Point< Real , Dim > nearest( Point< Real , Dim > point , Real barycentricCoordinates[K+1] ) const;
	Point< Real , Dim > nearest( Point< Real , Dim > point ) const { Real barycentricCoordinates[K+1] ; return nearest( point , barycentricCoordinates ); }

	struct NearestKey
	{
		void init( Simplex simplex );
		Point< Real , Dim > nearest( Point< Real , Dim > point , Real barycentricCoordinates[K+1] ) const { _nearest( point , barycentricCoordinates ) ; return operator()( barycentricCoordinates ); }
		Point< Real , Dim > nearest( Point< Real , Dim > point ) const { Real barycentricCoordinates[K+1] ; return nearest( point , barycentricCoordinates ); }
		Point< Real , Dim > operator()( const Real weights[K+1] ) const
		{
			Point< Real , Dim > q;
			Real weightSum = 0;
			for( unsigned int k=0 ; k<K ; k++ ) q += _dirs[k] * weights[k+1] , weightSum += weights[k+1];
			return q + _base * ( weights[0] + weightSum );
		}

	protected:
		Point< Real , Dim > _base , _dirs[K];
		SquareMatrix< Real , K > _Dinv;
		typename Simplex< Real , Dim , K-1 >::NearestKey _faceKeys[K+1];
		void _nearest( Point< Real , Dim > point , Real barycentricCoordinates[K+1] ) const;

		friend typename Simplex< Real , Dim , K+1 >::NearestKey;
	};
protected:
	void _nearest( Point< Real , Dim > point , Real barycentricCoordinates[K+1] ) const;
	friend Simplex< Real , Dim , K+1 >;
};

template< class Real , unsigned int Dim >	
struct Simplex< Real , Dim , 0 >
{
	Point< Real , Dim > p[1];
	Point< Real , Dim >& operator[]( unsigned int k ){ return p[k]; }
	const Point< Real , Dim >& operator[]( unsigned int k ) const { return p[k]; }
	Real squareMeasure( void ) const { return (Real)1.; }
	Real measure( void ) const { return (Real)1.; }
	Point< Real , Dim > center( void ) const { return p[0]; }
	Point< Real , Dim > operator()( const Real weights[1] ) const { return p[0] * weights[0]; }
#ifdef NEW_SIMPLEX
	void split( const Real values[1] , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
	{
		if( values[0] ) back.push_back( *this );
		else            front.push_back( *this );
	}
	void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
	{
		const Real values[] = { Point< Real , Dim >::Dot( p[0] , pNormal ) - pOffset };
		return split( values , back , front );
	}
#else // !NEW_SIMPLEX
	void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
	{
		if( Point< Real , Dim >::Dot( p[0] , pNormal ) < pOffset ) back.push_back( *this );
		else                                                       front.push_back( *this );
	}
#endif // NEW_SIMPLEX

	Point< Real , Dim > nearest( Point< Real , Dim > point , Real barycentricCoordinates[1] ) const { _nearest( point , barycentricCoordinates ) ; return operator()( barycentricCoordinates ); }
	Point< Real , Dim > nearest( Point< Real , Dim > point ) const { Real barycentricCoordinates[1] ; return nearest( point , barycentricCoordinates ); }

	struct NearestKey
	{
		void init( Simplex simplex ){ _base = simplex[0]; }
		Point< Real , Dim > nearest( Point< Real , Dim > point , Real barycentricCoordinates[1] ) const { _nearest( point , barycentricCoordinates ) ; return operator()( barycentricCoordinates ); }
		Point< Real , Dim > nearest( Point< Real , Dim > point ) const { Real barycentricCoordinates[1] ; return nearest( point , barycentricCoordinates ); }
		Point< Real , Dim > operator()( const Real weights[1] ) const { return _base * weights[0]; }
	protected:
		Point< Real , Dim > _base;
		void _nearest( Point< Real , Dim > point , Real barycentricCoordinates[1] ) const { barycentricCoordinates[0] = (Real)1.; }

		friend typename Simplex< Real , Dim , 1 >::NearestKey;
	};
protected:
	void _nearest( Point< Real , Dim > point , Real barycentricCoordinates[1] ) const { barycentricCoordinates[0] = (Real)1.; }
	friend Simplex< Real , Dim , 1 >;
};

template< class Real , unsigned int Dim , unsigned int K >
Simplex< Real , Dim , K > operator * ( XForm< Real , Dim+1 > xForm , Simplex< Real , Dim , K > simplex )
{
	for( unsigned int k=0 ; k<=K ; k++ ) simplex[k] = xForm * simplex[k];
	return simplex;
}

#ifdef NEW_SIMPLEX
template< typename Index >
struct EdgeTable
{
protected:
	struct _EdgeKey
	{
		Index key1 , key2;
		_EdgeKey( Index k1=0 , Index k2=0 ) : key1(k1) , key2(k2) {}
		bool operator == ( const _EdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
		struct Hasher{ size_t operator()( const _EdgeKey &key ) const { return (size_t)( key.key1 * key.key2 ); } };
	};
	std::unordered_map< _EdgeKey , Index , typename _EdgeKey::Hasher > _edgeTable;
public:
	template< typename InitializationFunction >
	Index &operator()( Index v1 , Index v2 , InitializationFunction &initializationFunction )
	{
		auto iter = _edgeTable.find( _EdgeKey(v1,v2) );
		if( iter==_edgeTable.end() )
		{
			Index idx = initializationFunction();
			_edgeTable[ EdgeKey(v1,v2) ] = idx;
			return idx;
		}
		else return iter->second;
	};
};
#endif // NEW_SIMPLEX

template< unsigned int K , typename Index >
struct SimplexIndex
{
	Index idx[K+1];
	template< class ... Ints >
	SimplexIndex( Ints ... values ){ static_assert( sizeof...(values)==K+1 || sizeof...(values)==0 , "[ERROR] Invalid number of coefficients" ) ; _init( 0 , values ... ); }
	Index &operator[] ( unsigned int i ) { return idx[i] ;}
	const Index &operator[] ( unsigned int i ) const { return idx[i]; }
#ifdef NEW_SIMPLEX
	template< typename Real , typename Vertex >
	void split( const Real values[K+1] , std::vector< Vertex > &vertices , EdgeTable< Index > &edgeTable , std::vector< SimplexIndex >& back , std::vector< SimplexIndex >& front ) const;
#endif // NEW_SIMPLEX
protected:
	void _init( unsigned int k )
	{
		if( !k ) memset( idx , 0 , sizeof(idx) );
		else ERROR_OUT( "Should never be called" );
	}
	template< class ... Ints > void _init( unsigned int k , Index v , Ints ... values )
	{
		idx[k] = v;
		if( k+1<=K ) _init( k+1 , values ... );
	}

	friend std::ostream &operator << ( std::ostream &os , const SimplexIndex &s )
	{
		os << "{ ";
		for( int d=0 ; d<=K ; d++ )
		{
			if( d ) os << " , ";
			os << s[d];
		}
		return os << " }";
	}

};
#ifdef NEW_SIMPLEX
template< typename Index >
struct SimplexIndex< 0 , Index >
{
	Index idx[1];
	SimplexIndex( Index i=0 ){ idx[0]=i; }
	Index &operator[] ( unsigned int i ) { return idx[i] ;}
	const Index &operator[] ( unsigned int i ) const { return idx[i]; }
	template< typename Real , typename Vertex >
	void split( const Real values[1] , std::vector< Vertex > &vertices , EdgeTable< Index > &edgeTable , std::vector< SimplexIndex >& back , std::vector< SimplexIndex >& front ) const
	{
		if( values[0]<0 ) back.push_back( *this );
		else              front.push_back( *this );
	}
};
#endif // NEW_SIMPLEX

template< typename Real , unsigned int Dim , unsigned int K >
struct SimplicialComplex
{
	SimplicialComplex( const std::vector< Simplex< Real , Dim , K > > &simplices ) : _simplices( simplices ){}
	virtual size_t size( void ) const { return _simplices.size(); }
	virtual Simplex< Real , Dim , K > operator[]( size_t idx ) const { return _simplices[idx]; }
protected:
	SimplicialComplex( void ) :_simplices(__simplices) {}
	const std::vector< Simplex< Real , Dim , K > > &_simplices;
	const std::vector< Simplex< Real , Dim , K > > __simplices;
};

template< typename Real , unsigned int Dim , unsigned int K , typename IndexType >
struct IndexedSimplicialComplex : public SimplicialComplex< Real , Dim , K >
{
	IndexedSimplicialComplex( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< K , IndexType > > &simplices ) : _vertices(vertices) , _simplices(simplices){}
	IndexedSimplicialComplex( IndexedSimplicialComplex && isc )
	{
		std::swap( _vertices , isc._vertices );
		std::swap( _simplices , isc._simplices );
	}

	size_t size( void ) const { return _simplices.size(); }
	Simplex< Real , Dim , K > operator[]( size_t idx ) const
	{
		Simplex< Real , Dim , K > s;
		for( unsigned int k=0 ; k<=K ; k++ ) s[k] = _vertices[ _simplices[idx][k] ];
		return s;
	}
protected:
	const std::vector< Point< Real , Dim > > &_vertices;
	const std::vector< SimplexIndex< K , IndexType > > &_simplices;
};

template< typename Real , unsigned int Dim , unsigned int K >
struct TransformedSimplicialComplex : public SimplicialComplex< Real , Dim , K >
{
	TransformedSimplicialComplex( const SimplicialComplex< Real , Dim , K > &simplicialComplex , const XForm< Real , Dim+1 > &xForm ) : _simplicialComplex(simplicialComplex) , _xForm(xForm){}
	size_t size( void ) const { return _simplicialComplex.size(); }
	Simplex< Real , Dim , K > operator[]( size_t idx ) const { return _xForm * _simplicialComplex[idx]; }
protected:
	const SimplicialComplex< Real , Dim , K > &_simplicialComplex;
	XForm< Real , Dim+1 > _xForm;
};


template< class Real > Point2D< Real > RandomDiskPoint( void );
template< class Real > Point3D< Real > RandomBallPoint( void );

template< class Real > Point2D< Real > RandomCirclePoint( void );
template< class Real > Point3D< Real > RandomSpherePoint( void );

template<class Real>
XForm3x3<Real> RotationMatrix( Real a , Real b , Real c , Real d );

template<class Real>
XForm3x3<Real> RotationMatrix( const Point3D<Real>& axis , const Real& angle );

template<class Real>
XForm3x3<Real> RandomRotationMatrix( void );

template<class Real>
double Length(const Point3D<Real>& p);

template<class Real>
double SquareLength(const Point3D<Real>& p);

template<class Real>
double Distance(const Point3D<Real>& p1,const Point3D<Real>& p2);

template <class Real>
void CrossProduct(const Point3D<Real>& p1,const Point3D<Real>& p2,Point3D<Real>& p);

template <class Real>
Point3D<Real> CrossProduct(const Point3D<Real>& p1,const Point3D<Real>& p2)
{
	// Avoid Point3D<Real>:: in front of every call
	return Point3D<Real>::CrossProduct(p1, p2);
}

template <class Real>
Real Dot(const Point3D<Real>& p1,const Point3D<Real>& p2)
{
	// Avoid Point3D<Real>:: in front of every call
	return Point3D<Real>::Dot(p1, p2);
}

template <class Real>
Real SquareNorm(const Point3D<Real>& p)
{
	// Avoid Point3D<Real>:: in front of every call
	return Point3D<Real>::SquareNorm(p);
}

template<class Real, int Dim>
Real SquareNorm(const Point<Real, Dim> &p)
{
	// Avoid Point<Real, Dim>:: in front of every call
	return Point<Real, Dim>::SquareNorm(p);
}

template< class Real >
double SquareNorm( std::complex< Real > c )
{
	return c.real()*c.real() + c.imag()*c.imag();
}

template<class Real>
void Transform(const XForm4x4<Real>& xForm,const Point3D<Real>& p,Point3D<Real>& q);

template<class Real>
void TransformNoTranslate(const XForm4x4<Real>& xForm,const Point3D<Real>& p,Point3D<Real>& q);

template< class Real , int Count >
class BarycentricCoordinatesImplementation
{
	template< class _Real , int _Count >
	friend class BarycentricCoordinatesImplementation;
protected:
	template< class Vertex , int Dim >
	static void _BarycentricCoordinates( const Vertex& p , const Vertex v[Count] , Real a[Count] , bool inside );
public:
	template< int Dim >
	static void BarycentricCoordinates( const Point< Real , Dim >& p , const Point< Real , Dim > v[Count] , Real a[Count] , bool inside ){ return _BarycentricCoordinates< Point< Real , Dim > , Dim >( p , v , a , inside ); }
	static void BarycentricCoordinates( const Point3D< Real >& p , const Point3D< Real > v[Count] , Real a[Count] , bool inside ){ return _BarycentricCoordinates< Point3D< Real > , 3 >( p , v , a , inside ); }
};
template< class Real >
class BarycentricCoordinatesImplementation< Real , 1 >
{
	template< class _Real , int _Count >
	friend class BarycentricCoordinatesImplementation;
protected:
	template< class Vertex , int Dim > static void _BarycentricCoordinates( const Vertex& p , const Vertex v[1] , Real a[1] , bool inside ){ a[0] = (Real)1; }
};
template< class Real , int Dim , int Count >
void BarycentricCoordinates( const Point< Real , Dim >& p , const Point< Real , Dim > v[Count] , Real a[Count] , bool inside=false )
{
	return BarycentricCoordinatesImplementation< Real , Count >::template BarycentricCoordinates< Dim >( p , v , a , inside );
}
template< class Real , int Count >
void BarycentricCoordinates( const Point3D< Real >& p , const Point3D< Real > v[Count] , Real a[Count] , bool inside=false )
{
	return BarycentricCoordinatesImplementation< Real , Count >::template BarycentricCoordinates( p , v , a , inside );
}

////////////////////////////////////////////////////////////////////////////////
/*! Uses a barycentric coordinate vector to interpolate three data values
//  @param[in]  coords      bary centric coordinates
//  @param[in]  d0, d1, d2  data values to interpolate
//  @return     interpolated data value
*///////////////////////////////////////////////////////////////////////////////
template<typename BaryCoords, typename DataType>
inline DataType BarycentricInterpolate(const BaryCoords &coords
	, const DataType &d0, const DataType &d1, const DataType &d2);

////////////////////////////////////////////////////////////////////////////////
/*! Computes a triangle's inscribed circle
//  http://en.wikipedia.org/wiki/Incircle
//  @param[in]  p0, p1, p2      triangle vertex positions
//  @param[out] center          incircle center
//  @param[out] radius          incircle radius
*///////////////////////////////////////////////////////////////////////////////
template<typename PointType, typename Real>
inline void Incircle(const PointType &p0, const PointType &p1,
	const PointType &p2, PointType &center, Real &radius);

////////////////////////////////////////////////////////////////////////////////
/*! Computes a triangle's circumscribed circle
//  http://en.wikipedia.org/wiki/Circumscribed_circle
//  @param[in]  p0, p1, p2      triangle vertex positions
//  @param[in]  tri             triangle to process
//  @param[out] center          incircle center
*///////////////////////////////////////////////////////////////////////////////
template<typename PointType, typename Real>
inline void Circumcircle(const PointType &p0, const PointType &p1
	, const PointType &p2, PointType &center, Real &radius);

class Edge
{
public:
	double p[2][2];
	double Length(void) const{
		double d[2];
		d[0]=p[0][0]-p[1][0];
		d[1]=p[0][1]-p[1][1];

		return sqrt(d[0]*d[0]+d[1]*d[1]);
	}
};
class Triangle
{
public:
	double p[3][3];
	double Area(void) const{
		double v1[3],v2[3],v[3];
		for(int d=0;d<3;d++){
			v1[d]=p[1][d]-p[0][d];
			v2[d]=p[2][d]-p[0][d];
		}
		v[0]= v1[1]*v2[2]-v1[2]*v2[1];
		v[1]=-v1[0]*v2[2]+v1[2]*v2[0];
		v[2]= v1[0]*v2[1]-v1[1]*v2[0];
		return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])/2;
	}
	double AspectRatio(void) const{
		double d=0;
		int i,j;
		for(i=0;i<3;i++){
			for(i=0;i<3;i++)
				for(j=0;j<3;j++){d+=(p[(i+1)%3][j]-p[i][j])*(p[(i+1)%3][j]-p[i][j]);}
		}
		return Area()/d;
	}
};

class ConeVertex
{
public:
	/** Index of vertex in the vertex list */
	int v;
	/** Vertex index quadrupled so it becomes integer (4 for non-singular
	*  vertices) */
	char two_cone_angle_over_pi;

	ConeVertex()
		: v(-1), two_cone_angle_over_pi(0) { }
	ConeVertex(int i, int doubleConeAngleOverPi)
		: v(i), two_cone_angle_over_pi(doubleConeAngleOverPi)
	{
	}
};

#if 1
typedef SimplexIndex< 1 , int > EdgeIndex;
typedef SimplexIndex< 2 , int > TriangleIndex;
typedef SimplexIndex< 3 , int > TetrahedronIndex;
#else

class EdgeIndex
{
	unsigned int _v[2];
public:
	EdgeIndex( void ) { _v[0] = _v[1] = 0; }
	EdgeIndex( unsigned int v0 , unsigned int v1 ){ _v[0] = v0 , _v[1] = v1; }
	unsigned int& operator[] ( int idx ) { return _v[idx]; }
	const unsigned int& operator[] ( int idx ) const { return _v[idx]; }
};

class TriangleIndex
{
protected:
	unsigned int _v[3];
public:
	TriangleIndex( void ) { _v[0] = _v[1] = _v[2] = 0; }
	TriangleIndex( unsigned int v0 , unsigned int v1 , unsigned int v2 ){ _v[0] = v0 , _v[1] = v1 , _v[2] = v2; }
	unsigned int &operator[]( unsigned int idx ) { return _v[idx]; }
	unsigned int  operator[]( unsigned int idx ) const { return _v[idx]; }
};
std::ostream &operator << ( std::ostream &os , const TriangleIndex &t )
{
	return os << "( " << t[0] << " , " << t[1] << " , " << t[2] << " ) ";
}

struct TetrahedronIndex
{
protected:
	unsigned int _v[4];
public:
	TetrahedronIndex( void ) { _v[0] = _v[1] = _v[2] = _v[3] = 0; }
	TetrahedronIndex( unsigned int v0 , unsigned int v1 , unsigned int v2 , unsigned int v3 ){ _v[0] = v0; _v[1] = v1; _v[2] = v2; _v[3] = v3; }
	unsigned int &operator[]( unsigned int idx ) { return _v[idx]; }
	unsigned int  operator[]( unsigned int idx ) const { return _v[idx]; }
};
#endif

template< class Data >
class TriangleIndexWithData : public TriangleIndex
{
public:
	Data data[3];
	TriangleIndexWithData( void ) { idx[0] = idx[1] = idx[2] = 0 ;}
	TriangleIndexWithData( unsigned int v0 , unsigned int v1 , unsigned int v2 ){ idx[0] = v0 , idx[1] = v1 , idx[2] = v2; }
	TriangleIndexWithData( unsigned int v0 , unsigned int v1 , unsigned int v2 , Data d0 , Data d1 , Data d2 ){ idx[0] = v0 , idx[1] = v1 , idx[2] = v2 , data[0] = d0 , data[1] = d1 , data[2] = d2; }
};

class SeamEdge : public EdgeIndex
{
public:
	// NOTE: edge is from index 1 to index 0
	/** The rotation across the seam edge */
	unsigned char rot;
	/** The face with which this seam was associated in the obj file
	*  (used to make writing the seams back to an obj file easy) */
	unsigned int face_idx;
};


template< class Real , class Vertex >
void Subdivide1To4( std::vector< TriangleIndex >& triangles , std::vector< Vertex >& vertices )
{
#define EDGE_KEY( i1 , i2 ) ( (i1)>(i2) ? ( ( (long long) (i1) )<<32 ) | ( (long long) (i2) ) : ( ( (long long) (i2) )<<32 ) | ( (long long) (i1) ) )

	std::vector< TriangleIndex > _triangles( triangles.size()*4 );
	std::vector< Vertex > _vertices = vertices;
	
	std::unordered_map< long long , int > vMap;
	for( int i=0 ; i<triangles.size() ; i++ )	
	{
		long long keys[] = { EDGE_KEY( triangles[i][1] , triangles[i][2] ) , EDGE_KEY( triangles[i][2] , triangles[i][0] ) , EDGE_KEY( triangles[i][0] , triangles[i][1] ) };
		int eIndex[3];
		for( int j=0 ; j<3 ; j++ )
			if( vMap.find( keys[j] )==vMap.end() )
			{
				vMap[ keys[j] ] = eIndex[j] = (int)_vertices.size();
				_vertices.push_back( ( vertices[ triangles[i][(j+1)%3] ] + vertices[ triangles[i][(j+2)%3] ] ) / 2 );
			}
			else eIndex[j] = vMap[ keys[j] ];
		_triangles[4*i+0] = TriangleIndex( eIndex[0] , eIndex[1] , eIndex[2] );
		_triangles[4*i+1] = TriangleIndex( triangles[i][0] , eIndex[2] , eIndex[1] );
		_triangles[4*i+2] = TriangleIndex( eIndex[2] , triangles[i][1] , eIndex[0] );
		_triangles[4*i+3] = TriangleIndex( eIndex[1] , eIndex[0] , triangles[i][2] );
	}
	triangles = _triangles;
	vertices = _vertices;
#undef EDGE_KEY
}
template< class Real , class Vertex , class Data >
void Subdivide1To4( std::vector< TriangleIndexWithData< Data > >& triangles , std::vector< Vertex >& vertices )
{
#define EDGE_KEY( i1 , i2 ) ( (i1)>(i2) ? ( ( (long long) (i1) )<<32 ) | ( (long long) (i2) ) : ( ( (long long) (i2) )<<32 ) | ( (long long) (i1) ) )

	std::vector< TriangleIndex > _triangles( triangles.size()*4 );
	std::vector< Vertex > _vertices = vertices;
	
	std::unordered_map< long long , int > vMap;
	for( int i=0 ; i<triangles.size() ; i++ )	
	{
		long long keys[] = { EDGE_KEY( triangles[i][1] , triangles[i][2] ) , EDGE_KEY( triangles[i][2] , triangles[i][0] ) , EDGE_KEY( triangles[i][0] , triangles[i][1] ) };
		Data data[3];
		int eIndex[3];
		for( int j=0 ; j<3 ; j++ )
		{
			if( vMap.find( keys[j] )==vMap.end() )
			{
				vMap[ keys[j] ] = eIndex[j] = (int)_vertices.size();
				_vertices.push_back( ( vertices[ triangles[i][(j+1)%3] ] + vertices[ triangles[i][(j+2)%3] ] ) / 2 );
			}
			else eIndex[j] = vMap[ keys[j] ];
			data[j] = ( triangles[i].data[(j+1)%3] + triangles[i].data[(j+2)%3] ) / 2;
		}
		_triangles[4*i+0] = TriangleIndex( eIndex[0] , eIndex[1] , eIndex[2] , data[0] , data[1] , data[2] );
		_triangles[4*i+1] = TriangleIndex( triangles[i][0] , eIndex[2] , eIndex[1] , triangles[i][0].data , data[1] , data[2] );
		_triangles[4*i+2] = TriangleIndex( eIndex[2] , triangles[i][1] , eIndex[0] , data[2] , triangles[i][1].data , data[0] );
		_triangles[4*i+3] = TriangleIndex( eIndex[1] , eIndex[0] , triangles[i][2] , data[1] , data[0] , triangles[i][2].data );
	}
	triangles = _triangles;
	vertices = _vertices;
#undef EDGE_KEY
}

#if 0
////////////////////////////////////////////////////////////////////////////////
/*! @class ParameterizedTriangleIndex
//  Holds holds both the vertex and uv indices of each triangle corner.
*///////////////////////////////////////////////////////////////////////////////
class ParameterizedTriangleIndex : public TriangleIndex
{
	using TriangleIndex::_v;
	int m_uv[3];
public: 
	ParameterizedTriangleIndex()
		: TriangleIndex() { m_uv[0] = m_uv[1] = m_uv[2] = 0; }

	ParameterizedTriangleIndex(int v0, int v1, int v2)
		: TriangleIndex(v0, v1, v2) { m_uv[0] = m_uv[1] = m_uv[2] = 0; }
	ParameterizedTriangleIndex(const TriangleIndex &t)
		: TriangleIndex(t) { m_uv[0] = m_uv[1] = m_uv[2] = 0; }
	int &uv(int idx)        { return m_uv[idx]; }
	int  uv(int idx) const  { return m_uv[idx]; }
	////////////////////////////////////////////////////////////////////////////
	/*! Typecast to TriangleIndex strips uv indices
	*///////////////////////////////////////////////////////////////////////////
	operator TriangleIndex() const
	{
		TriangleIndex out( _v[0] , _v[1] , _v[2] );
		return out;
	}
	operator TriangleIndex&()
	{
		return *((TriangleIndex *) this);
	}
};
#endif

#if 1
template< class Real , unsigned int Dim >
class MinimalAreaTriangulation
{
	static double _Area( Point< Real , Dim > v0 , Point< Real , Dim > v1 , Point< Real , Dim > v2 );
	double *_bestTriangulation;
	size_t *_midPoint;
	double _GetArea( size_t i , size_t j , const std::vector< Point< Real , Dim > > &vertices );
	template< typename Index >
	void _GetTriangulation( size_t i , size_t j , const std::vector< Point< Real , Dim > > &vertices,std::vector< SimplexIndex< 2 , Index > > &triangles , size_t &idx );
public:
	MinimalAreaTriangulation( void );
	~MinimalAreaTriangulation( void );
	double GetArea( const std::vector< Point< Real , Dim > > &vertices);
	template< typename Index >
	void GetTriangulation( const std::vector< Point< Real , Dim > > &vertices , std::vector< SimplexIndex< 2 , Index > > &triangles );
};
#else
template< class Real >
class MinimalAreaTriangulation
{
	double* bestTriangulation;
	int* midPoint;
	double GetArea( const int& i , const int& j , const std::vector<Point3D<Real> >& vertices );
	void GetTriangulation( const int& i , const int& j , const std::vector<Point3D<Real> >& vertices,std::vector<TriangleIndex>& triangles , int& idx);
public:
	MinimalAreaTriangulation(void);
	~MinimalAreaTriangulation(void);
	double GetArea(const std::vector<Point3D<Real> >& vertices);
	void GetTriangulation( const std::vector<Point3D<Real> >& vertices , std::vector<TriangleIndex>& triangles );
};
#endif

template<class Vertex>
class Mesh
{
public:
	std::vector<Vertex> vertices;
	std::vector<std::vector<int> > polygons;
};

template< class Real >
void SplitTriangle( std::vector< Point3D< Real > >& vertices , TriangleIndex triangle , Point3D< Real > pNormal , Real pOffset ,
	std::vector< TriangleIndex >& backTriangles , std::vector< TriangleIndex >& frontTriangles );
template< class Real >
void SplitTriangle( std::vector< Point3D< Real > >& vertices , TriangleIndex triangle , int direction , Real offset ,
	std::vector< TriangleIndex >& backTriangles , std::vector< TriangleIndex >& frontTriangles );

template< class Real > Point3D< Real > NearestPointOnTriangle( Point3D< Real > point , const Point3D< Real > triangle[3] , Real* b );
template< class Real > Point3D< Real > NearestPointOnEdge( Point3D< Real > point , const Point3D< Real > edge[2] , Real& b0 , Real& b1 );

#include "Geometry.inl"
#endif // GEOMETRY_INCLUDED
