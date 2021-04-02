/*
Copyright (c) 2019, Michael Kazhdan
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

///////////////////
// Polynomial 1D //
///////////////////
template< unsigned int Degree >
Polynomial< 1 , Degree >::Polynomial( void ){ memset( _coefficients , 0 , sizeof( _coefficients ) ); }

template< unsigned int Degree >
Polynomial< 1 , Degree >::Polynomial( double c ) : Polynomial() { _coefficients[0] = c; }

template< unsigned int Degree >
template< typename ... Doubles >
Polynomial< 1 , Degree >::Polynomial( Doubles ... coefficients )
{
	static_assert( sizeof...(coefficients)<=(Degree+1) , "[ERROR] More coefficients than the degree supports" );
	memset( _coefficients , 0 , sizeof( _coefficients ) );
	const double c[] = { coefficients ... };
	memcpy( _coefficients , c , sizeof(c) );
}

template< unsigned int Degree >
template< unsigned int _Degree >
Polynomial< 1 , Degree >::Polynomial( const Polynomial< 1 , _Degree > &p )
{
	for( int d=0 ; d<=Degree && d<=_Degree ; d++ ) _coefficients[d] = p._coefficients[d];
	for( int d=_Degree+1 ; d<=Degree ; d++ ) _coefficients[d] = 0;
}

template< unsigned int Degree >
template< unsigned int _Degree >
Polynomial< 1 , Degree > &Polynomial< 1 , Degree >::operator= ( const Polynomial< 1 , _Degree > &p )
{
	for( int d=0 ; d<=Degree && d<=_Degree ; d++ ) _coefficients[d] = p._coefficients[d];
	for( int d=_Degree+1 ; d<=Degree ; d++ ) _coefficients[d] = 0;
	return *this;
}

template< unsigned int Degree >
const double &Polynomial< 1 , Degree >::_coefficient( const unsigned int indices[] , unsigned int maxDegree ) const
{
	if( indices[0]>maxDegree ) ERROR_OUT( "degree out of bounds: %d > %d\n" , indices[0] , maxDegree );
	return _coefficients[ indices[0] ];
}

template< unsigned int Degree >
double &Polynomial< 1 , Degree >::_coefficient( const unsigned int indices[] , unsigned int maxDegree )
{
	if( indices[0]>maxDegree ) ERROR_OUT( "degree out of bounds: %d > %d\n" , indices[0] , maxDegree );
	return _coefficients[ indices[0] ];
}

template< unsigned int Degree >
double Polynomial< 1 , Degree >::_evaluate( const double coordinates[] , unsigned int maxDegree ) const
{
	double value = 0 , tmp = 1;
	for( unsigned int d=0 ; d<=Degree && d<=maxDegree ; d++ )
	{
		value += tmp * _coefficients[d];
		tmp *= coordinates[0];
	}
	return value;
}

template< unsigned int Degree >
Polynomial< 1 , Degree > Polynomial< 1 , Degree >::_evaluate( const Ray< double , 1 > &ray , unsigned int maxDegree ) const
{
	Polynomial< 1 , 1 > _p = Polynomial< 1 , 1 >( ray.position[0] , ray.direction[0] );
	Polynomial< 1 , Degree > p( 0. ) , __p( 1. );
	for( unsigned int d=0 ; d<=maxDegree ; d++ )
	{
		p += __p * _coefficients[d];
		__p = __p * _p;
	}
	return p;
}

template< unsigned int Degree >
bool Polynomial< 1 , Degree >::_isZero( unsigned int maxDegree ) const
{
	for( unsigned int d=0 ; d<=maxDegree ; d++ ) if( _coefficients[d]!=0 ) return false;
	return true;
}

template< unsigned int Degree >
bool Polynomial< 1 , Degree >::_isConstant( unsigned int maxDegree ) const
{
	for( unsigned int d=1 ; d<=maxDegree ; d++ ) if( _coefficients[d]!=0 ) return false;
	return true;
}

template< unsigned int Degree >
const double &Polynomial< 1 , Degree >::operator[]( unsigned int d ) const { return _coefficient( &d , Degree ); }

template< unsigned int Degree >
double &Polynomial< 1 , Degree >::operator[]( unsigned int d ) { return _coefficient( &d , Degree ); }

template< unsigned int Degree >
const double &Polynomial< 1 , Degree >::coefficient( unsigned int d ) const { return _coefficient( &d , Degree ); }

template< unsigned int Degree >
double &Polynomial< 1 , Degree >::coefficient( unsigned int d ) { return _coefficient( &d , Degree ); }

template< unsigned int Degree >
double Polynomial< 1 , Degree >::operator()( double x ) const
{
	double value = _coefficients[0] , tmp = 1;
	for( unsigned int d=1 ; d<=Degree ; d++ )
	{
		tmp *= x;
		value += tmp * _coefficients[d];
	}
	return value;
}

template< unsigned int Degree >
Polynomial< 1 , Degree-1 > Polynomial< 1 , Degree >::d( unsigned int ) const
{
	Polynomial< 1 , Degree-1 > derivative;
	for( int i=0 ; i<Degree ; i++ ) derivative._coefficients[i] = _coefficients[i+1]*(i+1);
	return derivative;
}

template< unsigned int Degree >
Polynomial< 1 , Degree > Polynomial< 1 , Degree >::operator()( const Ray< double , 1 > &ray ) const { return _evaluate( ray , Degree ); }

template< unsigned int Degree >
void Polynomial< 1 , Degree >::Scale( double s )
{
	for( int d=0 ; d<=Degree ; d++ ) _coefficients[d] *= s;
}

template< unsigned int Degree >
void Polynomial< 1 , Degree >::Add( const Polynomial< 1 , Degree > &p )
{
	for( int d=0 ; d<=Degree ; d++ ) _coefficients[d] += p._coefficients[d];
}

template< unsigned int Degree1 , unsigned int Degree2 >
Polynomial< 1 , Degree1 + Degree2 > operator * ( const Polynomial< 1 , Degree1 > &p1 , const Polynomial< 1 , Degree2 > &p2 )
{
	Polynomial< 1 , Degree1 + Degree2 > p;
	for( int d1=0 ; d1<=Degree1 ; d1++ ) for( int d2=0 ; d2<=Degree2 ; d2++ ) p._coefficients[ d1+d2 ] += p1._coefficients[d1] * p2._coefficients[d2];
	return p;
}

template< unsigned int Degree1 , unsigned int Degree2 >
Polynomial< 1 , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< 1 , Degree1 > &p1 , const Polynomial< 1 , Degree2 > &p2 )
{
	Polynomial< 1 , Max< Degree1 , Degree2 >::Value > p;
	for( int d=0 ; d<=Degree1 ; d++ ) p._coefficients[d] += p1._coefficients[d];
	for( int d=0 ; d<=Degree2 ; d++ ) p._coefficients[d] += p2._coefficients[d];
	return p;
}

template< unsigned int Degree1 , unsigned int Degree2 >
Polynomial< 1 , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< 1 , Degree1 > &p1 , const Polynomial< 1 , Degree2 > &p2 ){ return p1 + (-p2); }

template< unsigned int Degree >
std::ostream &operator << ( std::ostream &stream , const Polynomial< 1 , Degree > &poly )
{
	static const unsigned int Dim = 1;
	auto PrintCoefficient = [&]( double c , bool first )
	{
		if( c<0 ) stream << " - ";
		else if( c>0 && !first ) stream << " + ";
		stream << fabs( c );
	};

	auto PrintMonomial = [&]( unsigned int d )
	{
		if     ( d==0 ) ;
		else if( d==1 ) stream << " * x_" << Dim;
		else            stream << " * x_" << Dim << "^" << d;
	};

	bool first = true;
	if( poly._isZero( Degree ) ) stream << "0";
	for( int d=0 ; d<=Degree ; d++ )
	{
		if( poly._coefficients[d] )
		{
			PrintCoefficient( poly._coefficients[d] , first );
			PrintMonomial( d );
			first = false;
		}
	}
	return stream;
}

template< unsigned int Degree >
SquareMatrix< double , Degree+1 > Polynomial< 1 , Degree >::EvaluationMatrix( const double positions[Degree+1] )
{
	SquareMatrix< double , Degree+1 > E;
	for( unsigned int i=0 ; i<=Degree ; i++ ) for( unsigned int j=0 ; j<=Degree ; j++ ) E(i,j) = pow( positions[j] , i );
	return E;
}

///////////////////////////////
// Polynomial Dim-dimensions //
///////////////////////////////
template< unsigned int Dim , unsigned int Degree > Polynomial< Dim , Degree >::Polynomial( void ){}

template< unsigned int Dim , unsigned int Degree > Polynomial< Dim , Degree >::Polynomial( double c ){ _polynomials[0] = Polynomial< Dim-1 , Degree >( c ); }

template< unsigned int Dim , unsigned int Degree >
template< unsigned int _Degree >
Polynomial< Dim , Degree >::Polynomial( const Polynomial< Dim , _Degree > &p )
{
	for( int d=0 ; d<=Degree && d<=_Degree ; d++ ) _polynomials[d] = p._polynomials[d];
	for( int d=_Degree+1 ; d<=Degree ; d++ ) _polynomials[d] = Polynomial< Dim-1 , Degree >();
}

template< unsigned int Dim , unsigned int Degree >
template< unsigned int _Degree >
Polynomial< Dim , Degree > &Polynomial< Dim , Degree >::operator= ( const Polynomial< Dim , _Degree > &p )
{
	for( int d=0 ; d<=Degree && d<=_Degree ; d++ ) _polynomials[d] = p._polynomials[d];
	for( int d=_Degree+1 ; d<=Degree ; d++ ) _polynomials[d] = Polynomial< Dim-1 , Degree >();
	return *this;
}

template< unsigned int Dim , unsigned int Degree >
const double &Polynomial< Dim , Degree >::_coefficient( const unsigned int indices[] , unsigned int maxDegree ) const
{
	if( indices[0]>maxDegree ) ERROR_OUT( "degree out of bounds: %d > %d\n" , indices[0] , maxDegree );
	return _polynomials[ indices[0] ]._coefficient( indices+1 , maxDegree-indices[0] );
}

template< unsigned int Dim , unsigned int Degree >
double& Polynomial< Dim , Degree >::_coefficient( const unsigned int indices[] , unsigned int maxDegree )
{
	if( indices[0]>maxDegree ) ERROR_OUT( "degree out of bounds: %d > %d\n" , indices[0] , maxDegree );
	return _polynomials[ indices[0] ]._coefficient( indices+1 , maxDegree-indices[0] );
}

template< unsigned int Dim , unsigned int Degree >
double Polynomial< Dim , Degree >::_evaluate( const double coordinates[] , unsigned int maxDegree ) const
{
	double sum = 0 , tmp = 1;
	for( unsigned int d=0 ; d<=maxDegree ; d++ )
	{
		sum += _polynomials[d]._evaluate( coordinates+1 , maxDegree-d ) * tmp;
		tmp *= coordinates[0];
	}
	return sum;
}

template< unsigned int Dim , unsigned int Degree >
Polynomial< 1 , Degree > Polynomial< Dim , Degree >::_evaluate( const Ray< double , Dim > &ray , unsigned int maxDegree ) const
{
	Polynomial< 1 , 1 > _p = Polynomial< 1 , 1 >( ray.position[0] , ray.direction[0] );
	Ray< double , Dim-1 > _ray;
	for( int d=1 ; d<Dim ; d++ ) _ray.position[d-1] = ray.position[d] , _ray.direction[d-1] = ray.direction[d];

	Polynomial< 1 , Degree > p( 0. ) , __p( 1. );
	for( unsigned int d=0 ; d<=maxDegree ; d++ )
	{
		p += _polynomials[d]._evaluate( _ray , maxDegree-d ) * __p;
		__p = __p * _p;
	}
	return p;
}

template< unsigned int Dim , unsigned int Degree >
bool Polynomial< Dim , Degree >::_isZero( unsigned int maxDegree ) const
{
	for( unsigned int d=0 ; d<=maxDegree ; d++ ) if( !_polynomials[d]._isZero( maxDegree-d ) ) return false;
	return true;
}

template< unsigned int Dim , unsigned int Degree >
bool Polynomial< Dim , Degree >::_isConstant( unsigned int maxDegree ) const
{
	if( !_polynomials[0]._isConstant( Degree ) ) return false;
	for( unsigned int d=1 ; d<=maxDegree ; d++ ) if( !_polynomials[d]._isZero( maxDegree-d ) ) return false;
	return true;
}

template< unsigned int Dim , unsigned int Degree >
template< typename ... UnsignedInts >
const double &Polynomial< Dim , Degree >::coefficient( UnsignedInts ... indices ) const
{
	static_assert( sizeof...(indices)==Dim  , "[ERROR] Polynomial< Dim , Degree >::coefficient: Invalid number of indices" );
	unsigned int _indices[] = { indices ... };
	return _coefficient( _indices , Degree );
}

template< unsigned int Dim , unsigned int Degree >
template< typename ... UnsignedInts >
double &Polynomial< Dim , Degree >::coefficient( UnsignedInts ... indices )
{
	static_assert( sizeof...(indices)==Dim , "[ERROR] Polynomial< Dim , Degree >::coefficient: Invalid number of indices" );
	unsigned int _indices[] = { indices ... };
	return _coefficient( _indices , Degree );
}

template< unsigned int Dim , unsigned int Degree >
template< typename ... Doubles >
double Polynomial< Dim , Degree >::operator()( Doubles ... coordinates ) const
{
	static_assert( sizeof...(coordinates)==Dim , "[ERROR] Polynomial< Dim , Degree >::operator(): Invalid number of coordinates" );
	double _coordinates[] = { coordinates... };
	return _evaluate( _coordinates , Degree );
}

template< unsigned int Dim , unsigned int Degree >
double Polynomial< Dim , Degree >::operator()( Point< double , Dim > p ) const { return _evaluate( &p[0] , Degree ); }

/** This method returns the partial derivative with respect to the prescribed dimension.*/
template< unsigned int Dim , unsigned int Degree >
Polynomial< Dim , Degree-1 > Polynomial< Dim , Degree >::d( int dim ) const
{
	Polynomial< Dim , Degree-1 > derivative;
	if( dim==0 ) for( int d=0 ; d<Degree ; d++ ) derivative._polynomials[d] = _polynomials[d+1] * (d+1);
	else         for( int d=0 ; d<Degree ; d++ ) derivative._polynomials[d] = _polynomials[d].d( dim-1 );
	return derivative;
}

template< unsigned int Dim , unsigned int Degree >
Polynomial< 1 , Degree > Polynomial< Dim , Degree >::operator()( const Ray< double , Dim > &ray ) const { return _evaluate( ray , Degree ); }

template< unsigned int Dim , unsigned int Degree >
std::ostream &operator << ( std::ostream &stream , const Polynomial< Dim , Degree > &poly )
{
	auto PrintSign = [&]( bool first )
	{
		if( !first ) stream << " + ";
	};

	auto PrintMonomial = [&]( unsigned int d )
	{
		if     ( d==0 ) ;
		else if( d==1 ) stream << " * x_" << Dim;
		else            stream << " * x_" << Dim << "^" << d;
	};

	bool first = true;
	if( poly._isZero( Degree ) ) stream << "0";
	for( int d=0 ; d<=Degree ; d++ )
	{
		if( !poly._polynomials[d]._isZero( Degree-d ) )
		{
			PrintSign( first );
			if( poly._polynomials[d]._isConstant( Degree-d ) ) stream <<         poly._polynomials[d]        ;
			else                                               stream << "( " << poly._polynomials[d] << " )";
			PrintMonomial( d );
			first = false;
		}
	}
	return stream;
}

template< unsigned int Dim , unsigned int Degree >
void Polynomial< Dim , Degree >::Scale( double s )
{
	for( int d=0 ; d<=Degree ; d++ ) _polynomials[d] *= s;
}

template< unsigned int Dim , unsigned int Degree >
void Polynomial< Dim , Degree >::Add( const Polynomial< Dim , Degree > &p )
{
	for( int d=0 ; d<=Degree ; d++ ) _polynomials[d] += p._polynomials[d];
}

template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 >
Polynomial< Dim , Degree1 + Degree2 > operator * ( const Polynomial< Dim , Degree1 > &p1 , const Polynomial< Dim , Degree2 > &p2 )
{
	Polynomial< Dim , Degree1 + Degree2 > p;
	for( int d1=0 ; d1<=Degree1 ; d1++ ) for( int d2=0 ; d2<=Degree2 ; d2++ ) p._polynomials[ d1+d2 ]  += p1._polynomials[d1] * p2._polynomials[d2];
	return p;
}

template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 >
Polynomial< Dim , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< Dim , Degree1 > &p1 , const Polynomial< Dim , Degree2 > &p2 )
{
	Polynomial< Dim , Max< Degree1 , Degree2 >::Value > p;
	for( int d=0 ; d<=Degree1 ; d++ ) p._polynomials[d] += p1._polynomials[d];
	for( int d=0 ; d<=Degree2 ; d++ ) p._polynomials[d] += p2._polynomials[d];
	return p;
}

template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 >
Polynomial< Dim , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< Dim , Degree1 > &p1 , const Polynomial< Dim , Degree2 > &p2 ){ return p1 + (-p2); }
