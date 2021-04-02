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

#ifndef POLYNOMIAL_INCLUDED
#define POLYNOMIAL_INCLUDED
#include <iostream>
#include "Geometry.h"
#include "Algebra.h"
#include "Exceptions.h"


/** Helper functionality for computing the minimum of two integers.*/
template< unsigned int D1 , unsigned int D2 > struct Min{ static const unsigned int Value = D1<D2 ? D1 : D2; };
/** Helper functionality for computing the maximum of two integers.*/
template< unsigned int D1 , unsigned int D2 > struct Max{ static const unsigned int Value = D1>D2 ? D1 : D2; };

/** The generic, recursively defined, Polynomial class of total degree Degree. */
template< unsigned int Dim , unsigned int Degree >
class Polynomial : public VectorSpace< double , Polynomial< Dim , Degree > >
{
	template< unsigned int _Dim , unsigned int _Degree > friend class Polynomial;
	template< unsigned int _Dim , unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< _Dim , Degree1 + Degree2 > operator * ( const Polynomial< _Dim , Degree1 > & , const Polynomial< _Dim , Degree2 > & );
	template< unsigned int _Dim , unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< _Dim , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< _Dim , Degree1 > & , const Polynomial< _Dim , Degree2 > & );
	template< unsigned int _Dim , unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< _Dim , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< _Dim , Degree1 > & , const Polynomial< _Dim , Degree2 > & );
	template< unsigned int _Dim , unsigned int _Degree > friend std::ostream &operator << ( std::ostream & , const Polynomial< _Dim , _Degree > & );

	/** The polynomials in Dim-1 dimensions.
	*** The total polynomial is assumed to be _polynomials[0] + _polynomials[1] * (x_Dim) + _polynomials[2] * (x_Dim)^2 + ... */
	Polynomial< Dim-1 , Degree > _polynomials[Degree+1];

	/** This method returns the specified coefficient of the polynomial.*/
	const double &_coefficient( const unsigned int indices[] , unsigned int maxDegree ) const;

	/** This method returns the specified coefficient of the polynomial.*/
	double &_coefficient( const unsigned int indices[] , unsigned int maxDegree );

	/** This method evaluates the polynomial at the specified set of coordinates.*/
	double _evaluate( const double coordinates[] , unsigned int maxDegree ) const;

	/** This method evaluates a Dim-dimensional polynomial along a Dim-dimensional Ray, and returns the associated 1-dimensional polynomial. */
	Polynomial< 1, Degree > _evaluate( const Ray< double , Dim > &ray , unsigned int maxDegree ) const;

	/** This method returns true if the polynomial is zero. */
	bool _isZero( unsigned int maxDegree ) const;

	/** This method returns true if the polynomial is a constant. */
	bool _isConstant( unsigned int maxDegree ) const;

public:
	/** The default constructor initializes the coefficients to zero.*/
	Polynomial( void );

	/** This constructor creates a constant polynomial */
	Polynomial( double c );

	/** The constructor copies over as much of the polynomial as will fit.*/
	template< unsigned int _Degree >
	Polynomial( const Polynomial< Dim , _Degree > &p );

	/** The equality operator copies over as much of the polynomial as will fit.*/
	template< unsigned int _Degree >
	Polynomial &operator= ( const Polynomial< Dim , _Degree > &p );

	/** This method returns the associated coefficient of the polynomial */
	template< typename ... UnsignedInts >
	const double &coefficient( UnsignedInts ... indices ) const;

	/** This method returns the associated coefficient of the polynomial */
	template< typename ... UnsignedInts >
	double &coefficient( UnsignedInts ... indices );

	/** This method evaluates the polynomial at the prescribed point.*/
	template< typename ... Doubles >
	double operator()( Doubles ... coordinates ) const;

	/** This method evaluates the polynomial at the prescribed point.*/
	double operator()( Point< double , Dim > p ) const;

	/** This method returns the partial derivative with respect to the prescribed dimension.*/
	Polynomial< Dim , Degree-1 > d( int dim ) const;

	/** This method returns the 1D polynomial obtained by evaluating the polynomial along the ray.*/
	Polynomial< 1 , Degree > operator()( const Ray< double , Dim > &ray ) const;

	/////////////////////////
	// VectorSpace methods //
	/////////////////////////
	/** This method scales the polynomial */
	void Scale( double s );

	/** This method adds in the polynomial */
	void Add( const Polynomial & p );
};

/** This function returns the product of two polynomials.*/
template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 >
Polynomial< Dim , Degree1 + Degree2 > operator * ( const Polynomial< Dim , Degree1 > &p1 , const Polynomial< Dim , Degree2 > &p2 );

/** This function returns the sum of two polynomials. */
template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 >
Polynomial< Dim , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< Dim , Degree1 > &p1 , const Polynomial< Dim , Degree2 > &p2 );

/** This function returns the difference of two polynomials. */
template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 >
Polynomial< Dim , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< Dim , Degree1 > &p1 , const Polynomial< Dim , Degree2 > &p2 );

/** This function prints out the polynomial.*/
template< unsigned int Dim , unsigned int Degree >
std::ostream &operator << ( std::ostream &stream , const Polynomial< Dim , Degree > &poly );

/** A specialized instance of the Polynomial class in one variable */
template< unsigned int Degree >
class Polynomial< 1 , Degree > : public VectorSpace< double , Polynomial< 1 , Degree > >
{
	template< unsigned int _Dim , unsigned int _Degree > friend class Polynomial;
	template< unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< 1 , Degree1 + Degree2 > operator * ( const Polynomial< 1 , Degree1 > & , const Polynomial< 1 , Degree2 > & );
	template< unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< 1 , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< 1 , Degree1 > & , const Polynomial< 1 , Degree2 > & );
	template< unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< 1 , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< 1 , Degree1 > & , const Polynomial< 1 , Degree2 > & );
	template< unsigned int _Degree > friend std::ostream &operator << ( std::ostream & , const Polynomial< 1 , _Degree > & );
	template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< Dim , Degree1 + Degree2 > operator * ( const Polynomial< Dim , Degree1 > & , const Polynomial< Dim , Degree2 > & );
	template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< Dim , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< Dim , Degree1 > & , const Polynomial< Dim , Degree2 > & );
	template< unsigned int Dim , unsigned int Degree1 , unsigned int Degree2 > friend Polynomial< Dim , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< Dim , Degree1 > & , const Polynomial< Dim , Degree2 > & );
	template< unsigned int Dim , unsigned int _Degree > friend std::ostream &operator << ( std::ostream & , const Polynomial< Dim , _Degree > & );

	/** The coefficients of the polynomial. */
	double _coefficients[Degree+1];

	/** This method returns the specified coefficient of the polynomial.*/
	const double &_coefficient( const unsigned int indices[] , unsigned int maxDegree ) const;

	/** This method returns the specified coefficient of the polynomial.*/
	double &_coefficient( const unsigned int indices[] , unsigned int maxDegree );

	/** This method evaluates the polynomial at the specified set of coordinates.*/
	double _evaluate( const double coordinates[] , unsigned int maxDegree ) const;

	/** This method evaluates a Dim-dimensional polynomial along a Dim-dimensional Ray, and returns the associated 1-dimensional polynomial. */
	Polynomial _evaluate( const Ray< double , 1 > &ray , unsigned int maxDegree ) const;

	/** This method returns true if the polynomial is zero. */
	bool _isZero( unsigned int maxDegree ) const;

	/** This method returns true if the polynomial is a constant. */
	bool _isConstant( unsigned int maxDegree ) const;
public:
	/** The default constructor initializes the coefficients to zero.*/
	Polynomial( void );

	/** This constructor creates a constant polynomial */
	Polynomial( double c );

	/** This constructor initializes the coefficients (starting with lower degrees).
	* If higher degree coefficients are not provided, they are assumed to be zero.*/
	template< typename ... Doubles >
	Polynomial( Doubles ... coefficients );

	/** The constructor copies over as much of the polynomial as will fit.*/
	template< unsigned int _Degree >
	Polynomial( const Polynomial< 1 , _Degree > &p );

	/** The equality operator copies over as much of the polynomial as will fit.*/
	template< unsigned int _Degree >
	Polynomial &operator= ( const Polynomial< 1 , _Degree > &p );

	/** This method returns the d-th coefficient.*/
	const double &operator[]( unsigned int d ) const;

	/** This method returns the d-th coefficient.*/
	double &operator[]( unsigned int d );

	/** This method returns the d-th coefficient.*/
	const double &coefficient( unsigned int d ) const;

	/** This method returns the d-th coefficient.*/
	double &coefficient( unsigned int d );

	/** This method evaluates the polynomial at a given value.*/
	double operator()( double x ) const;

	/** This method returns the derivative of the polynomial.*/
	Polynomial< 1 , Degree-1 > d( unsigned int d=0 ) const;

	/** This method returns the 1D polynomial obtained by evaluating the polynomial along the 1D ray.*/
	Polynomial operator()( const Ray< double , 1 > &ray ) const;

	static SquareMatrix< double , Degree+1 > EvaluationMatrix( const double positions[Degree+1] );

	/////////////////////////
	// VectorSpace methods //
	/////////////////////////
	/** This method scales the polynomial*/
	void Scale( double s );

	/** This method adds in the polynomial */
	void Add( const Polynomial & p );
};

/** This function returns the product of two polynomials.*/
template< unsigned int Degree1 , unsigned int Degree2 >
Polynomial< 1 , Degree1 + Degree2 > operator * ( const Polynomial< 1 , Degree1 > &p1 , const Polynomial< 1 , Degree2 > &p2 );

/** This function returns the sum of two polynomials. */
template< unsigned int Degree1 , unsigned int Degree2 >
Polynomial< 1 , Max< Degree1 , Degree2 >::Value > operator + ( const Polynomial< 1 , Degree1 > &p1 , const Polynomial< 1 , Degree2 > &p2 );

/** This function returns the difference of two polynomials. */
template< unsigned int Degree1 , unsigned int Degree2 >
Polynomial< 1 , Max< Degree1 , Degree2 >::Value > operator - ( const Polynomial< 1 , Degree1 > &p1 , const Polynomial< 1 , Degree2 > &p2 );

/** This function prints out the polynomial.*/
template< unsigned int Degree >
std::ostream &operator << ( std::ostream &stream , const Polynomial< 1 , Degree > &poly );

////////////////////////////////////////////////
// Classes specialized for 1D, 2D, 3D, and 4D //
////////////////////////////////////////////////
/** A polynomial in one variable of degree Degree */
template< unsigned int Degree >
using Polynomial1D = Polynomial< 1 , Degree >;

/** A polynomial in two variables of degree Degree */
template< unsigned int Degree >
using Polynomial2D = Polynomial< 2 , Degree >;

/** A polynomial in three variable of degree Degree */
template< unsigned int Degree >
using Polynomial3D = Polynomial< 3 , Degree >;

/** A polynomial in four variable of degree Degree */
template< unsigned int Degree >
using Polynomial4D = Polynomial< 4 , Degree >;

#include "Polynomial.inl"
#endif // POLYNOMIAL_INCLUDED
