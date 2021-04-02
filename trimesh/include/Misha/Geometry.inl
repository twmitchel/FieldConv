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

#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif // M_PI

#ifndef _WIN32
#include <stdlib.h>
#include <string.h>
#endif
#include <float.h>
#include <unordered_map>

inline long long HalfEdgeKey( int i1 , int i2 )
{
	return ( ( (long long) i1 )<<32 ) | ( (long long) i2 );
}
inline long long EdgeKey( int i1 , int i2 )
{
	if( i1>i2 ) return HalfEdgeKey( i1 , i2 );
	else		return HalfEdgeKey( i2 , i1 );
}
inline void FactorEdgeKey( long long key , int& idx1 , int& idx2 )
{
	long long i1 , i2;
	i1 = key>>32;
	i2 = (key<<32)>>32;
	idx1 = int( i1 );
	idx2 = int( i2 );
}
inline long long OppositeHalfEdgeKey( long long key )
{
	int i1 , i2;
	FactorEdgeKey( key , i1 , i2 );
	return HalfEdgeKey( i2 , i1 );
}


///////////
// Point //
///////////
template<class Real,int Dim>
void Point<Real,Dim>::Add		(const Point<Real,Dim>& p)	{	for(int d=0;d<Dim;d++)	coords[d]+=p.coords[d];	}
template<class Real,int Dim>
void Point<Real,Dim>::Scale		(Real s)					{	for(int d=0;d<Dim;d++)	coords[d]*=s;	}
template<class Real,int Dim>
Real Point<Real,Dim>::InnerProduct(const Point<Real,Dim>& p)	const
{
	Real dot=0;
	for(int i=0;i<Dim;i++)	dot+=p.coords[i]*coords[i];
	return dot;
}


#if !FAST_POINT
/////////////
// Point3D //
/////////////
template< class Real >
Point3D<Real> Point3D<Real>::CrossProduct( const Point3D<Real>& p1 , const Point3D<Real> & p2 )
{
	Point3D<Real> p;
	p.coords[0]= p1.coords[1]*p2.coords[2]-p1.coords[2]*p2.coords[1];
	p.coords[1]=-p1.coords[0]*p2.coords[2]+p1.coords[2]*p2.coords[0];
	p.coords[2]= p1.coords[0]*p2.coords[1]-p1.coords[1]*p2.coords[0];
	return p;
}
#endif // FAST_POINT

////////////
// Matrix //
////////////
template<class Real,int Cols,int Rows>
void Matrix<Real,Cols,Rows>::Add(const Matrix<Real,Cols,Rows>& m)
{
	for(int i=0;i<Cols;i++)	for(int j=0;j<Rows;j++)	coords[i][j]+=m.coords[i][j];
}
template<class Real,int Cols,int Rows>
void Matrix<Real,Cols,Rows>::Scale(Real s)
{
	for(int i=0;i<Cols;i++)	for(int j=0;j<Rows;j++)	coords[i][j]*=s;
}
template<class Real,int Cols,int Rows>
Real Matrix<Real,Cols,Rows>::InnerProduct(const Matrix<Real,Cols,Rows>& m) const
{
	Real dot=0;
	for(int i=0;i<Cols;i++)
		for(int j=0;j<Rows;j++)
			dot+=m.coords[i][j]*coords[i][j];
	return dot;
}
template<class Real,int Cols,int Rows>
template<int Cols1>
Matrix<Real,Cols1,Rows> Matrix<Real,Cols,Rows>::operator * (const Matrix<Real,Cols1,Cols>& m) const
{
	Matrix<Real,Cols1,Rows> n;
	for(int i=0;i<Cols1;i++)
		for(int j=0;j<Rows;j++)
			for(int k=0;k<Cols;k++)
				n.coords[i][j]+=m.coords[i][k]*coords[k][j];
	return n;
}
template<class Real,int Cols,int Rows>
template<class Real2>
Point<Real2,Rows> Matrix<Real,Cols,Rows>::operator () (const Point<Real2,Cols>& v) const	{	return (*this)*v;	}
template<class Real,int Cols,int Rows>
template<class Real2>
Point< Real2 , Rows > Matrix< Real , Cols , Rows >::operator * ( const Point< Real2 , Cols >& v ) const
{
	Point< Real2 , Rows > out;
#if 1
	for( int j=0 ; j<Cols ; j++ )
	{
		const Real* _coords = coords[j];
		Real2 _v = v.coords[j];
		for( int i=0 ; i<Rows ; i++ ) out.coords[i] += Real2( _coords[i] ) * _v;
	}
#else
	for(int i=0;i<Rows;i++)
		for(int j=0;j<Cols;j++)
			out.coords[i] += Real2( coords[j][i] ) * v.coords[j];
#endif
	return out;
}
template<class Real,int Cols,int Rows>
Matrix<Real,Rows,Cols> Matrix<Real,Cols,Rows>::transpose(void) const
{
	Matrix<Real,Rows,Cols> out;
	for(int i=0;i<Cols;i++)
		for(int j=0;j<Rows;j++)
			out.coords[j][i]=coords[i][j];
	return out;
}

//////////////////
// SquareMatrix //
//////////////////
template<> inline double SquareMatrix< double , 1 >::determinant( void ) const { return coords[0][0];}
template<> inline double SquareMatrix< double , 2 >::determinant( void ) const { return coords[0][0]*coords[1][1] - coords[0][1]*coords[1][0]; }
template<> inline double SquareMatrix< double , 3 >::determinant( void ) const
{
	return
		coords[0][0]*( coords[1][1]*coords[2][2] - coords[2][1]*coords[1][2] ) +
		coords[1][0]*( coords[2][1]*coords[0][2] - coords[0][1]*coords[2][2] ) +
		coords[2][0]*( coords[0][1]*coords[1][2] - coords[0][2]*coords[1][1] ) ;
}
template<class Real,int Dim>
Real SquareMatrix<Real,Dim>::subDeterminant( int c , int r ) const
{
	SquareMatrix<double,Dim-1> temp;
	for( int i=0 , ii=0 ; i<Dim ; i++ )
	{
		if( i==c ) continue;
		for( int j=0 , jj=0 ; j<Dim ; j++ )
		{
			if( j==r ) continue;
			temp.coords[ii][jj] = coords[i][j];
			jj++;
		}
		ii++;
	}
	return Real( temp.determinant() );
}

template<class Real,int Dim>
Real SquareMatrix<Real,Dim>::determinant( void ) const
{
	Real det = Real(1);
	// Gaussian Elimination
	SquareMatrix xForm , temp;
	xForm = (*this);
	for( int i=0 ; i<Dim ; i++ )
	{
		int p = i ; Real v = (Real)fabs( xForm(i,i) );
		for( int j=i+1 ; j<Dim ; j++ ) if( fabs( xForm(i,j) )>v ) p = j , v = (Real)fabs( xForm(i,j) );
		if( !v ) return Real(0);
		if( i!=p ) det *= -xForm(i,p);
		else       det *=  xForm(i,p);
		temp.SetIdentity();
		Real scl = Real(1)/xForm(i,p);
		for( int j=0 ; j<Dim ; j++ ) temp(p,j) = - xForm(i,j) * scl;
		temp(i,p) = Real(1);
		temp(p,p) = -xForm(i,i) * scl;
		temp(i,i) = Real(0);
		temp(p,i) = scl; // Note that this is last so that if p=i the value is the right one
		xForm = temp * xForm;
	}
	return det;
}
template< class Real , int Dim >
Real SquareMatrix< Real , Dim >::trace( void ) const
{
	Real tr = (Real)0;
	for( int i=0 ; i<Dim ; i++ ) tr += coords[i][i];
	return tr;
}
template< class Real , int Dim >
SquareMatrix< Real , Dim > SquareMatrix< Real , Dim >::inverse( void ) const
{
	bool success;
	return inverse( success );
}
template< class Real , int Dim >
SquareMatrix< Real , Dim > SquareMatrix< Real , Dim >::inverse( bool& success ) const
{
#if 1
	// Gaussian Elimination
	SquareMatrix xForm , iXForm , temp;
	iXForm.SetIdentity() , xForm = (*this);
	for( int i=0 ; i<Dim ; i++ )
	{
		int p = i ; Real v = (Real)fabs( xForm(i,i) );
		for( int j=i+1 ; j<Dim ; j++ ) if( fabs( xForm(i,j) )>v ) p = j , v = (Real)fabs( xForm(i,j) );
		if( !v )
		{
			fprintf( stderr , "[WARNING] Failed to invert matrix\n" );
			success = false;
			return SquareMatrix();
		}
		// temp(i,j): mapping of the i-th row to the j-th row
		temp.SetIdentity();
		Real scl = Real(1)/xForm(i,p);
		for( int j=0 ; j<Dim ; j++ ) temp(p,j) = - xForm(i,j) * scl;
		temp(i,p) = Real(1);
		temp(p,p) = -xForm(i,i) * scl;
		temp(i,i) = Real(0);
		temp(p,i) = scl; // Note that this is last so that if p=i the value is the right one
		xForm = temp * xForm , iXForm = temp * iXForm;
	}
	success = true;
	return iXForm;
#else
	SquareMatrix iXForm;
	Real d=determinant();
	for(int i=0;i<Dim;i++)
		for(int j=0;j<Dim;j++)
			if(((i+j)&1)==0)	iXForm.coords[j][i]= subDeterminant(i,j)/d;
			else				iXForm.coords[i][j]=-subDeterminant(j,i)/d;
	return iXForm;
#endif
}
template< >
SquareMatrix< float , 2 > SquareMatrix< float , 2 >::inverse( bool& success ) const
{
	SquareMatrix iXForm;
	float det = ( coords[0][0]*coords[1][1]-coords[0][1]*coords[1][0] );
	if( !det ) success = false;
	float d = 1.f / det;
	iXForm.coords[0][0] =  coords[1][1] * d;
	iXForm.coords[1][1] =  coords[0][0] * d;
	iXForm.coords[0][1] = -coords[0][1] * d;
	iXForm.coords[1][0] = -coords[1][0] * d;
	success = true;
	return iXForm;
}
template< >
SquareMatrix< double , 2 > SquareMatrix< double , 2 >::inverse( bool& success ) const
{
	SquareMatrix iXForm;
	double det = ( coords[0][0]*coords[1][1]-coords[0][1]*coords[1][0] );
	if( !det ) success = false;
	double d = 1. / det;
	iXForm.coords[0][0] =  coords[1][1] * d;
	iXForm.coords[1][1] =  coords[0][0] * d;
	iXForm.coords[0][1] = -coords[0][1] * d;
	iXForm.coords[1][0] = -coords[1][0] * d;
	success = true;
	return iXForm;
}
template<class Real,int Dim>
void SquareMatrix<Real,Dim>::Multiply (const SquareMatrix<Real,Dim>& m)
{
	SquareMatrix temp=*this;
	for(int i=0;i<Dim;i++)
		for(int j=0;j<Dim;j++)
		{
			this->coords[i][j]=0;
			for(int k=0;k<Dim;k++)	this->coords[i][j]+=temp.coords[k][j]*m.coords[i][k];
		}
}
template<class Real,int Dim>
void SquareMatrix<Real,Dim>::SetIdentity(void)
{
	memset(this->coords,0,sizeof(Real)*Dim*Dim);
	for(int i=0;i<Dim;i++)	this->coords[i][i]=1;
}
template<class Real,int Dim>
template<class Real2>
Point<Real2,Dim-1> SquareMatrix<Real,Dim>::operator () (const Point<Real2,Dim-1>& v) const
{
	Real2 scale=1;
	Point<Real2,Dim-1> out;
	for(int i=0;i<Dim-1;i++)
	{
		for( int j=0 ; j<Dim-1 ; j++ ) out.coords[i] += v.coords[j]*Real2( this->coords[j][i] );
		out.coords[i] += Real2( this->coords[Dim-1][i] );
		scale += Real2( this->coords[i][Dim-1] );
	}
	for(int i=0;i<Dim-1;i++)	out.coords[i]/=scale;
	return out;
}

template<class Real>
//Real Random(void){return Real(rand())/RAND_MAX;}
Real Random( void )
{
	static const unsigned long long ULL_RAND_MAX = ( unsigned long long )( RAND_MAX ) + 1;
	static const double RAND_MAX_SQUARED = double( ULL_RAND_MAX * ULL_RAND_MAX - 1 );
	unsigned long long r1 = rand() , r2 = rand();
	long long foo = r1 * ULL_RAND_MAX + r2;
	return Real( double(foo) / RAND_MAX_SQUARED );
}

template<class Real>
Real Random2( void )
{
	long long temp= (long long) ( rand() )*RAND_MAX+rand();
	return Real( (double(temp)/(RAND_MAX+1))/(RAND_MAX+1) );
}

template<class Real>
Point2D<Real> RandomDiskPoint(void){
	Point2D<Real> p;
	while(1)
	{
		p.coords[0]=Real(1.0-2.0*Random2<Real>());
		p.coords[1]=Real(1.0-2.0*Random2<Real>());
		double l=SquareLength(p);
		if( l<=1 ) return p;
	}
}
template<class Real>
Point3D<Real> RandomBallPoint(void){
	Point3D<Real> p;
	while(1)
	{
		p.coords[0]=Real(1.0-2.0*Random2<Real>());
		p.coords[1]=Real(1.0-2.0*Random2<Real>());
		p.coords[2]=Real(1.0-2.0*Random2<Real>());
		double l=SquareLength(p);
		if( l<=1 ){return p;}
	}
}
template<class Real>
Point2D<Real> RandomCirclePoint(void)
{
	Point2D<Real> p = RandomDiskPoint<Real>();
	Real l = Real(Length(p));
	p.coords[0] /= l;
	p.coords[1] /= l;
	return p;
}
template<class Real>
Point3D<Real> RandomSpherePoint(void)
{
	Point3D<Real> p = RandomBallPoint<Real>();
	Real l = Real(Length(p));
	p.coords[0] /= l;
	p.coords[1] /= l;
	p.coords[2] /= l;
	return p;
}

template<class Real>
XForm3x3<Real> RotationMatrix( const Point3D<Real>& axis , const Real& angle )
{
	double a = cos( angle / 2 );
	double b , c , d;
	Point3D< Real > ax = axis * Real(sin( angle / 2 ) / Length( axis ));
	b = ax[0] , c = ax[1] , d = ax[2];
	return RotationMatrix< Real >( Real( a ) , Real( b ) , Real( c ) , Real( d ) );
}
template<class Real>
XForm3x3<Real> RotationMatrix( Real a , Real b , Real c , Real d )
{
	XForm3x3< Real > rot;
	rot( 0 , 0 ) = 1 - 2*c*c - 2*d*d;
	rot( 1 , 0 ) = 2*b*c - 2*a*d;
	rot( 2 , 0 ) = 2*b*d + 2*a*c;
	rot( 0 , 1 ) = 2*b*c + 2*a*d;
	rot( 1 , 1 ) = 1 - 2*b*b - 2*d*d;
	rot( 2 , 1 ) = 2*c*d - 2*a*b;
	rot( 0 , 2 ) = 2*b*d - 2*a*c;
	rot( 1 , 2 ) = 2*c*d + 2*a*b;
	rot( 2 , 2 ) = 1 - 2*b*b - 2*c*c;
	return rot;
}

template<class Real>
XForm3x3<Real> RandomRotationMatrix( void )
{
	Point3D< Real > axis = RandomSpherePoint< Real > ( );
	Real angle = Real( 2.0 * M_PI * Random< Real > ( ) );
	return RotationMatrix( axis , angle );
}


template<class Real>
double SquareLength(const Point2D<Real>& p){return p.coords[0]*p.coords[0]+p.coords[1]*p.coords[1];}

template<class Real>
double SquareLength(const Point3D<Real>& p){return p.coords[0]*p.coords[0]+p.coords[1]*p.coords[1]+p.coords[2]*p.coords[2];}

template<class Real>
double Length(const Point2D<Real>& p){return sqrt(SquareLength(p));}

template<class Real>
double Length(const Point3D<Real>& p){return sqrt(SquareLength(p));}

template<class Real>
double SquareDistance(const Point3D<Real>& p1,const Point3D<Real>& p2){
	return (p1.coords[0]-p2.coords[0])*(p1.coords[0]-p2.coords[0])+(p1.coords[1]-p2.coords[1])*(p1.coords[1]-p2.coords[1])+(p1.coords[2]-p2.coords[2])*(p1.coords[2]-p2.coords[2]);
}

template<class Real>
double DotProduct(const Point3D<Real>& p1,const Point3D<Real>& p2){
	return p1.coords[0]*p2.coords[0]+p1.coords[1]*p2.coords[1]+p1.coords[2]*p2.coords[2];
}

template<class Real>
double Distance(const Point3D<Real>& p1,const Point3D<Real>& p2){return sqrt(SquareDistance(p1,p2));}

template <class Real>
void CrossProduct(const Point3D<Real>& p1,const Point3D<Real>& p2,Point3D<Real>& p){
	p.coords[0]= p1.coords[1]*p2.coords[2]-p1.coords[2]*p2.coords[1];
	p.coords[1]=-p1.coords[0]*p2.coords[2]+p1.coords[2]*p2.coords[0];
	p.coords[2]= p1.coords[0]*p2.coords[1]-p1.coords[1]*p2.coords[0];
}

template<class Real>
void Transform(const XForm4x4<Real>& xForm,const Point3D<Real>& p,Point3D<Real>& q)
{
	q.coords[0]=xForm.coords[0][0]*p.coords[0]+xForm.coords[1][0]*p.coords[1]+xForm.coords[2][0]*p.coords[2]+xForm.coords[3][0];
	q.coords[1]=xForm.coords[0][1]*p.coords[0]+xForm.coords[1][1]*p.coords[1]+xForm.coords[2][1]*p.coords[2]+xForm.coords[3][1];
	q.coords[2]=xForm.coords[0][2]*p.coords[0]+xForm.coords[1][2]*p.coords[1]+xForm.coords[2][2]*p.coords[2]+xForm.coords[3][2];
	Real scl   =xForm.coords[0][3]*p.coords[0]+xForm.coords[1][3]*p.coords[1]+xForm.coords[2][3]*p.coords[2]+xForm.coords[3][3];
	q.coords[0]/=scl;
	q.coords[1]/=scl;
	q.coords[2]/=scl;
}
template<class Real>
void TransformNoTranslate(const XForm4x4<Real>& xForm,const Point3D<Real>& p,Point3D<Real>& q)
{
	q.coords[0]=xForm.coords[0][0]*p.coords[0]+xForm.coords[1][0]*p.coords[1]+xForm.coords[2][0]*p.coords[2];
	q.coords[1]=xForm.coords[0][1]*p.coords[0]+xForm.coords[1][1]*p.coords[1]+xForm.coords[2][1]*p.coords[2];
	q.coords[2]=xForm.coords[0][2]*p.coords[0]+xForm.coords[1][2]*p.coords[1]+xForm.coords[2][2]*p.coords[2];
}

template<class Real>
Real SubDeterminant(const XForm4x4<Real>& xForm,int c1,int r1,int c2,int r2)
{
	return xForm.coords[c1][r1]*xForm.coords[c2][r2]-xForm.coords[c1][r2]*xForm.coords[c2][r1];
}
template<class Real>
Real SubDeterminant(const XForm4x4<Real>& xForm,int c,int r)
{
	int c1,r1,c2,r2,row;
	Real d=0,sgn=1.0;
	row=0;
	if(row==r){row++;}
	for(int i=0;i<4;i++)
	{
		if(i==c){continue;}
		c1=0;
		while(c1==i || c1==c){c1++;}
		c2=c1+1;
		while(c2==i || c2==c){c2++;}
		r1=0;
		while(r1==row || r1==r){r1++;}
		r2=r1+1;
		while(r2==row || r2==r){r2++;}
		
		d+=sgn*xForm.coords[i][row]*SubDeterminant(xForm,c1,r1,c2,r2);
		sgn*=-1.0;
	}
	return d;
}


template<class Real>
Real Determinant(const XForm4x4<Real>& xForm)
{
	Real d=Real(0);
	for(int i=0;i<4;i++)
		if((i&1)==0)	d+=SubDeterminant(xForm,i,0)*xForm.coords[i][0];
		else			d-=SubDeterminant(xForm,i,0)*xForm.coords[i][0];
	return d;
}

#if 0
template<class Real>
void InvertTransform(const XForm4x4<Real>& xForm,XForm4x4<Real>& iXForm)
{
	Real d=Determinant(xForm);
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			if(((i+j)&1)==0)	iXForm.coords[j][i]= SubDeterminant(xForm,i,j)/d;
			else				iXForm.coords[j][i]=-SubDeterminant(xForm,j,i)/d;
	return m;
}
#endif

//////////////////////////////
// MinimalAreaTriangulation //
//////////////////////////////
#if 1
template< typename Real , unsigned int Dim >
double MinimalAreaTriangulation< Real , Dim >::_Area( Point< Real , Dim > v0 , Point< Real , Dim > v1 , Point< Real , Dim > v2 )
{
	Point< Real , Dim > d[] = { v1-v0 , v2-v0 };
	SquareMatrix< double , 2 > M;
	for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) M(i,j) = Point< Real , Dim >::Dot( d[i] , d[j] );
	return sqrt( std::max< double >( 0 , M.determinant() ) ) / 2.;
}
template< class Real , unsigned int Dim >
MinimalAreaTriangulation< Real , Dim >::MinimalAreaTriangulation( void ) : _bestTriangulation(NULL) , _midPoint(NULL) {}
template< class Real , unsigned int Dim >
MinimalAreaTriangulation< Real , Dim >::~MinimalAreaTriangulation( void )
{
	if( _bestTriangulation ) delete[] _bestTriangulation;
	if( _midPoint ) delete[] _midPoint;
	_bestTriangulation=NULL;
	_midPoint=NULL;
}
template< class Real , unsigned int Dim >
template< typename Index >
void MinimalAreaTriangulation< Real , Dim >::GetTriangulation( const std::vector< Point< Real , Dim > >& vertices , std::vector< SimplexIndex< 2 , Index > >& triangles )
{
	triangles.resize( vertices.size() - 2 );
	if( vertices.size()==3 )
	{
		triangles[0][0]=0;
		triangles[0][1]=1;
		triangles[0][2]=2;
		return;
	}
	else if( vertices.size()==4 )
	{
		SimplexIndex< 2 , Index > tIndex[2][2];
		double area[] = { 0 , 0 };

		tIndex[0][0][0]=0;
		tIndex[0][0][1]=1;
		tIndex[0][0][2]=2;
		tIndex[0][1][0]=2;
		tIndex[0][1][1]=3;
		tIndex[0][1][2]=0;

		tIndex[1][0][0]=0;
		tIndex[1][0][1]=1;
		tIndex[1][0][2]=3;
		tIndex[1][1][0]=3;
		tIndex[1][1][1]=1;
		tIndex[1][1][2]=2;

		for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) area[i] += _Area( vertices[ tIndex[i][j][0] ] , vertices[ tIndex[i][j][1] ] , vertices[ tIndex[i][j][2] ] );

		if( area[0]>area[1] ) triangles[0] = tIndex[1][0] , triangles[1] = tIndex[1][1];
		else                  triangles[0] = tIndex[0][0] , triangles[1] = tIndex[0][1];
		return;
	}

	if( _bestTriangulation ) delete[] _bestTriangulation;
	if( _midPoint ) delete[] _midPoint;
	_bestTriangulation = NULL;
	_midPoint = NULL;
	size_t eCount=vertices.size();
	_bestTriangulation = new double[eCount*eCount];
	_midPoint = new size_t[eCount*eCount];
	for( unsigned int i=0 ; i<eCount*eCount ; i++ ) _bestTriangulation[i] = -1 , _midPoint[i] = -1;
	_GetArea( 0 , 1 , vertices );
	//	triangles.clear();
	size_t idx = 0;
	//	GetTriangulation(0,1,vertices,triangles);
	_GetTriangulation( 0 , 1 , vertices , triangles , idx );
}

template< class Real , unsigned int Dim >
double MinimalAreaTriangulation< Real , Dim >::GetArea( const std::vector< Point< Real , Dim > > &vertices)
{
	if( _bestTriangulation ) delete[] _bestTriangulation;
	if( _midPoint ) delete[] _midPoint;
	_bestTriangulation = NULL;
	_midPoint = NULL;
	size_t eCount = vertices.size();
	_bestTriangulation = new double[eCount*eCount];
	_midPoint = new size_t[eCount*eCount];
	for( int i=0 ; i<eCount*eCount ; i++ ) _bestTriangulation[i]=-1 , _midPoint[i] = -1;
	return _GetArea( 0 , 1 , vertices );
}
template< typename Real , unsigned int Dim >
template< typename Index >
void MinimalAreaTriangulation< Real , Dim >::_GetTriangulation( size_t i , size_t j , const std::vector< Point< Real , Dim > > &vertices , std::vector< SimplexIndex< 2 , Index > > &triangles , size_t &idx )
{
	SimplexIndex< 2 , Index > tIndex;
	size_t eCount = vertices.size();
	size_t ii = i;
	if( i<j ) ii += eCount;
	if( j+1>=ii ) return;
	ii = _midPoint[i*eCount+j];
	if( ii>=0 )
	{
		tIndex[0] = (Index)i;
		tIndex[1] = (Index)j;
		tIndex[2] = (Index)ii;
		triangles[idx++] = tIndex;
		_GetTriangulation( i , ii , vertices , triangles , idx );
		_GetTriangulation( ii , j , vertices , triangles , idx );
	}
}

template< class Real , unsigned int Dim >
double MinimalAreaTriangulation< Real , Dim >::_GetArea( size_t i , size_t j , const std::vector< Point< Real , Dim > > &vertices)
{
	double a=FLT_MAX , temp;
	size_t eCount = vertices.size();
	size_t idx = i*eCount+j;
	size_t ii = i;
	if( i<j ) ii += eCount;
	if( j+1>=ii)
	{
		_bestTriangulation[idx]=0;
		return 0;
	}
	size_t mid=-1;
	for( size_t r=j+1 ; r<ii ; r++ )
	{
		size_t rr=r%eCount;
		size_t idx1=i*eCount+rr,idx2=rr*eCount+j;
		temp = _Area( vertices[rr] , vertices[i] , vertices[j] );

		if( _bestTriangulation[idx1]>0 )
		{
			temp += _bestTriangulation[idx1];
			if( temp>a ) continue;
			if( _bestTriangulation[idx2]>0 ) temp += _bestTriangulation[idx2];
			else temp += _GetArea( rr , j , vertices );
		}
		else
		{
			if( _bestTriangulation[idx2]>0 ) temp += _bestTriangulation[idx2];
			else temp += _GetArea( rr , j , vertices );
			if( temp>a ) continue;
			temp += _GetArea( i , rr , vertices );
		}

		if( temp<a ) a = temp , mid = rr;
	}
	_bestTriangulation[idx]=a;
	_midPoint[idx]=mid;

	return a;
}
#else
template <class Real>
MinimalAreaTriangulation<Real>::MinimalAreaTriangulation(void)
{
	bestTriangulation=NULL;
	midPoint=NULL;
}
template <class Real>
MinimalAreaTriangulation<Real>::~MinimalAreaTriangulation(void)
{
	if(bestTriangulation)
		delete[] bestTriangulation;
	bestTriangulation=NULL;
	if(midPoint)
		delete[] midPoint;
	midPoint=NULL;
}
template <class Real>
void MinimalAreaTriangulation<Real>::GetTriangulation( const std::vector<Point3D<Real> >& vertices , std::vector<TriangleIndex>& triangles )
{
	triangles.resize( vertices.size() - 2 );
	if( vertices.size()==3 )
	{
		triangles[0][0]=0;
		triangles[0][1]=1;
		triangles[0][2]=2;
		return;
	}
	else if( vertices.size()==4 )
	{
		TriangleIndex tIndex[2][2];
		Real area[2];

		area[0]=area[1]=0;

		tIndex[0][0][0]=0;
		tIndex[0][0][1]=1;
		tIndex[0][0][2]=2;
		tIndex[0][1][0]=2;
		tIndex[0][1][1]=3;
		tIndex[0][1][2]=0;

		tIndex[1][0][0]=0;
		tIndex[1][0][1]=1;
		tIndex[1][0][2]=3;
		tIndex[1][1][0]=3;
		tIndex[1][1][1]=1;
		tIndex[1][1][2]=2;

		Point3D<Real> n,p1,p2;
		for(int i=0;i<2;i++)
			for(int j=0;j<2;j++)
				for(int k=0;k<3;k++)
				{
					p1.coords[k]=vertices[tIndex[i][j][1]].coords[k]-vertices[tIndex[i][j][0]].coords[k];
					p2.coords[k]=vertices[tIndex[i][j][2]].coords[k]-vertices[tIndex[i][j][0]].coords[k];
					CrossProduct(p1,p2,n);
					area[i] += Real( Length(n) );
				}
		if(area[0]>area[1])
		{
			triangles[0]=tIndex[1][0];
			triangles[1]=tIndex[1][1];
		}
		else
		{
			triangles[0]=tIndex[0][0];
			triangles[1]=tIndex[0][1];
		}
		return;
	}

	if(bestTriangulation) delete[] bestTriangulation;
	if(midPoint) delete[] midPoint;
	bestTriangulation=NULL;
	midPoint=NULL;
	size_t eCount=vertices.size();
	bestTriangulation=new double[eCount*eCount];
	midPoint=new int[eCount*eCount];
	for (unsigned int i = 0; i < eCount * eCount; i++)
        bestTriangulation[i] = -1;
	memset(midPoint,-1,sizeof(int)*eCount*eCount);
	GetArea(0,1,vertices);
//	triangles.clear();
	int idx = 0;
//	GetTriangulation(0,1,vertices,triangles);
	GetTriangulation( 0 , 1 , vertices , triangles , idx );
}
template <class Real>
double MinimalAreaTriangulation<Real>::GetArea(const std::vector<Point3D<Real> >& vertices)
{
	if(bestTriangulation)
		delete[] bestTriangulation;
	if(midPoint)
		delete[] midPoint;
	bestTriangulation=NULL;
	midPoint=NULL;
	size_t eCount=vertices.size();
	bestTriangulation=new double[eCount*eCount];
	midPoint=new int[eCount*eCount];
	for(int i=0;i<eCount*eCount;i++)
		bestTriangulation[i]=-1;
	memset(midPoint,-1,sizeof(int)*eCount*eCount);
	return GetArea(0,1,vertices);
}
template<class Real>
void MinimalAreaTriangulation<Real>::GetTriangulation( const int& i , const int& j , const std::vector<Point3D<Real> >& vertices , std::vector<TriangleIndex>& triangles , int& idx )
{
	TriangleIndex tIndex;
	size_t eCount=vertices.size();
	int ii=i;
	if( i<j ) ii+=(int)eCount;
	if( j+1>=ii ) return;
	ii=midPoint[i*eCount+j];
	if(ii>=0)
	{
		tIndex[0]=i;
		tIndex[1]=j;
		tIndex[2]=ii;
		triangles[idx++] = tIndex;
		GetTriangulation( i , ii , vertices , triangles , idx );
		GetTriangulation( ii , j , vertices , triangles , idx );
	}
}

template<class Real>
double MinimalAreaTriangulation<Real>::GetArea(const int& i,const int& j,const std::vector<Point3D<Real> >& vertices)
{
	double a=FLT_MAX,temp;
	size_t eCount=vertices.size();
	size_t idx=i*eCount+j;
	size_t ii=i;
	if(i<j) ii+=eCount;
	if(j+1>=(int) ii)
	{
		bestTriangulation[idx]=0;
		return 0;
	}
	int mid=-1;
	for(unsigned int r=j+1;r<ii;r++)
	{
		int rr=r%eCount;
		size_t idx1=i*eCount+rr,idx2=rr*eCount+j;
		Point3D<Real> p,p1,p2;
		for(int k=0;k<3;k++)
		{
			p1.coords[k]=vertices[i].coords[k]-vertices[rr].coords[k];
			p2.coords[k]=vertices[j].coords[k]-vertices[rr].coords[k];
		}
		CrossProduct(p1,p2,p);
		temp=Length(p);

		if(bestTriangulation[idx1]>0)
		{
			temp+=bestTriangulation[idx1];
			if(temp>a)
				continue;
			if(bestTriangulation[idx2]>0)
				temp+=bestTriangulation[idx2];
			else
				temp+=GetArea(rr,j,vertices);
		}
		else
		{
			if(bestTriangulation[idx2]>0)
				temp+=bestTriangulation[idx2];
			else
				temp+=GetArea(rr,j,vertices);
			if(temp>a)
				continue;
			temp+=GetArea(i,rr,vertices);
		}

		if(temp<a)
		{
			a=temp;
			mid=rr;
		}
	}
	bestTriangulation[idx]=a;
	midPoint[idx]=mid;

	return a;
}
#endif

template< class Real >
void SplitTriangle( std::vector< Point3D< Real > >& vertices , TriangleIndex triangle , Point3D< Real > pNormal , Real pOffset ,
				    std::vector< TriangleIndex >& backTriangles , std::vector< TriangleIndex >& frontTriangles )
{
	int bVerts[4] , fVerts[4] , bCount = 0 , fCount = 0;
	Real values[3];
	for( int i=0 ; i<3 ; i++ ) values[i] = Point3D< Real >::Dot( vertices[ triangle[i] ] , pNormal ) - pOffset;
	for( int i=0 ; i<3 ; i++ )
	{
		int i1 = (i+2)%3;
		int i2 = (i+1)%3;
		if( values[i]*values[i1]<0 )
		{
			Real t = values[i] / ( values[i] - values[i1] );
			Point3D< Real > newVert = vertices[ triangle[i1] ]*t + vertices[ triangle[ i ] ]*Real(1.0-t);
			vertices.push_back( newVert );
			bVerts[ bCount++ ] = vertices.size()-1;
			fVerts[ fCount++ ] = vertices.size()-1;
		}
		if( values[i]==0 )
		{
			if( values[i1]<0 || values[i2]<0 ) bVerts[ bCount++ ] = triangle[i];
			if( values[i1]>0 || values[i2]>0 ) fVerts[ fCount++ ] = triangle[i];
		}
		else
			if( values[i]<0 ) bVerts[ bCount++ ] = triangle[i];
			else			  fVerts[ fCount++ ] = triangle[i];
	}
	if( bCount==3 )
	{
		backTriangles.resize( 1 );
		backTriangles[0][0] = bVerts[0];
		backTriangles[0][1] = bVerts[1];
		backTriangles[0][2] = bVerts[2];
	}
	if( bCount==4 )
	{
		backTriangles.resize( 2 );
		backTriangles[0][0] = bVerts[0];
		backTriangles[0][1] = bVerts[1];
		backTriangles[0][2] = bVerts[2];
		backTriangles[1][0] = bVerts[2];
		backTriangles[1][1] = bVerts[3];
		backTriangles[1][2] = bVerts[0];
	}
	if( fCount==3 )
	{
		frontTriangles.resize( 1 );
		frontTriangles[0][0] = fVerts[0];
		frontTriangles[0][1] = fVerts[1];
		frontTriangles[0][2] = fVerts[2];
	}
	if( fCount==4 )
	{
		frontTriangles.resize( 2 );
		frontTriangles[0][0] = fVerts[0];
		frontTriangles[0][1] = fVerts[1];
		frontTriangles[0][2] = fVerts[2];
		frontTriangles[1][0] = fVerts[2];
		frontTriangles[1][1] = fVerts[3];
		frontTriangles[1][2] = fVerts[0];
	}
}
template< class Real >
void SplitTriangle( std::vector< Point3D< Real > >& vertices , TriangleIndex triangle , int direction , Real offset ,
				    std::vector< TriangleIndex >& backTriangles , std::vector< TriangleIndex >& frontTriangles )
{
	int bVerts[4] , fVerts[4] , bCount = 0 , fCount = 0;
	Real values[3];
	for( int i=0 ; i<3 ; i++ ) values[i] = vertices[ triangle[i] ][ direction ] - offset;
	for( int i=0 ; i<3 ; i++ )
	{
		int i1 = (i+2)%3;
		int i2 = (i+1)%3;
		if( values[i]*values[i1]<0 )
		{
			Real t = values[i] / ( values[i] - values[i1] );
			Point3D< Real > newVert = vertices[ triangle[i1] ]*t + vertices[ triangle[ i ] ]*Real(1.0-t);
			vertices.push_back( newVert );
			bVerts[ bCount++ ] = vertices.size()-1;
			fVerts[ fCount++ ] = vertices.size()-1;
		}
		if( values[i]==0 )
		{
			if( values[i1]<0 || values[i2]<0 ) bVerts[ bCount++ ] = triangle[i];
			if( values[i1]>0 || values[i2]>0 ) fVerts[ fCount++ ] = triangle[i];
		}
		else
			if( values[i]<0 ) bVerts[ bCount++ ] = triangle[i];
			else			  fVerts[ fCount++ ] = triangle[i];
	}
	if( bCount==3 )
	{
		backTriangles.resize( 1 );
		backTriangles[0][0] = bVerts[0];
		backTriangles[0][1] = bVerts[1];
		backTriangles[0][2] = bVerts[2];
	}
	if( bCount==4 )
	{
		backTriangles.resize( 2 );
		backTriangles[0][0] = bVerts[0];
		backTriangles[0][1] = bVerts[1];
		backTriangles[0][2] = bVerts[2];
		backTriangles[1][0] = bVerts[2];
		backTriangles[1][1] = bVerts[3];
		backTriangles[1][2] = bVerts[0];
	}
	if( fCount==3 )
	{
		frontTriangles.resize( 1 );
		frontTriangles[0][0] = fVerts[0];
		frontTriangles[0][1] = fVerts[1];
		frontTriangles[0][2] = fVerts[2];
	}
	if( fCount==4 )
	{
		frontTriangles.resize( 2 );
		frontTriangles[0][0] = fVerts[0];
		frontTriangles[0][1] = fVerts[1];
		frontTriangles[0][2] = fVerts[2];
		frontTriangles[1][0] = fVerts[2];
		frontTriangles[1][1] = fVerts[3];
		frontTriangles[1][2] = fVerts[0];
	}
}

template< class Real >
int SplitEdge( Point3D< Real > v1 , Point3D< Real > v2 , Point3D< Real > pNormal , Real pOffset , Point3D< Real >& out1 , Point3D< Real >& out2 , Point3D< Real >& out3 )
{
	Real values[2];
	bool frontSet , backSet;
	frontSet = backSet = false;
	bool swap = false;

	values[0] = Point3D< Real >::Dot( v1 , pNormal ) - pOffset;
	values[1] = Point3D< Real >::Dot( v2 , pNormal ) - pOffset;
	if( values[1]<values[0] )
	{
		Real value = values[0];
		values[0] = values[1];
		values[1] = value;
		Point3D< Real > v = v1;
		v1 = v2;
		v2 = v;
		swap = true;
	}
	for( int i=0 ; i<2 ; i++ )
	{
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		out1 = v1 , out2 = v2;
		return -1;
	}
	if( !backSet )
	{
		out2 = v1 , out3 = v2;
		return 1;
	}
	Real t = values[0] / ( values[0] - values[1] );
	out1 = v1 , out3 = v2;
	out2 = v2*t + v1*Real(1.0-t);
	return 0;
}

inline void PrintLoop( std::vector< std::pair< int , int > >& polyLoop , int start )
{
	int i = start;
	do
	{
		printf( " -> %d" , polyLoop[i].first );
		i = polyLoop[i].second;
	}
	while( i!=start );
	printf( "\n" );
}
// This method will perform a flood-fill on the facets and construct polygons corresponding to the boundaries
// of the connected components.
// The assumption is that the "Facet" class supports "int operator[]( int )" to return the vertex at the specified
// index.
struct VertexList : public std::pair< unsigned int , int* >
{
	VertexList( unsigned int sz=0 , int* list=NULL ) { first = sz , second = list; }
	const int& operator[] ( int idx ) const { return second[idx]; }
	int& operator[] ( int idx ) { return second[idx]; }
	int size( void ) const { return first; }
	static unsigned int Size( const VertexList& l ){ return l.first; }
};

template< class Facet >
void MergeFacets
( const Facet* facets , int facetCount , unsigned int (*FacetSize)( const Facet& ) ,
 std::vector< VertexList >& polygons ,
 std::vector< VertexList >& parents ,
 bool (*IsLockedEdge)( int , int )=NULL ,
 bool (*IsLockedVertex)( int )=NULL
)
{
	std::vector< int > _parents , _polygon;
	std::unordered_map< long long , int > eMap;
	std::vector< std::pair< int , int > > polyLoop;

	// Track the triangles adjacent to the half-edges
	for( int i=0 ; i<facetCount ; i++ )
	{
		unsigned int fSize = FacetSize( facets[i] );
		for( int j=0 ; j<fSize ; j++ ) eMap[ HalfEdgeKey( facets[i][j] , facets[i][(j+1)%fSize] ) ] = i;
	}
	std::vector< bool > processed( facetCount , false );
	while( 1 )
	{
		_parents.clear();
		_polygon.clear();

		// Get a triangle that has not been processed
		int seed;
		for( seed=0 ; seed<facetCount ; seed++ ) if( !processed[seed] ) break;
		if( seed==facetCount ) break;

		processed[seed] = true;
		_parents.push_back( seed );

		// A linked list to represent the polygon.
		// First: the vertex label.
		// Second: pointer to the next vertex in the list
		int loopStart = 0;
		unsigned int fSize = FacetSize( facets[seed] );
		polyLoop.resize( fSize );
		for( int j=0 ; j<fSize ; j++ ) polyLoop[j] = std::pair< int , int >( facets[seed][j] , (j+1)%fSize );
		int i1 = loopStart;
		do
		{
			int i2 = polyLoop[i1].second;
			int v1 = polyLoop[i1].first;
			int v2 = polyLoop[i2].first;
			// If this is a valid edge to flood-fill across
			std::unordered_map< long long , int >::iterator iter = eMap.find( HalfEdgeKey( v2 , v1 ) );
			// Try to grow across the current edge
			while( (!IsLockedEdge || !IsLockedEdge( v1 , v2 ) ) && iter!=eMap.end() && !processed[iter->second] )
			{
				int f = iter->second;
				unsigned int fSize = FacetSize( facets[f] );
				int v;
				for( v=0 ; v<fSize ; v++ ) if( facets[f][v]==v2 && facets[f][(v+1)%fSize]==v1 ) break;
				if( v==fSize ) fprintf( stderr , "[Error] Couldn't find opposite edge in facet\n" ) , exit( 0 );
				// Add in the chain of vertices into the current loop.
				polyLoop[i1].second = polyLoop.size();
				for( int i=(v+2)%fSize ; i!=v ; i=( (i+1)%fSize ) )
					polyLoop.push_back( std::pair< int , int >( facets[f][i] , polyLoop.size()+1 ) );
				polyLoop.back().second = i2;
				_parents.push_back( f );
				processed[f] = true;
				i2 = polyLoop[i1].second;
				v2 = polyLoop[i2].first;
				iter = eMap.find( HalfEdgeKey( v2 , v1 ) );
			}
			// If that fails, advance to the next edge
			i1 = i2;
		}
		while( i1!=loopStart );

		// Trim dangling vertices
		bool done;
		do
		{
			done = true;
			int i = loopStart;
			do
			{
				if( polyLoop[i].first==polyLoop[ polyLoop[ polyLoop[i].second ].second ].first )
					if( !IsLockedVertex || !IsLockedVertex( polyLoop[polyLoop[i].second].first ) )
					{
						polyLoop[i].second = polyLoop[ polyLoop[ polyLoop[i].second ].second ].second;
						loopStart = polyLoop[i].second;
						done = false;
					}
				i = polyLoop[i].second;
			}
			while( i!=loopStart );
		}
		while( !done );

		// Now add the polygon to the list
		int i = loopStart;
		do
		{
			_polygon.push_back( polyLoop[i].first );
			i = polyLoop[i].second;
		}
		while( i!=loopStart );
		int pSize;

		polygons.push_back( VertexList( (unsigned int)_polygon.size() , new int[_polygon.size()] ) );
		parents.push_back ( VertexList( (unsigned int)_parents.size() , new int[_parents.size()] ) );
		memcpy( polygons.back().second , &_polygon[0] , sizeof( int ) * _polygon.size() );
		memcpy(  parents.back().second , &_parents[0] , sizeof( int ) * _parents.size() );
	}
}

/* BROKEN (NOT CALLED)
 * \todo fix backIndex
 template< class Real >
int SplitEdge( std::vector< Point3D< Real > >& vertices , const int* edge , Point3D< Real > pNormal , Real pOffset , int* backEdge , int* frontEdge )
{
	Real values[2];
	bool frontSet , backSet;
	frontSet = backSet = false;
	bool swap = false;

	int i1 = edge[0] , i2 = edge[1];
	Point3D< Real >& v1 = vertices[ i1 ];
	Point3D< Real >& v2 = vertices[ i2 ];
	values[0] = Point3D< Real >::Dot( v1 , pNormal ) - pOffset;
	values[1] = Point3D< Real >::Dot( v2 , pNormal ) - pOffset;
	if( values[1]<values[0] )
	{
		Real value = values[0];
		values[0] = values[1];
		values[1] = value;
		Point3D< Real > v = v1;
		v1 = v2;
		v2 = v;
		int i = i1;
		i1 = i2;
		i2 = i;
		swap = true;
	}
	for( int i=0 ; i<2 ; i++ )
	{
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		if( swap ) backEdge[0] = i2 , backEdge[1] = i1;
		else       backEdge[0] = i1 , backEdge[1] = i2;
		return -1;
	}
	if( !backSet )
	{
		if( swap ) frontEdge[0] = i2 , frontEdge[1] = i1;
		else       frontEdge[0] = i1 , frontEdge[1] = i2;
		return 1;
	}
	Real t = values[0] / ( values[0] - values[1] );
	int vIndex = vertices.size();
	vertices.push_back( v2*t + v1*Real(1.0-t) );
	backIndex[1] = frontIndex[0] = vIndex;
	if( swap ) backIndex[0] = i2 , frontIndex[1] = i1;
	else       backIndex[0] = i1 , frontIndex[1] = i2;
	return 0;
}
*/

// The assumption here is that the polygon can be split at most into two parts.
template< class Real >
void SplitPolygon(std::vector< Point3D< Real > >& vertices,
			const std::vector< int >& polygon, Point3D< Real > pNormal,
			Real pOffset, std::vector< int >& backPolygon,
			std::vector< int >& frontPolygon )
{
	std::vector< Real > values;
	values.resize( polygon.size() );
	backPolygon.clear();
	frontPolygon.clear();
	bool frontSet , backSet;
	frontSet = backSet = false;


	for( int i=0 ; i<polygon.size() ; i++ )
	{
		values[i] = Point3D< Real >::Dot( vertices[ polygon[i] ] , pNormal ) - pOffset;
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		backPolygon.resize( polygon.size() );
		for( int i=0 ; i<polygon.size() ; i++ ) backPolygon[i] = polygon[i];
		return;
	}
	if( !backSet )
	{
		frontPolygon.resize( polygon.size() );
		for( int i=0 ; i<polygon.size() ; i++ ) frontPolygon[i] = polygon[i];
		return;
	}
	frontSet = backSet = false;
	for( int i=0 ; i<polygon.size() ; i++ )
	{
		int i1 = ( i-1+(int)polygon.size() )%polygon.size();
		int i2 = ( i+1                     )%polygon.size();
		if( values[i]*values[i1]<0 )
		{
			Real t = values[i] / ( values[i] - values[i1] );
			Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
			vertices.push_back( newVert );
			backPolygon.push_back( vertices.size()-1 );
			frontPolygon.push_back( vertices.size()-1 );
			frontSet = backSet = true;
		}
		if( values[i]==0 )
		{
			backPolygon.push_back( polygon[i] );
			frontPolygon.push_back( polygon[i] );
		}
		else
			if( values[i]<0 ) backPolygon.push_back( polygon[i] ) , backSet = true;
			else			  frontPolygon.push_back( polygon[i] ) , frontSet = true;
	}
	if( !backSet ) backPolygon.clear();
	if( !frontSet ) frontPolygon.clear();
}
template< class Real >
void SplitPolygon( std::vector< Point3D< Real > >& vertices , const int* polygon , Real* values , int count , Point3D< Real > pNormal , Real pOffset ,
				   int* backPolygon , int& backCount , int* frontPolygon , int& frontCount )
{
	backCount = frontCount = 0;
	bool frontSet = false , backSet = false;

	for( int i=0 ; i<count ; i++ )
	{
		values[i] = Point3D< Real >::Dot( vertices[ polygon[i] ] , pNormal ) - pOffset;
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		for( int i=0 ; i<count ; i++ ) backPolygon[i] = polygon[i];
		backCount = count;
		return;
	}
	if( !backSet )
	{
		for( int i=0 ; i<count ; i++ ) frontPolygon[i] = polygon[i];
		frontCount = count;
		return;
	}
	frontSet = backSet = false;
	for( int i=0 ; i<count ; i++ )
	{
		int i1 = ( i-1+(int)count )%count;
		int i2 = ( i+1            )%count;
		if( values[i]*values[i1]<0 )
		{
			Real t = values[i] / ( values[i] - values[i1] );
			Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
			vertices.push_back( newVert );
			backPolygon[backCount++] = frontPolygon[frontCount++] = vertices.size()-1;
			frontSet = backSet = true;
		}
		if( values[i]==0 ) backPolygon[backCount++] = frontPolygon[frontCount++] = polygon[i];
		else
			if( values[i]<0 ) backPolygon [ backCount++] = polygon[i] ,  backSet = true;
			else			  frontPolygon[frontCount++] = polygon[i] , frontSet = true;
	}
	if( !backSet  )  backCount = 0;
	if( !frontSet ) frontCount = 0;
}

template< class Real >
void SplitPolygon( std::vector< Point3D< Real > >& vertices , std::unordered_map< long long , int >* vTable , const std::vector< int >& polygon , Point3D< Real > pNormal , Real pOffset ,
				   std::vector< int >& backPolygon , std::vector< int >& frontPolygon )
{
	std::vector< Real > values;
	values.resize( polygon.size() );
	backPolygon.clear();
	frontPolygon.clear();
	bool frontSet , backSet;
	frontSet = backSet = false;


	for( int i=0 ; i<polygon.size() ; i++ )
	{
		values[i] = Point3D< Real >::Dot( vertices[ polygon[i] ] , pNormal ) - pOffset;
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		backPolygon.resize( polygon.size() );
		for( int i=0 ; i<polygon.size() ; i++ ) backPolygon[i] = polygon[i];
		return;
	}
	if( !backSet )
	{
		frontPolygon.resize( polygon.size() );
		for( int i=0 ; i<polygon.size() ; i++ ) frontPolygon[i] = polygon[i];
		return;
	}
	frontSet = backSet = false;
	for( int i=0 ; i<polygon.size() ; i++ )
	{
		int i1 = ( i-1+(int)polygon.size() )%polygon.size();
		int i2 = ( i+1                     )%polygon.size();
		if( values[i]*values[i1]<0 )
		{
			int vIndex;
			Real t = values[i] / ( values[i] - values[i1] );
			if( vTable )
			{
				long long key = EdgeKey( polygon[i1] , polygon[i] );
				if( vTable->find( key ) == vTable->end() )
				{
					Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
					vertices.push_back( newVert );
					(*vTable)[key] = ( (int)vertices.size() ) - 1;
				}
				vIndex = (*vTable)[key];
			}
			else
			{
				Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
				vIndex = vertices.size();
				vertices.push_back( newVert );
			}
			backPolygon.push_back ( vIndex );
			frontPolygon.push_back( vIndex );
			frontSet = backSet = true;
		}
		if( values[i]==0 )
		{
			backPolygon.push_back( polygon[i] );
			frontPolygon.push_back( polygon[i] );
		}
		else
			if( values[i]<0 ) backPolygon.push_back( polygon[i] ) , backSet = true;
			else			  frontPolygon.push_back( polygon[i] ) , frontSet = true;
	}
	if( !backSet ) backPolygon.clear();
	if( !frontSet ) frontPolygon.clear();
}
#if 1
template< class Real >
void SplitPolygon( std::vector< Point3D< Real > >& vertices , std::unordered_map< long long , int >* vTable , std::unordered_map< long long , int >* heTable ,
				   const int* polygon , Real* values , int count , Point3D< Real > pNormal , Real pOffset ,
				   int* backPolygon , int& backCount , int* frontPolygon , int& frontCount )
{
	backCount = frontCount = 0;
	bool frontSet = false , backSet = false;

	for( int i=0 ; i<count ; i++ )
	{
		values[i] = Point3D< Real >::Dot( vertices[ polygon[i] ] , pNormal ) - pOffset;
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		for( int i=0 ; i<count ; i++ ) backPolygon[i] = polygon[i];
		backCount = count;
		return;
	}
	if( !backSet )
	{
		for( int i=0 ; i<count ; i++ ) frontPolygon[i] = polygon[i];
		frontCount = count;
		return;
	}
	frontSet = backSet = false;
	for( int i=0 ; i<count ; i++ )
	{
		int i1 = ( i-1+(int)count )%count;
		if( values[i]*values[i1]<0 )
		{
			int vIndex;
			if( vTable )
			{
				long long key = EdgeKey( polygon[i1] , polygon[i] );
				if( vTable->find( key ) == vTable->end() )
				{
					Real t = values[i] / ( values[i] - values[i1] );
					Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
					vertices.push_back( newVert );
					(*vTable)[key] = ( (int)vertices.size() ) - 1;
				}
				vIndex = (*vTable)[key];

				if( heTable )
				{
					long long key  = HalfEdgeKey( polygon[i1] , polygon[i] );
					long long key1 = HalfEdgeKey( polygon[i1] , vIndex     );
					long long key2 = HalfEdgeKey( vIndex      , polygon[i] );
					if( heTable->find( key ) != heTable->end() ) (*heTable)[key1] = (*heTable)[key2] = (*heTable)[key];
				}
			}
			else
			{
				Real t = values[i] / ( values[i] - values[i1] );
				Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
				vIndex = vertices.size();
				vertices.push_back( newVert );
			}
			backPolygon[backCount++] = frontPolygon[frontCount++] = vIndex;
			frontSet = backSet = true;
		}
		if( values[i]==0 ) backPolygon[backCount++] = frontPolygon[frontCount++] = polygon[i];
		else
			if( values[i]<0 ) backPolygon [ backCount++] = polygon[i] ,  backSet = true;
			else			  frontPolygon[frontCount++] = polygon[i] , frontSet = true;
	}
	if(  !backSet )  backCount = 0;
	if( !frontSet ) frontCount = 0;
}
#else
template< class Real >
void SplitPolygon( std::vector< Point3D< Real > >& vertices , std::unordered_map< long long , int >* vTable ,
				   const int* polygon , Real* values , int count , Point3D< Real > pNormal , Real pOffset ,
				   int* backPolygon , int& backCount , int* frontPolygon , int& frontCount )
{
	backCount = frontCount = 0;
	bool frontSet = false , backSet = false;

	for( int i=0 ; i<count ; i++ )
	{
		values[i] = Point3D< Real >::Dot( vertices[ polygon[i] ] , pNormal ) - pOffset;
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		for( int i=0 ; i<count ; i++ ) backPolygon[i] = polygon[i];
		backCount = count;
		return;
	}
	if( !backSet )
	{
		for( int i=0 ; i<count ; i++ ) frontPolygon[i] = polygon[i];
		frontCount = count;
		return;
	}
	frontSet = backSet = false;
	for( int i=0 ; i<count ; i++ )
	{
		int i1 = ( i-1+(int)count )%count;
		if( values[i]*values[i1]<0 )
		{
			int vIndex;
			if( vTable )
			{
				long long key;
				if( polygon[i1]>polygon[i] ) key = ( ( (long long) polygon[i1] )<<32 ) | ( (long long) polygon[i ] );
				else						 key = ( ( (long long) polygon[i ] )<<32 ) | ( (long long) polygon[i1] );
				if( vTable->find( key ) == vTable->end() )
				{
					Real t = values[i] / ( values[i] - values[i1] );
					Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
					vertices.push_back( newVert );
					(*vTable)[key] = ( (int)vertices.size() ) - 1;
				}
				vIndex = (*vTable)[key];
			}
			else
			{
				Real t = values[i] / ( values[i] - values[i1] );
				Point3D< Real > newVert = vertices[ polygon[i1] ]*t + vertices[ polygon[ i ] ]*Real(1.0-t);
				vIndex = vertices.size();
				vertices.push_back( newVert );
			}
			backPolygon[backCount++] = frontPolygon[frontCount++] = vIndex;
			frontSet = backSet = true;
		}
		if( values[i]==0 ) backPolygon[backCount++] = frontPolygon[frontCount++] = polygon[i];
		else
			if( values[i]<0 ) backPolygon [ backCount++] = polygon[i] ,  backSet = true;
			else			  frontPolygon[frontCount++] = polygon[i] , frontSet = true;
	}
	if(  !backSet )  backCount = 0;
	if( !frontSet ) frontCount = 0;
}
#endif
template< class Real >
void SplitPolygon( const std::vector< Point3D< Real > >& vertices ,
				   const int* polygon , Real* values , int count , Point3D< Real > pNormal , Real pOffset ,
				   int* backPolygon , int& backCount , int* frontPolygon , int& frontCount , std::vector< Point3D< Real > >& newVertices )
{
	backCount = frontCount = 0;
	bool frontSet = false , backSet = false;

	for( int i=0 ; i<count ; i++ )
	{
		if( polygon[i]<0 ) values[i] = Point3D< Real >::Dot( newVertices[ -1-polygon[i] ] , pNormal ) - pOffset;
		else               values[i] = Point3D< Real >::Dot(    vertices[    polygon[i] ] , pNormal ) - pOffset;
		if( values[i]<0 ) backSet  = true;
		if( values[i]>0 ) frontSet = true;
	}
	if( !frontSet )
	{
		for( int i=0 ; i<count ; i++ ) backPolygon[i] = polygon[i];
		backCount = count;
		return;
	}
	if( !backSet )
	{
		for( int i=0 ; i<count ; i++ ) frontPolygon[i] = polygon[i];
		frontCount = count;
		return;
	}
	frontSet = backSet = false;
	for( int i=0 ; i<count ; i++ )
	{
		int j = ( i-1+(int)count )%count;
		if( values[i]*values[j]<0 )
		{
			int vIndex;
			Real t = values[i] / ( values[i] - values[j] );
			int _i = polygon[i] , _j = polygon[j];
			Point3D< Real > newVert;
			if( _j<0 ) newVert  = newVertices[ -1-_j ]*t;
			else       newVert  =    vertices[    _j ]*t;
			if( _i<0 ) newVert += newVertices[ -1-_i ]*Real(1.-t);
			else       newVert +=    vertices[    _i ]*Real(1.-t);

			newVertices.push_back( newVert );
			vIndex = -newVertices.size();

			backPolygon[backCount++] = frontPolygon[frontCount++] = vIndex;
			frontSet = backSet = true;
		}
		if( values[i]==0 ) backPolygon[backCount++] = frontPolygon[frontCount++] = polygon[i];
		else
			if( values[i]<0 ) backPolygon [ backCount++] = polygon[i] ,  backSet = true;
			else			  frontPolygon[frontCount++] = polygon[i] , frontSet = true;
	}
	if(  !backSet )  backCount = 0;
	if( !frontSet ) frontCount = 0;
}
template< class Real , int Count >
template< class Vertex , int Dim >
void BarycentricCoordinatesImplementation< Real , Count >::_BarycentricCoordinates( const Vertex& p , const Vertex v[Count] , Real a[Count] , bool inside )
{
	SquareMatrix< Real , Count-1 > D;
	Vertex q = p - v[0] , dirs[Count-1];
	for( int i=0 ; i<Count-1 ; i++ ) dirs[i] = v[i+1]-v[0];
	for( int i=0 ; i<Count-1 ; i++ ) for( int j=0 ; j<Count-1 ; j++ ) D(i,j) = Vertex::Dot( dirs[i] , dirs[j] );
	Point< Real , Count-1 > coeffs;
	for( int i=0 ; i<Count-1 ; i++ ) coeffs[i] = Vertex::Dot( q , dirs[i] );
	coeffs = D.inverse() * coeffs;
	a[0] = (Real)1.;
	for( int i=0 ; i<Count-1 ; i++ ) a[0] -= coeffs[i] , a[i+1] = coeffs[i];
	if( inside )
	{
		int idx = -1;
		for( int i=0 ; i<Count ; i++ ) if( a[i]<0 ) idx = i;
		if( idx!=-1 )
		{
			Vertex _v[Count-1];
			Real _a[Count-1];
			a[idx] = 0;
			for( int i=0 , j=0 ; i<Count ; i++ ) if( i!=idx ) _v[j++] = v[i];
			BarycentricCoordinatesImplementation< Real , Count-1 >::template _BarycentricCoordinates< Vertex , Dim >( p , _v , _a  , inside );
			for( int i=0 , j=0 ; i<Count ; i++ ) if( i!=idx ) a[i] = _a[j++];
		}
	}
}
template< class Real >
Point3D< Real > RandomTrianglePoint( const Point3D< Real >& v1 , const Point3D< Real >& v2 , const Point3D< Real >& v3 )
{
	// pick random point in triangle
	Real r1 = (Real)sqrt( Random< Real >() ) , r2 = Random< Real >();
	return v1 * (1-r1) + v2 * r1 * (1-r2) + v3 * r1 * r2;
}

////////////////////////////////////////////////////////////////////////////////
/*! Uses a barycentric coordinate vector to interpolate three data values
//  @param[in]  coords      bary centric coordinates
//  @param[in]  d0, d1, d2  data values to interpolate
//  @return     interpolated data value
*///////////////////////////////////////////////////////////////////////////////
template<typename BaryCoords, typename DataType>
inline DataType BarycentricInterpolate(const BaryCoords &coords
        , const DataType &d0, const DataType &d1, const DataType &d2)
{
    // Use barycentric coordinates normalized w/ L1 norm
    return (coords[0] * d0  + coords[1] * d1 + coords[2] * d2)
           / (coords[0] + coords[1] + coords[2]);
}

////////////////////////////////////////////////////////////////////////////////
/*! Computes a triangle's circumscribed circle
//  http://en.wikipedia.org/wiki/Circumscribed_circle
//  @param[in]  p0, p1, p2      triangle vertex positions
//  @param[in]  tri             triangle to process
//  @param[out] center          incircle center
*///////////////////////////////////////////////////////////////////////////////
template<typename PointType, typename Real>
inline void Circumcircle(const PointType &p0, const PointType &p1
        , const PointType &p2, PointType &center, Real &radius)
{
    Point3D<Real> e[3];
    e[0] = Point3D<Real>(p2 - p1);
    e[1] = Point3D<Real>(p0 - p2);
    e[2] = Point3D<Real>(p1 - p0);
    Real a2 = SquareLength(e[0]);
    Real b2 = SquareLength(e[1]);
    Real c2 = SquareLength(e[2]);
    Real a = sqrt(a2);
    Real b = sqrt(b2);
    Real c = sqrt(c2);
    Real doubleA = Length(Point3D<Real>::CrossProduct(e[0], e[1]));
    // Radius =  (a * b * c) / (4A)
    // (a, b, and c are edge lengths, A is area)
    radius = (a * b * c) / (2 * doubleA);
    // Circumcenter Barycentric Coordinates:
    //  (a^2 (b^2 + c^2 - a^2), b^2 (c^2 + a^2 - b^2), c^2 (a^2 + b^2 - c^2))
    Point3D<Real> centerBaryCoords(a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2)
                                 , c2 * (a2 + b2 - c2));
    center = BarycentricInterpolate(centerBaryCoords, p0, p1, p2);
}

////////////////////////////////////////////////////////////////////////////////
/*! Computes a triangle's inscribed circle
//  http://en.wikipedia.org/wiki/Incircle
//  @param[in]  p0, p1, p2      triangle vertex positions
//  @param[out] center          incircle center
//  @param[out] radius          incircle radius
*///////////////////////////////////////////////////////////////////////////////
template<typename PointType, typename Real>
inline void Incircle(const PointType &p0, const PointType &p1
        , const PointType &p2, PointType &center, Real &radius)
{
    Point3D<Real> e[3];
    e[0] = Point3D<Real>(p2 - p1);
    e[1] = Point3D<Real>(p0 - p2);
    e[2] = Point3D<Real>(p1 - p0);
    Real a = Length(e[0]);
    Real b = Length(e[1]);
    Real c = Length(e[2]);
    Real doubleA = Length(Point3D<Real>::CrossProduct(e[0], e[1]));
    // Radius =  (2A) / (a + b + c)
    // (a, b, and c are edge lengths, A is area)
    radius = doubleA / (a + b + c);
    // Incenter Barycentric Coordinates: (a, b, c)
    Point3D<Real> centerBaryCoords(a, b, c);
    center = BarycentricInterpolate(centerBaryCoords, p0, p1, p2);
}

template< class Real >
Point3D< Real > NearestPointOnEdge( Point3D< Real > point , const Point3D< Real > edge[2] , Real& b0 , Real& b1 )
{
	Point3D< Real > d = edge[1] - edge[0] , p = point - edge[0];
	Real dot = Point3D< Real >::Dot( p , d );
	if( dot<0 ) 
	{
		b0 = 1.;
		return edge[0];
	}
	else if( dot>Point3D< Real >::SquareNorm( d ) ) { 
		b1 = 1.; 
		return edge[1];
	}
	else
	{
		// Solve for the minimizer of:
		//                            E(t) = || p - t*d ||^2
		//                                 = || p ||^2 - 2 t < p , d > + t^2 || d ||^2
		//            =>             0 = -< p , d > + t || d ||^2
		//            <=>    t = < p , d > / || d ||^2
		Real t = dot / Point3D< Real >::SquareNorm( d );
		b0 = (Real)( 1.-t ) , b1 = t;
		return edge[0] + d * t;
	}
}

template< class Real >
Point3D< Real > NearestPointOnTriangle( Point3D< Real > point , const Point3D< Real > triangle[3] , Real* b )
{

	b[0] = b[1] = b[2] = 0;
	Point3D< Real > d[] = { triangle[1]-triangle[0] , triangle[2]-triangle[0] } , p = point - triangle[0] , n = CrossProduct( d[0] , d[1] );

	if( !Length(n) ) return ( triangle[0] + triangle[1] + triangle[2] ) / (Real)3.;


	if     ( Point3D< Real >::Dot( point-triangle[0] , CrossProduct( n , triangle[1]-triangle[0] ) )<0 ){ Point3D< Real > edge[] = { triangle[0] , triangle[1] } ; return NearestPointOnEdge( point , edge , b[0] , b[1] ); }
	else if( Point3D< Real >::Dot( point-triangle[1] , CrossProduct( n , triangle[2]-triangle[1] ) )<0 ){ Point3D< Real > edge[] = { triangle[1] , triangle[2] } ; return NearestPointOnEdge( point , edge , b[1] , b[2] ); }
	else if( Point3D< Real >::Dot( point-triangle[2] , CrossProduct( n , triangle[0]-triangle[2] ) )<0 ){ Point3D< Real > edge[] = { triangle[2] , triangle[0] } ; return NearestPointOnEdge( point , edge , b[2] , b[0] ); }
	else
	{
		// Solve for the minimizer of:
		//                            E(s,t) = || p - s*d[0]-t*d[1] ||^2
		//                                   = || p ||^2 - 2 s < p , d[0] > - 2 t < p , d[1] > + 2 s t < d[0] , d[1] > + s^2 || d[0] ||^2 + t^2 || d[1] ||^2
		//   =>  (0,0) = ( -< p , d[0] > + t < d[0] , d[1] > + s || d[0] ||^2 , -< p , d[1] > + s < d[0] , d[1] > + t || d[1] ||^2
		//            <=> | < p , d[0] > | = | < d[0] , d[0] >   < d[0] , d[1] > | | s |
		//                | < p , d[1] > |   | < d[0] , d[1] >   < d[1] , d[1] > | | t |
		SquareMatrix< Real , 2 > M , M_inverse;
		M(0,0) = Point3D< Real >::SquareNorm( d[0] ) , M(1,0) = M(0,1) = Point3D< Real >::Dot( d[0] , d[1] ) , M(1,1) = Point3D< Real >::SquareNorm( d[1] );
		Real det = M(0,0)*M(1,1) - M(0,1)*M(1,0);
		M_inverse(0,0) = M(1,1) , M_inverse(0,1) = -M(0,1) , M_inverse(1,0) = -M(1,0) , M_inverse(1,1) = M(0,0);
		M_inverse /= det;
		Point2D< Real > st = M_inverse * Point2D< Real >( Point3D< Real >::Dot( p , d[0] ) , Point3D< Real >::Dot( p , d[1] ) );
		// Sanity check
#if 1
#else
		if( st[0]<0 || st[1]<0 || st[0]+st[1]>1 ) ERROR_OUT( "Bad barycentric coordinates: " , 1.-st[0]-st[1] , " " , st[0] , " " , st[1] , " " , Length(n) );
#endif
		b[0] = (Real)( 1. - st[0] - st[1] ) , b[1] = (Real)( st[0] , b[2] = st[1] );
		Point3D< Real > ret = triangle[0] * ( Real )( 1. - st[0] - st[1] ) + d[0] * st[0] + d[1] * st[1];

		return triangle[0] * ( Real )( 1. - st[0] - st[1] ) + d[0] * st[0] + d[1] * st[1];
	}
}

/////////////
// Simplex //
/////////////
#ifdef NEW_SIMPLEX
template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
{
	Real values[K+1];
	for( unsigned int k=0 ; k<=K ; k++ ) values[k] = Point< Real , Dim >::Dot( p[k] , pNormal ) - pOffset;
	return split( values , back , front );
}
template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::split( const Real values[K+1] , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
{
	bool frontSet = false , backSet = false;

	// Evaluate the hyper-plane's function at the vertices and mark if strictly front/back vertices have been found
	for( unsigned int k=0 ; k<=K ; k++ ) backSet |= ( values[k]<0 ) , frontSet |= ( values[k]>0 );
#else // !NEW_SIMPLEX
template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
{
	Real values[K+1];
	bool frontSet = false , backSet = false;

	// Evaluate the hyper-plane's function at the vertices and mark if strictly front/back vertices have been found
	for( unsigned int k=0 ; k<=K ; k++ )
	{
		values[k] = Point< Real , Dim >::Dot( p[k] , pNormal ) - pOffset;
		backSet |= ( values[k]<0 ) , frontSet |= ( values[k]>0 );
	}
#endif // NEW_SIMPLEX
	// If all the vertices are behind or on, or all the vertices are in front or on, we are done.
	if( !frontSet ){ back.push_back( *this ) ; return; }
	if( !backSet ){ front.push_back( *this ) ; return; }

	// Pick some intersection of the hyper-plane with a simplex edge
	unsigned int v1 , v2;
	Point< Real , Dim > midPoint;
	{
		for( unsigned int i=0 ; i<K ; i++ ) for( unsigned int j=i+1 ; j<=K ; j++ ) if( values[i]*values[j]<0 )
		{
			v1 = i , v2 = j;
			Real t1 = values[i] / ( values[i] - values[j] ) , t2 = (Real)( 1. - t1 );
			midPoint = p[j]*t1 + p[i]*t2;
		}
	}
	// Iterate over each face of the simplex, split it with the hyper-plane and connect the sub-simplices to the mid-point
	for( unsigned int i=0 ; i<=K ; i++ )
	{
		if( i!=v1 && i!=v2 ) continue;
		Simplex< Real , Dim , K-1 > f;		// The face
		Simplex< Real , Dim , K > s;		// The sub-simplex
		for( unsigned int j=0 , idx=0 ; j<=K ; j++ )	if( j!=i ) f[idx++] = p[j];
		std::vector< Simplex< Real , Dim , K-1 > > _back , _front;
		f.split( pNormal , pOffset , _back , _front );
		s[i] = midPoint;

		for( unsigned int j=0 ; j<_back.size() ; j++ ) 
		{
			for( unsigned int k=0 ; k<K ; k++ ) s[ k<i ? k : k+1 ] = _back[j][k];
			back.push_back( s );
		}

		for( unsigned int j=0 ; j<_front.size() ; j++ ) 
		{
			for( unsigned int k=0 ; k<K ; k++ ) s[ k<i ? k : k+1 ] = _front[j][k];
			front.push_back( s );
		}
	}
}

template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::_nearest( Point< Real , Dim > point , Real barycentricCoordinates[K+1] ) const
{
	// Compute the projection of the point onto the plane containing the simplex
	{
		Point< Real , K > x;
		SquareMatrix< Real , K > D;
		Point< Real , Dim > q = point - p[0] , dirs[K];
		for( int i=0 ; i<K ; i++ ) dirs[i] = p[i+1] - p[0];
		for( int i=0 ; i<K ; i++ ) for( int j=0 ; j<K ; j++ ) D(i,j) = Point< Real , Dim >::Dot( dirs[i] , dirs[j] );
		for( int i=0 ; i<K ; i++ ) x[i] = Point< Real , Dim >::Dot( q , dirs[i] );
		x = D.inverse() * x;
		barycentricCoordinates[0] = (Real)1.;
		for( int i=0 ; i<K ; i++ ) barycentricCoordinates[0] -= x[i] , barycentricCoordinates[i+1] = x[i];
	}

	for( int k=0 ; k<=K ; k++ ) if( barycentricCoordinates[k]<0 )
	{
		Simplex< Real , Dim , K-1 > s;
		for( int i=0 ; i<=K ; i++ )
			if     ( i<k ) s[i  ] = p[i];
			else if( i>k ) s[i-1] = p[i];
		Real _barycentricCoordinates[K];
		s._nearest( point , _barycentricCoordinates );
		for( int i=0 ; i<=K ; i++ )
			if     ( i<k ) barycentricCoordinates[i] = _barycentricCoordinates[i  ];
			else if( i>k ) barycentricCoordinates[i] = _barycentricCoordinates[i-1];
		barycentricCoordinates[k] = 0;
		break;
	}
}

template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::NearestKey::init( Simplex< Real , Dim , K > simplex )
{
	SquareMatrix< Real , K > D;

	_base = simplex[0];
	for( int k=0 ; k<K ; k++ ) _dirs[k] = simplex[k+1] - simplex[0];
	for( int i=0 ; i<K ; i++ ) for( int j=0 ; j<K ; j++ ) D(i,j) = Point< Real , Dim >::Dot( _dirs[i] , _dirs[j] );
	_Dinv = D.inverse();

	Simplex< Real , Dim , K-1 > s;
	for( int k=0 ; k<=K ; k++ )
	{
		for( int i=0 ; i<=K ; i++ )
			if     ( i<k ) s[i  ] = simplex[i];
			else if( i>k ) s[i-1] = simplex[i];
		_faceKeys[k].init( s );
	}
}

template< class Real , unsigned int Dim , unsigned int K >
void Simplex< Real , Dim , K >::NearestKey::_nearest( Point< Real , Dim > point , Real barycentricCoordinates[K+1] ) const
{
	// Compute the projection of the point onto the plane containing the simplex
	{
		Point< Real , K > x;
		Point< Real , Dim > q = point - _base;
		for( unsigned int k=0 ; k<K ; k++ ) x[k] = Point< Real , Dim >::Dot( q , _dirs[k] );
		x = _Dinv * x;
		barycentricCoordinates[0] = (Real)1.;
		for( int i=0 ; i<K ; i++ ) barycentricCoordinates[0] -= x[i] , barycentricCoordinates[i+1] = x[i];
	}

	for( int k=0 ; k<=K ; k++ ) if( barycentricCoordinates[k]<0 )
	{
		Real _barycentricCoordinates[K];
		_faceKeys[k]._nearest( point , _barycentricCoordinates );
		for( int i=0 ; i<=K ; i++ )
			if     ( i<k ) barycentricCoordinates[i] = _barycentricCoordinates[i  ];
			else if( i>k ) barycentricCoordinates[i] = _barycentricCoordinates[i-1];
		barycentricCoordinates[k] = 0;
		break;
	}
}

#ifdef NEW_SIMPLEX
//////////////////
// SimplexIndex //
//////////////////
template< unsigned int K , typename Index >
template< typename Real , typename Vertex >
void SimplexIndex< K , Index >::split( const Real values[K+1] , std::vector< Vertex > &vertices , EdgeTable< Index > &edgeTable , std::vector< SimplexIndex >& back , std::vector< SimplexIndex >& front ) const
{
	bool frontSet = false , backSet = false;

	// Evaluate the hyper-plane's function at the vertices and mark if strictly front/back vertices have been found
	for( unsigned int k=0 ; k<=K ; k++ ) backSet |= ( values[k]<0 ) , frontSet |= ( values[k]>0 );
	// If all the vertices are behind or on, or all the vertices are in front or on, we are done.
	if( !frontSet ){ back.push_back( *this ) ; return; }
	if( !backSet ){ front.push_back( *this ) ; return; }

	// Pick some intersection of the hyper-plane with a simplex edge
	unsigned int v1=-1 , v2=-1;
	auto InitializationFunction = [&]( void )
	{
		Real t1 = values[v1] / ( values[v1] - values[v2] ) , t2 = (Real)( 1. - t1 );
		vertices.push_back( vertices[ idx[v2] ]*t1 + vertices[ idx[v1] ]*t2 );
		return (Index)( vertices.size()-1 );
	};
	for( unsigned int i=0 ; i<K ; i++ ) for( unsigned int j=i+1 ; j<=K ; j++ ) if( values[i]*values[j]<0 ) v1 = i , v2 = j;
	Index midPointIndex = edgeTable( idx[v1] , idx[v2] , InitializationFunction ); 

	// Iterate over each face of the simplex, split it with the hyper-plane and connect the sub-simplices to the mid-point
	for( unsigned int i=0 ; i<=K ; i++ )
	{
		if( i!=v1 && i!=v2 ) continue;
		Real _values[K];
		SimplexIndex< K-1 , Index > f;		// The face
		SimplexIndex< K , Index > s;		// The sub-simplex
		for( unsigned int j=0 , _idx=0 ; j<=K ; j++ ) if( j!=i ){ f[_idx] = idx[j] ; _values[_idx] = values[j] ; _idx++; }
		std::vector< SimplexIndex< K-1 , Index > > _back , _front;
		f.split( _values , vertices , edgeTable , _back , _front );
		s[i] = midPointIndex;

		for( unsigned int j=0 ; j<_back.size() ; j++ ) 
		{
			for( unsigned int k=0 ; k<K ; k++ ) s[ k<i ? k : k+1 ] = _back[j][k];
			back.push_back( s );
		}

		for( unsigned int j=0 ; j<_front.size() ; j++ ) 
		{
			for( unsigned int k=0 ; k<K ; k++ ) s[ k<i ? k : k+1 ] = _front[j][k];
			front.push_back( s );
		}
	}
}

#endif // NEW_SIMPLEX


/////////
// Ray //
/////////
template< typename Real , unsigned int Dim >
Ray< Real , Dim >::Ray( void ){}

template< typename Real , unsigned int Dim >
Ray< Real , Dim >::Ray( const Point< Real , Dim > &p , const Point< Real , Dim > &d ) : position(p) , direction(d) {}

template< typename Real , unsigned int Dim >
Point< Real , Dim > Ray< Real , Dim >::operator() ( double s ) const { return position+direction*s; }

template< typename Real , unsigned int Dim >
Ray< Real , Dim >  Ray< Real , Dim >::operator +  ( const Point< Real , Dim > &p ) const { return Ray( position+p , direction );}

template< typename Real , unsigned int Dim >
Ray< Real , Dim > &Ray< Real , Dim >::operator += ( const Point< Real , Dim > &p ){ position += p ; return *this; }

template< typename Real , unsigned int Dim >
Ray< Real , Dim >  Ray< Real , Dim >::operator -  ( const Point< Real , Dim > &p ) const { return Ray( position-p , direction );}

template< typename Real , unsigned int Dim >
Ray< Real , Dim > &Ray< Real , Dim >::operator -= ( const Point< Real , Dim > &p ){ position -= p ; return *this; }

template< typename Real , unsigned int Dim >
Ray< Real , Dim > operator * ( const SquareMatrix< Real , Dim+1 > &m , const Ray< Real , Dim >& r )
{
	return Ray< Real , Dim >( m * r.position , SquareMatrix< Real , Dim >(m) * r.direction );
}
