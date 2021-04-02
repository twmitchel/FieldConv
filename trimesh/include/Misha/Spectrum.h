/*
Copyright (c) 2019, Michael Kazhdan and Thomas Mitchel
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

#ifndef SPECTRUM_INCLUDED
#define SPECTRUM_INCLUDED

#include <functional>
#include <Spectra/SymGEigsSolver.h>
#include "Misha/FEM.h"
#include "Misha/Solver.h"

template< typename Real >
struct Spectrum
{
	template< typename GReal >
	void set( const std::vector< Point3D< GReal > > &vertices , const std::vector< TriangleIndex > &triangles , unsigned int dimension , Real offset , bool lump );
	void setFrom ( const Eigen::MatrixXd& eVal, const Eigen::MatrixXd& eVec);
	void read( const std::string &fileName );
	void write( const std::string &fileName ) const;
	template< typename SpectralFunctor > Real spectralDistance( const SpectralFunctor &F , unsigned int i , unsigned int j , int beginE , int endE ) const;
	template< typename SpectralFunctor > Real spectralDistance( const SpectralFunctor &F , unsigned int i , TriangleIndex tri , Point3D< Real > weights , int beginE , int endE ) const;
	Real spectralDistance( unsigned int i , unsigned int j , int beginE , int endE ) const;
	Real spectralDistance( unsigned int i , TriangleIndex tri , Point3D< Real > weights , int beginE , int endE ) const;
	Real biharmonicDistance( unsigned int i , unsigned int j ) const;
	Real biharmonicDistance( unsigned int i , TriangleIndex tri , Point3D< Real > weights ) const;
	Real commuteDistance( unsigned int i , unsigned int j ) const;
	Real commuteDistance( unsigned int i , TriangleIndex tri , Point3D< Real > weights ) const;
	Real diffusionDistance( Real diffusionTime , unsigned int i , unsigned int j ) const;
	Real diffusionDistance( Real diffusionTime , unsigned int i , TriangleIndex tri , Point3D< Real > weights ) const;
	size_t size( void ) const { return _eigenvalues.size(); }
	Real &eValue( unsigned int idx ){ return _eigenvalues[idx]; }
	const Real &eValue( unsigned int idx ) const { return _eigenvalues[idx]; }
	std::vector< Real > &eVector( unsigned int idx ){ return _eigenvectors[idx]; }
	const std::vector< Real > &eVector( unsigned int idx ) const { return _eigenvectors[idx]; }
	Real HKS( unsigned int i , Real diffusionTime ) const;
protected:
	static const unsigned long long _MAGIC_NUMBER;
	std::vector< Real > _eigenvalues;
	std::vector< std::vector< Real > > _eigenvectors;
};


//////////////
// Spectrum //
//////////////
template< typename Real > const unsigned long long Spectrum< Real >::_MAGIC_NUMBER = 0x2019ull;

template< typename Real >
template< typename GReal >
void Spectrum< Real >::set( const std::vector< Point3D< GReal > > &vertices , const std::vector< TriangleIndex > &triangles , unsigned int dimension , Real offset , bool lump )
{
	// [Definition]
	//	We define the generalized eigensystem (A,B) to be the system A v = \lambda B v
	// [Fact 1]
	// If (v,\lambda) is a solution to the ges (A,B) then (v,\lambda+\epsilon) is a solution to the ges (A+\epsilon*B,B):
	//		(A + \epsilon B) v = (\lambda + \epsilon) B v
	//	<=>	A v + \epsilon B v = \lambda B v + \epsilon B V
	//	<=>                A v = \lambda B v
	// [Fact 2]
	// If (w,\delta) is a solution to the ges (A^{-1},B^{-1}) then (A^{-1}w,1/\delta) is a solution to the ges (A,B):
	//		A^{-1} w = \delta B^{-1} w
	//	<=> v = \delta B^{-1} A v
	//	<=> 1/\delta B v = A v
	// [Corollary]
	// If (w,\delta) is a solution to the ges ( (A+\epsilon*B)^{-1} , B^{-1} ) then:
	//	=> ( (A+\epsilon*B)^{-1} w , 1/\delta ) is a solution to the ges (A+\epsilon*B,B)
	//	=> ( (A+\epsilon*B)^{-1} w , 1\delta-\epsilon ) is a solution to the ges (A,B)

	typedef EigenSolverCholeskyLDLt< Real > Solver;
	struct InverseOperator
	{
		Solver solver;
		InverseOperator( const SparseMatrix< Real , int > &M ) : _M(M) , solver( M ){}
		int rows( void ) const { return (int)_M.rows; }
		int cols( void ) const { return (int)_M.rows; }
		void perform_op( const Real *in , Real *out ) const { const_cast< Solver & >(solver).solve( in , out ); };
	protected:
		const SparseMatrix< Real , int > &_M;
	};

	struct InverseBOperator
	{
		Solver solver;
		InverseBOperator( const SparseMatrix< Real , int > &M ) : _M(M) , solver( M ){}
		int rows( void ) const { return (int)_M.rows; }
		int cols( void ) const { return (int)_M.rows; }
		void solve( const Real *in , Real *out ) const { _M.Multiply( in , out ); }
		void mat_prod( const Real *in , Real *out ) const { const_cast< Solver & >(solver).solve( in , out ); };
	protected:
		const SparseMatrix< Real , int > &_M;
	};

	std::vector< TriangleIndex > _triangles = triangles;
	FEM::RiemannianMesh< Real > mesh( GetPointer( _triangles ) , _triangles.size() );
	mesh.template setMetricFromEmbedding< 3 >( [&]( unsigned int idx ){ return Point3D< Real >( vertices[idx] ); } );
	mesh.makeUnitArea();
	SparseMatrix< Real , int > M = mesh.template massMatrix< FEM::BASIS_0_WHITNEY >( lump ) , S = mesh.template stiffnessMatrix< FEM::BASIS_0_WHITNEY >();
	
	
	/*
   for( int i=0 ; i<M.rows ; i++ ) 
	{
	   for( int j=0 ; j<M.rowSizes[i] ; j++ )
	   { 

	      if (S[i][j].Value < 1.0e-4)
	      {
	         S[i][j].Value = 1.0e-4;
	      }
	      
	      if (M[i][j].Value < 1.0e-4)
	      {
	         M[i][j].Value = 1.0e-4;
	      }
	      
	   }
	}
	*/
	

	// Offset the stiffness matrix so that it becomes positive definite
   #pragma omp parallel for
	for( int i=0 ; i<M.rows ; i++ ) for( int j=0 ; j<M.rowSizes[i] ; j++ ) S[i][j].Value += M[i][j].Value * offset;


	InverseOperator op( S );
	
	InverseBOperator Bop( M );

	Spectra::SymGEigsSolver< Real , Spectra::LARGEST_ALGE , InverseOperator , InverseBOperator , Spectra::GEIGS_REGULAR_INVERSE > geigs( &op , &Bop , dimension , 2*dimension );
	geigs.init();
	int nconv = geigs.compute();
	if( nconv!=dimension ) fprintf( stderr , "[WARNING] Number of converged is not equal to dimension: %d != %d\n" , nconv , dimension );
	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;
	if( geigs.info()==Spectra::SUCCESSFUL )
	{
		evalues = geigs.eigenvalues();
		evecs = geigs.eigenvectors();
	}
	else if( geigs.info()==Spectra::NOT_COMPUTED )    fprintf( stderr , "[ERROR] Not computed\n" ) , exit(0);
	else if( geigs.info()==Spectra::NOT_CONVERGING 	) fprintf( stderr , "[ERROR] Not converging\n" ) , exit(0);
	else if( geigs.info()==Spectra::NUMERICAL_ISSUE ) fprintf( stderr , "[ERROR] Numerical issue\n" ) , exit(0);
	else                                              fprintf( stderr , "[ERROR] Failed\n" ) , exit(0);

	_eigenvalues.resize( evecs.cols() );
	_eigenvectors.resize( evecs.cols() );

	for( int i=0 ; i<evecs.cols() ; i++ )
	{
		_eigenvectors[i].resize( vertices.size() );
		_eigenvalues[i] = (Real)1./evalues[i] - offset;
		std::vector< Real > w( vertices.size() );
		for( int j=0 ; j<evecs.rows() ; j++ ) w[j] = evecs(j,i);
		op.perform_op( &w[0] , &_eigenvectors[i][0] );
		Real l2Norm = 0;
#pragma omp parallel for reduction( + : l2Norm )
		for( int j=0 ; j<M.rows ; j++ ) for( int k=0 ; k<M.rowSizes[j] ; k++ ) l2Norm += M[j][k].Value * _eigenvectors[i][j] * _eigenvectors[i][ M[j][k].N ];
		l2Norm = (Real)sqrt( l2Norm );
#pragma omp parallel for
		for( int j=0 ; j<_eigenvectors[i].size() ; j++ ) _eigenvectors[i][j] /= l2Norm;
	}
}

template< typename Real>
void Spectrum< Real >::setFrom ( const Eigen::MatrixXd& eVal, const Eigen::MatrixXd& eVec)
{
   _eigenvalues.resize(eVal.rows());
   _eigenvectors.resize(eVal.rows());
   
   for (int l = 0; l < eVal.rows(); l++)
   {
      _eigenvalues[l] = eVal(l, 0);
      _eigenvectors[l].resize(0);
      
      for (int i = 0; i < eVec.cols(); i++)
      {
         _eigenvectors[l].push_back (eVec(l, i));
      }
   }
}

template< typename Real >
void Spectrum< Real >::read( const std::string &fileName )
{
	FILE *fp = fopen( fileName.c_str() , "rb" );
	if( !fp ) fprintf( stderr , "[ERROR] Failed to open file for reading: %s\n" , fileName.c_str() ) , exit( 0 );
	/// Read simple header
	// Read Magic Number
	unsigned long long magicNum;
	if( fread( &magicNum , sizeof(unsigned long long) , 1 , fp )!=1 ) fprintf( stderr , "[ERROR] Failed to read magic number\n" ) , exit( 0 );
	if( magicNum!=_MAGIC_NUMBER ) fprintf( stderr , "[ERROR] Bad magic number: %s\n" , fileName.c_str() ) , exit( 0 );

	// type size
	unsigned int typeSize;
	if( fread( &typeSize , sizeof(unsigned int) , 1 , fp )!=1 ) fprintf( stderr , "[ERROR] Failed to read type size\n" ) , exit( 0 );
	bool needCasting = typeSize != sizeof(Real);
	if( needCasting ) fprintf( stderr , "[WARNING] Types sizes don't match: %d != %d\n" , (int)typeSize , (int)sizeof(Real) );
	// Num of eigenvalues
	unsigned int numOfValues;
	if( fread( &numOfValues , sizeof(unsigned int) , 1 , fp )!=1 ) fprintf( stderr , "[ERROR] Failed to read number of values\n" ) , exit( 0 );
	_eigenvalues.resize( numOfValues );
	_eigenvectors.resize( numOfValues );
	// Dimensions
	unsigned int dimension;
	if( fread( &dimension , sizeof(unsigned int) , 1,  fp )!=1 ) fprintf( stderr , "[ERROR] Failed to read dimension\n" ) , exit( 0 );
	for( int i=0 ; i<_eigenvectors.size() ; i++ ) _eigenvectors[i].resize( dimension );
	/// Content
	if( needCasting )
	{
		// Eigenvalues
		unsigned char *tempMemory = new unsigned char[ typeSize*numOfValues ];
		if( fread( tempMemory , typeSize , numOfValues , fp )!=numOfValues ) fprintf( stderr , "[ERROR] Failed to read values\n" ) , exit( 0 );
#pragma omp parallel for
		for( int i=0 ; i<_eigenvalues.size() ; i++ )
		{
			switch( typeSize )
			{
				case sizeof(float):  _eigenvalues[i] = (Real)reinterpret_cast< float       * >(tempMemory)[i] ; break;
					case sizeof(double): _eigenvalues[i] = (Real)reinterpret_cast< double      * >(tempMemory)[i] ; break;
					default: fprintf( stderr , "[ERROR] Could not determine type from size: %d\n" , (int) typeSize ) , exit( 0 );
			}
		}
		delete[] tempMemory;
		// Eigenvectors
		tempMemory = new unsigned char[ typeSize*dimension ];
		for( int i=0 ; i<_eigenvectors.size() ; i++ )
		{
			if( fread( tempMemory , typeSize , dimension , fp )!=dimension ) fprintf( stderr , "[ERROR] Failed to read eigenvectors\n" ) , exit( 0 );
#pragma omp parallel for
			for( int j=0 ; j<_eigenvectors[i].size() ; j++ )
			{
				switch(typeSize )
				{
					case sizeof(float):  _eigenvectors[i][j] = (Real)reinterpret_cast< float       * >(tempMemory)[j] ; break;
						case sizeof(double): _eigenvectors[i][j] = (Real)reinterpret_cast< double      * >(tempMemory)[j] ; break;
						default: fprintf( stderr , "[ERROR] Could not determine type from size: %d\n" , (int) typeSize ) , exit( 0 );
				}
			}
		}
		delete[] tempMemory;
	}
	else
	{
		// Eigenvalues
		if( fread( _eigenvalues.data() , sizeof(Real) , numOfValues , fp )!=numOfValues ) fprintf( stderr , "[ERROR] Failed to read eigenvalues\n" ) , exit( 0 );
		// Eigenvectors
		for( int i=0 ; i<_eigenvectors.size() ; i++ ) if( fread( _eigenvectors[i].data() , sizeof(Real) , dimension , fp )!=dimension ) fprintf( stderr , "[ERROR] Failed to read eigenvectors\n" ) , exit( 0 );
	}
	fclose( fp );
}
template< typename Real >
void Spectrum< Real >::write( const std::string &fileName ) const
{
	FILE *fp = fopen( fileName.c_str() , "wb" );
	if( !fp ) fprintf( stderr , "[ERROR] Failed to open file for writing: %s\n" , fileName.c_str() ) , exit( 0 );
	// Write simple header
	// Write Magic Number
	fwrite( &_MAGIC_NUMBER , sizeof(unsigned long long) , 1 , fp );

	// type size
	unsigned int typeSize = sizeof(Real);
	fwrite( &typeSize , sizeof(unsigned int) , 1 , fp );
	// Num of eigenvalues
	unsigned int numOfValues = (unsigned int)_eigenvectors.size();
	fwrite( &numOfValues , sizeof(unsigned int) , 1 , fp );
	// Dimensions
	unsigned int dimension = (unsigned int)_eigenvectors[0].size();
	fwrite( &dimension , sizeof(unsigned int) , 1,  fp );
	// Eigenvalues
	fwrite( _eigenvalues.data() , sizeof(Real) , numOfValues , fp );
	// Eigenvectors
	for( int i=0 ; i<_eigenvectors.size() ; i++ ) fwrite( _eigenvectors[i].data() , sizeof(Real) , dimension , fp );
	fclose( fp );
}

template< typename Real >
template< typename SpectralFunctor >
Real Spectrum< Real >::spectralDistance( const SpectralFunctor &F , unsigned int i , unsigned int j , int beginE , int endE ) const
{
	Real distance = (Real)0;
	beginE = std::max< int >( beginE , 0 );
	endE = std::min< int >( endE , _eigenvectors.size() );
	for( unsigned int k=beginE ; k<endE ; k++ )
	{
		auto& v = _eigenvectors[k];
		Real temp = ( v[i]-v[j] ) * F( _eigenvalues[k] );
		distance += temp * temp;
	}
	return (Real)sqrt( distance );
}
template< typename Real >
template< typename SpectralFunctor >
Real Spectrum< Real >::spectralDistance( const SpectralFunctor &F , unsigned int i , TriangleIndex tri , Point3D< Real> weights , int beginE , int endE ) const
{
	Real distance = (Real)0;
	beginE = std::max< int >( beginE , 0 );
	endE = std::min< int >( endE , _eigenvectors.size() );
	for( unsigned int k=beginE ; k<endE ; k++ )
	{
		auto& v = _eigenvectors[k];
		Real v1 = v[i] , v2 = v[ tri[0] ] * weights[0] + v[ tri[1] ] * weights[1] + v[ tri[2] ] * weights[2];
		Real temp = ( v1-v2 ) * F( _eigenvalues[k] );
		distance += temp*temp;
	}
	return (Real)sqrt( distance );
}

template< typename Real >
Real Spectrum< Real >::spectralDistance( unsigned int i , unsigned int j , int beginE , int endE ) const
{
	Real distance = (Real)0;
	beginE = std::max< int >( beginE , 0 );
	endE = std::min< int >( endE , (int)_eigenvectors.size() );
	for( int k=beginE ; k<endE ; k++ )
	{
		auto& v = _eigenvectors[k];
		Real temp = ( v[i]-v[j] );
		distance += temp * temp;
	}
	return (Real)sqrt( distance );
}
template< typename Real >
Real Spectrum< Real >::spectralDistance( unsigned int i , TriangleIndex tri , Point3D< Real> weights , int beginE , int endE ) const
{
	Real distance = (Real)0;
	beginE = std::max< int >( beginE , 0 );
	endE = std::min< int >( endE , (int)_eigenvectors.size() );
	for( int k=beginE ; k<endE ; k++ )
	{
		auto& v = _eigenvectors[k];
		Real v1 = v[i] , v2 = v[ tri[0] ] * weights[0] + v[ tri[1] ] * weights[1] + v[ tri[2] ] * weights[2];
		Real temp = ( v1-v2 );
		distance += temp*temp;
	}
	return (Real)sqrt( distance );
}


template< typename Real >
Real Spectrum< Real >::biharmonicDistance( unsigned int i , unsigned int j ) const
{
	return spectralDistance( []( double ev ){ return 1./ev; } , i , j , 1 , _eigenvectors.size() );
}
template< typename Real >
Real Spectrum< Real >::biharmonicDistance( unsigned int i , TriangleIndex tri , Point3D< Real> weights ) const
{
	return spectralDistance( []( double ev ){ return 1./ev; } , i , tri , weights , 1 , _eigenvectors.size() );
}
template< typename Real >
Real Spectrum< Real >::commuteDistance( unsigned int i , unsigned int j ) const
{
	return spectralDistance( []( double ev ){ return 1./std::sqrt(ev); } , i , j , 1 , _eigenvectors.size() );
}
template< typename Real >
Real Spectrum< Real >::commuteDistance( unsigned int i , TriangleIndex tri , Point3D< Real> weights ) const
{
	return spectralDistance( []( double ev ){ return 1./std::sqrt(ev); } , i , tri , weights , 1 , _eigenvectors.size() );
}
template< typename Real >
Real Spectrum< Real >::diffusionDistance( Real diffusionTime , unsigned int i , unsigned int j ) const
{
	return spectralDistance( [&]( double ev ){ (Real)exp( - ev * diffusionTime ); } , i , j , 1 , _eigenvectors.size() );
}
template< typename Real >
Real Spectrum< Real >::diffusionDistance( Real diffusionTime , unsigned int i , TriangleIndex tri , Point3D< Real> weights ) const
{
	return spectralDistance( [&]( double ev ){ (Real)exp( - ev * diffusionTime ); } , i , tri , weights , 1 , _eigenvectors.size() );
}

template< typename Real >
Real Spectrum< Real >::HKS( unsigned int i , Real diffusionTime ) const
{
	Real hks = (Real)0;
	for( int j=0 ; j<_eigenvectors.size() ; j++ ) hks += (Real)( exp( - _eigenvalues[j] * diffusionTime ) * _eigenvectors[j][i] * _eigenvectors[j][i] );
	return hks;
}

#endif // SPECTRUM_INCLUDED
