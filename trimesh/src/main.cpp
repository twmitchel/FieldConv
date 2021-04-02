#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h> 
#include <pybind11/numpy.h>
#include <Eigen/Dense>

#include <time.h>

// Spectral 
#include <cstdlib>
#include <vector>
#include <Misha/CmdLineParser.h>
#include <Misha/Ply.h>
#include <Misha/PlyVertexData.h>
#include <Misha/CmdLineParser.h>
#include <Misha/Spectrum.h>
#include <Misha/Miscellany.h>
#include <Misha/FileTricks.h>
#include <Misha/TriMesh.h>
#include <Misha/GetDescriptor.inl>
#include <Misha/Image.h>
#include <Misha/RegularGrid.h>

// Geometry central
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/halfedge_factories.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"


// libigl
#include <igl/decimate.h>
#include <igl/predicates/predicates.h>
#include <external/predicates/predicates.h>
#include <igl/delaunay_triangulation.h>
#include <igl/readPLY.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>


#ifndef M_PI
#define M_PI 3.1415926535897932384
#endif 

#include <sstream>
#include <chrono>

using namespace geometrycentral;
using namespace geometrycentral::surface;
namespace py = pybind11;
using namespace pybind11::literals;
// Equivalent to "from decimal import Decimal"
py::object delTri = py::module::import("scipy.spatial").attr("Delaunay");


// Geometry-central data
std::unique_ptr<HalfedgeMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;


bool sortOneRing(std::pair<double, int> v1, std::pair<double, int> v2){return (v1.first < v2.first);}
bool sortLabels(std::pair<int, int> v1, std::pair<int, int> v2){return (v1.first < v2.first);}

bool sortLogs(std::pair<Point2D<double>, int> v1, std::pair<Point2D<double>, int> v2){return (v1.first.squareNorm() < v2.first.squareNorm());}

// Unit orthogonal vector
Eigen::Vector3d unitOrthogonal(Eigen::Vector3d v)
{
 bool b0 = (v[0] <  v[1]) && (v[0] <  v[2]);
 bool b1 = (v[1] <= v[0]) && (v[1] <  v[2]);
 bool b2 = (v[2] <= v[0]) && (v[2] <= v[1]);

 Eigen::Vector3d b;
 b << double(b0), double(b1), double(b2);
 
 Eigen::Vector3d o = v.cross (b);

 return o.normalized ();
}

//==========================================================================================
// igl Delaunay nonsense
//==========================================================================================
int orient2dPredicates(const double* pa, const double* pb, const double* pc)
{
    const Eigen::Vector2d a(pa[0], pa[1]);
    const Eigen::Vector2d b(pb[0], pb[1]);
    const Eigen::Vector2d c(pc[0], pc[1]);

    const auto result = igl::predicates::orient2d<Eigen::Vector2d>(a, b, c);

    if (result == igl::predicates::Orientation::POSITIVE) {
        return 1;
    } else if (result == igl::predicates::Orientation::NEGATIVE) {
        return -1;
    } else {
        return 0;
    }
}

int inCirclePredicates(const double* pa, const double* pb, const double* pc, const double* pd)
{
    const Eigen::Vector2d a(pa[0], pa[1]);
    const Eigen::Vector2d b(pb[0], pb[1]);
    const Eigen::Vector2d c(pc[0], pc[1]);
    const Eigen::Vector2d d(pd[0], pd[1]);

    const auto result = igl::predicates::incircle(a, b, c, d);

    if (result == igl::predicates::Orientation::INSIDE) {
        return 1;
    } else if (result == igl::predicates::Orientation::OUTSIDE) {
        return -1;
    } else {
        return 0;
    }
};

// ========================================================================================
// Loads a mesh from a NumPy array
// ========================================================================================

// Loads a mesh from a NumPy array
std::tuple<std::unique_ptr<HalfedgeMesh>, std::unique_ptr<VertexPositionGeometry>>
loadMesh_np(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces) {

  // Set vertex positions
  std::vector<Vector3> vertexPositions(pos.rows());
  for (size_t i = 0; i < pos.rows(); i++) {
    vertexPositions[i][0] = pos(i, 0);
    vertexPositions[i][1] = pos(i, 1);
    vertexPositions[i][2] = pos(i, 2);
  }

  // Get face list
  std::vector<std::vector<size_t>> faceIndices(faces.rows());
  for (size_t i = 0; i < faces.rows(); i++) {
    faceIndices[i] = {faces(i, 0), faces(i, 1), faces(i, 2)};
  }

  return makeHalfedgeAndGeometry(faceIndices, vertexPositions);
}

// Loads a mesh from a NumPy array
std::tuple<std::unique_ptr<HalfedgeMesh>, std::unique_ptr<VertexPositionGeometry>>
loadMesh_np(Eigen::MatrixXd& pos, Eigen::MatrixXi& faces) {

  // Set vertex positions
  std::vector<Vector3> vertexPositions(pos.rows());
  for (size_t i = 0; i < pos.rows(); i++) {
    vertexPositions[i][0] = pos(i, 0);
    vertexPositions[i][1] = pos(i, 1);
    vertexPositions[i][2] = pos(i, 2);
  }

  // Get face list
  std::vector<std::vector<size_t>> faceIndices(faces.rows());
  for (size_t i = 0; i < faces.rows(); i++) {
    faceIndices[i] = {faces(i, 0), faces(i, 1), faces(i, 2)};
  }

  return makeHalfedgeAndGeometry(faceIndices, vertexPositions);
}



void importMesh ( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, std::vector< Point3D<float> >& vertices, std::vector< TriangleIndex >& triangles)
{

   vertices.resize ( pos.rows () );
   
   for (int i = 0; i < pos.rows (); i++)
   {
      vertices[i][0] = (float) pos(i, 0);
      vertices[i][1] = (float) pos(i, 1);
      vertices[i][2] = (float) pos(i, 2);
   }
   
   triangles.resize (faces.rows ());
   
   for (int i = 0; i < faces.rows (); i++)
   {
      triangles[i][0] = (int) faces(i, 0);
      triangles[i][1] = (int) faces(i, 1);
      triangles[i][2] = (int) faces(i, 2);
   }
   
}

void importSafeMesh ( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, std::vector< Point3D<float> >& vertices, std::vector< TriangleIndex >& triangles)
{

   vertices.resize ( pos.rows () );
   
   for (int i = 0; i < pos.rows (); i++)
   {
      vertices[i][0] = (float) pos(i, 0);
      vertices[i][1] = (float) pos(i, 1);
      vertices[i][2] = (float) pos(i, 2);
   }
   
   triangles.resize(0);
   
   
   for (int i = 0; i < faces.rows (); i++)
   {
      int v1 = (int) faces(i, 0);
      int v2 = (int) faces(i, 1);
      int v3 = (int) faces(i, 2);
      
      if (v1 != v2 && v2 != v3 && v1 != v3)
      {
         TriangleIndex T;
         T[0] = v1;
         T[1] = v2;
         T[2] = v3;
         
         triangles.push_back (T);
      }
      
      
   }
   
}

// =========================================================================================
// Decimate mesh
// =========================================================================================

std::vector<Eigen::MatrixXd> decimateMesh ( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, int target_nodes)
{
   Eigen::MatrixXd V = pos;
   Eigen::MatrixXi F (faces.rows(), 3);
   
   for (int l = 0; l < F.rows(); l++)
   {
      F(l, 0) = faces(l, 0);
      F(l, 1) = faces(l, 1);
      F(l, 2) = faces(l, 2);
   }
   
   int target_faces = 2*target_nodes;
   Eigen::MatrixXd VD;
   Eigen::MatrixXi FD;
   Eigen::VectorXi J;
   
   igl::decimate(V, F, target_faces, VD, FD, J);
   
   Eigen::MatrixXd FDD(FD.rows(), 3);
   
   for (int l = 0; l < FD.rows (); l++)
   {
      FDD(l, 0) = FD(l, 0);
      FDD(l, 1) = FD(l, 1);
      FDD(l, 2) = FD(l, 2);
   }
   
   std::vector<Eigen::MatrixXd> out {VD, FDD};
   
   return out;
}

//==========================================================================================
// Return total area of mesh
// =========================================================================================
double meshArea( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces)
{ 
   double area = 0;
   
   std::vector<Point3D<float>> vertices;
   std::vector<TriangleIndex> triangles;
   
   importMesh(pos, faces, vertices, triangles);
   
   for (int l = 0; l < triangles.size (); l++)
   {      
      double d1 = std::sqrt( (vertices[triangles[l][1]] - vertices[triangles[l][0]]).squareNorm ());
      double d2 = std::sqrt( (vertices[triangles[l][2]] - vertices[triangles[l][0]]).squareNorm ());
      double d3 = std::sqrt( (vertices[triangles[l][2]] - vertices[triangles[l][1]]).squareNorm ());
      
      double s = (d1 + d2 + d3) / 2;
      
      area += std::sqrt ( s * (s - d1) * (s - d2) * (s - d3) );
   }
   
   return area;
}

// =========================================================================================
// Assign vertex labels to decimated mesh
// =========================================================================================

Eigen::MatrixXd labelSampling ( Eigen::MatrixXd& pos, Eigen::MatrixXd& pos_sub, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labels)
{

   Eigen::MatrixXd L (pos_sub.rows(), 1);
   
   #pragma omp parallel
   for (int l = 0; l < pos_sub.rows(); l++)
   {
      double minD = 1.0e12;
      int minI = 0;
      
      for (int j = 0; j < pos.rows(); j++)
      {
         double d = (pos_sub.block(l, 0, 1, 3) - pos.block(j, 0, 1, 3)).norm();
         
         if (d < minD)
         {
            minD = d;
            minI = j;
         }
      }
      
      L(l, 0) = labels(minI, 0);
  }
  
  return L;

}

Eigen::MatrixXd invertMap (Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labels, Eigen::MatrixXd& pos)
{
   
   Eigen::MatrixXd labInv(pos.rows(), 1);
   
   std::vector<bool> bi;
   bi.resize(pos.rows(), false);
   
   for (int l = 0; l < labels.size(); l++)
   {
      labInv(labels[l]-1, 0) = l+1;
      bi[labels[l]-1] = true;
   }
   
   for (int l = 0; l < bi.size(); l++)
   {
      if (!bi[l])
      {
         double minDist = 1.0e10;
         int nearest;
         
         for (int i = 0; i < pos.rows(); i++)
         {
            if (bi[i])
            {
               double d = (pos.block(l, 0, 1, 3) - pos.block(i, 0, 1, 3)).norm();
               
               if (d < minDist)
               {
                  minDist = d;
                  nearest = i;
               }
            }
         }
         
      
        labInv(l, 0) = labInv(nearest, 0);
     }
  }
  
  return labInv;


}
   
   
Eigen::MatrixXd composeMap (Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labelsTem2Tar, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labelsTem2Sour, Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces)
{
   
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

  
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0, true);
   
   Eigen::MatrixXd labSour2Tar(pos.rows(), 1);
   
   std::vector<bool> bi;
   bi.resize(pos.rows(), false);
   
   for (int l = 0; l < labelsTem2Sour.size(); l++)
   {
      
      labSour2Tar(labelsTem2Sour[l]-1, 0) = labelsTem2Tar(l, 0);
      bi[labelsTem2Sour[l]-1] = true;
   }
   
   for (int l = 0; l < bi.size(); l++)
   {
      if (!bi[l])
      {
         double minDist = 1.0e10;
         int nearest;
         

      
         Vertex t = mesh->vertex( l );
      
         VertexData<double> geo = heatSolver.computeDistance(t);

      
         for (int i = 0; i < pos.rows(); i++)
         {
            if (bi[i])
            {
               Vertex p = mesh->vertex( i );
               double d = geo[p];
               
               if (d < minDist)
               {
                  minDist = d;
                  nearest = i;
               }
            }
         }
         
      
        labSour2Tar(l, 0) = labSour2Tar(nearest, 0);
     }
  }
  
  return labSour2Tar;


}


   
// =========================================================================================
// Compute robust normals
// =========================================================================================

Eigen::MatrixXd computeNormals ( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sample_points)
{
   // Load mesh
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   Eigen::MatrixXd normals (sample_points.rows(), 3);
   
   geometry->requireVertexNormals();
   
   for (int l = 0; l < sample_points.rows(); l++)
   {
      int sampleInd  = sample_points(l);
      
      Vector3 n  = geometry->vertexNormals[mesh->vertex(sampleInd)];
      
      for (int i = 0; i < 3; i++)
      {
         normals(l, i) = n[i];
      }
   }
   
  geometry->unrequireVertexNormals();
   return normals;
}

//===========================================================================================
// Lumped mass matrix
//===========================================================================================
Eigen::MatrixXd lumpedMass (Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces)
{
   std::vector<Point3D<float>> vertices;
   std::vector<TriangleIndex> triangles;
   
   importMesh (pos, faces, vertices, triangles);
   
   Eigen::MatrixXd M (pos.rows(), pos.rows());
   M.setZero();
   
   for (int l = 0; l < triangles.size(); l++)
   {      
      double d1 = std::sqrt( (vertices[triangles[l][1]] - vertices[triangles[l][0]]).squareNorm ());
      double d2 = std::sqrt( (vertices[triangles[l][2]] - vertices[triangles[l][0]]).squareNorm ());
      double d3 = std::sqrt( (vertices[triangles[l][2]] - vertices[triangles[l][1]]).squareNorm ());
      
      double s = (d1 + d2 + d3) / 2;
      
      double tArea = std::sqrt ( s * (s - d1) * (s - d2) * (s - d3) );
     
      
      for (int i = 0; i < 3; i++)
      {
         M(triangles[l][i], triangles[l][i]) += tArea / 3.0;
      }
   }
   
   
   return M;
}
   
   
   
   
// =========================================================================
// Return LB Eigenvalues + Eigenspectrum
// ========================================================================
Eigen::MatrixXd LBSpectrum (Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, int dim = 50 )
{

   std::vector<Point3D<float>> vertices;
   std::vector<TriangleIndex> triangles;
   
   importMesh (pos, faces, vertices, triangles);
   
   //printf("nVerts = [%d, %d], nTris = [%d, %d]\n", (int) pos.rows (), (int) vertices.size (), (int) faces.rows (), (int) triangles.size ());
   //fflush(stdout);
   
   // Compute specral decomposition
   Spectrum< double > spectrum;
   spectrum.set (vertices, triangles, dim, 100.f, true);
   
   Eigen::MatrixXd specOut (dim, pos.rows () + 1);
   
   for (int i = 0; i < dim; i++)
   {
      specOut(i, 0) = spectrum.eValue(i);
      
      for (int j = 0; j < pos.rows (); j++)
      {
         specOut(i, 1 + j) = spectrum.eVector(i)[j];
      }
   }
   
   return specOut;

}

// =============================================================================
// Biharmonic Distances
// =============================================================================

double biharmonicDist (int nodeX, int nodeY, Eigen::MatrixXd& spec)
{
   double d = 0.0;
   
   if (nodeX != nodeY)
   {
      for (int k = 1; k < spec.rows (); k++)
      {
         d += ( spec(k, nodeX + 1) - spec(k, nodeY + 1) ) * ( spec(k, nodeX + 1) - spec(k, nodeY + 1) ) / ( spec(k, 0) * spec(k, 0) );
      }
   }
   
   return sqrt(d);
}

double biharmonicDist (int node, int fv1, int fv2, int fv3, Eigen::MatrixXd& spec)
{
   
   double d1 = 0.0;
   double d2 = 0.0;
   double d3 = 0.0;

   for (int k = 1; k < spec.rows (); k++)
   {

      d1 +=  ( spec(k, node + 1) - spec(k, fv1 + 1) ) * ( spec(k, node + 1) - spec(k, fv1 + 1) ) / ( spec(k, 0) * spec(k, 0) );
      
      d2 +=  ( spec(k, node + 1) - spec(k, fv2 + 1) ) * ( spec(k, node + 1) - spec(k, fv2 + 1) ) / ( spec(k, 0) * spec(k, 0) );
      
      d3 +=  ( spec(k, node + 1) - spec(k, fv3 + 1) ) * ( spec(k, node + 1) - spec(k, fv3 + 1) ) / ( spec(k, 0) * spec(k, 0) );
        
   }
    
   return (1.0 / 3.0) * ( std::sqrt (d1) + std::sqrt(d2) + std::sqrt(d3) );
}

double biharmonicDist (int tv1, int tv2, int tv3, int fv1, int fv2, int fv3, Eigen::MatrixXd& spec)
{    
   if ( tv1 == fv1 && tv2 == fv2 && tv3 == fv3)
   {
      return 0.0;
   }
   else
   {
      double d1 = biharmonicDist ( tv1, fv1, fv2, fv3, spec);
      double d2 = biharmonicDist ( tv2, fv1, fv2, fv3, spec);
      double d3 = biharmonicDist ( tv3, fv1, fv2, fv3, spec);

      return (1.0 / 3.0) * ( d1 + d2 + d3);
   }
}

//=============================================================================
// Get one rings
//=============================================================================
std::vector<std::vector<int>> computeOneRings( std::vector<Point3D<float>>& vertices, std::vector<TriangleIndex>& triangles)
{
   std::vector<std::vector<int>> rings;
   rings.resize(vertices.size());
   
   for (int l = 0; l < triangles.size(); l++)
   {
      for (int i = 0; i < 3; i++)
      {
         rings[triangles[l][i]].push_back ( triangles[l][ (i+1)%3 ] );
         rings[triangles[l][i]].push_back ( triangles[l][ (i+2)%3 ] );
      }
   }
   
   for (int l = 0; l < rings.size(); l++)
   {
      std::sort( rings[l].begin(), rings[l].end() );     
      
      rings[l].erase( std::unique( rings[l].begin(), rings[l].end() ), rings[l].end() );
   }
   
   return rings;
}

//============================================================================
// Laplacian smoothing
//============================================================================
std::vector<Point3D<float>> laplacianSmooth ( const std::vector<Point3D<float>>& vertices, std::vector<std::vector<int>>& rings, int n = 1)
{
   std::vector<Point3D<float>> _vertices = vertices;
   
   std::vector<Point3D<float>> nodes;
   nodes.resize( vertices.size());
   
   for (int m = 0; m < n; m++)
   {
      if (m > 0) { _vertices = nodes; }
      
      for (int l = 0; l < _vertices.size(); l++)
      {
         nodes[l] = Point3D<float> (0.f, 0.f, 0.f);
         
         int r = rings[l].size();
         if (r == 0)
         {
            nodes[l] = _vertices[l];
         }
         else
         {
            for (int i = 0; i < rings[l].size(); i++)
            {
               nodes[l] += _vertices[rings[l][i]];
            }
            
            nodes[l] = nodes[l] / r;
         }
      }
   }
   
}
   
   
   
   
// ============================================================================
// Compute ECHO Descriptors
// ============================================================================
//std::vector<Eigen::MatrixXd> computeECHO( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::MatrixXd& eVals, Eigen::MatrixXd& eVecs, int nBins, float alpha)
std::vector<Eigen::MatrixXd> computeECHO( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, int nBins, float alpha)
{
   
   static const int NSamples = 7;

   std::vector< double > signalValues;
   std::vector< double > signal;
   std::vector< std::pair< Point2D< double > , Point2D< double > > > dualFrameField;
   std::vector< double > hks;
   Spectrum< double > spectrum;
   std::function< double ( double ) > spectralFunction = SpectralFunction( DISTANCE_BIHARMONIC , 0.1 );
   std::function< double ( double ) > weightFunction;

   int nRadialBins = nBins;

   weightFunction = []( double d2 ){ return std::exp( -d2 ); };

   std::vector<Point3D<float>> vertices;
   std::vector<TriangleIndex> triangles;
    

   importSafeMesh(pos, faces, vertices, triangles);
   
   //TriMesh< float > tMesh( vertices , triangles );
   
   //spectrum.setFrom (eVals, eVecs);
   
   spectrum.set (vertices, triangles, 200, 100.f, true);  
    
   TriMesh< float > tMesh( vertices , triangles );

   
   std::vector<double> hksTimes { 4 * std::log(10) / (15*spectrum.eValue(175) ), 
   4 * std::log(10) / ( 10 * spectrum.eValue(121) ),  4 * std::log(10) / spectrum.eValue(60),
   4 * std::log(10) / spectrum.eValue(30), 4 * std::log(10) / spectrum.eValue(5)};
   
   Eigen::MatrixXd hksM ( vertices.size(), 5);
   hksM.setZero();
   
   
   hks.resize( vertices.size() );
   for( int i=0 ; i<vertices.size() ; i++ ) 
   {
      hks[i] = spectrum.HKS( i , 0.1 );
      
      for (int j = 0; j < 5; j++)
      {
         hksM(i, j) = spectrum.HKS(i, hksTimes[j]);
      }
   }
   
   for( int i=0 ; i<spectrum.size() ; i++ )
   {
      double scale = spectralFunction( spectrum.eValue(i) );
      auto &ev = spectrum.eVector(i);
      for( int j=0 ; j<ev.size() ; j++ ) ev[j] *= scale;
   }

   auto squareEdgeLengthFunctor = [&]( TriangleIndex tri )
   {
      Point3D< double > d;
      for( int i=0 ; i<3 ; i++ ) d[i] = spectrum.spectralDistance( tri[(i+1)%3] , tri[(i+2)%3] , 1 , (int)spectrum.size() );
      for( int i=0 ; i<3 ; i++ ) d[i] *= d[i];
      return d;
   };
   
   tMesh.initMetricsFromSquareEdgeLengths( squareEdgeLengthFunctor );
   
   // Compute the frame field and the signal
   signal.resize( triangles.size() );
   dualFrameField.resize( triangles.size() );
   
       
   std::vector< Point2D< double > > triangleGradients;
   tMesh.metricGradient( hks , triangleGradients );

   for( int l=0 ; l<triangles.size(); l++)
   {
      SquareMatrix< double , 2 > m = tMesh.triangleMetric( l );
      signal[l] = std::sqrt( Point2D< double >::Dot( triangleGradients[l] , m * triangleGradients[l] ) );
      dualFrameField[l].first  = triangleGradients[l] / signal[l];
      dualFrameField[l].second = tMesh.metricRotate90( l , dualFrameField[l].first );
      dualFrameField[l].first = m * dualFrameField[l].first;
      dualFrameField[l].second = m * dualFrameField[l].second;
   }
    
   float rho = (float)( alpha * std::sqrt( tMesh.totalArea() / M_PI ) );

   // Compute descriptors
   int res = 2 * nBins + 1;
   
   Eigen::MatrixXd echoV ( vertices.size(), res * res);
   echoV.setZero();
   
   for (int l = 0; l < vertices.size (); l++)
   {
      RegularGrid< float, 2> F;
      
      F = spectralEcho< float, NSamples>(tMesh, spectrum, signal, dualFrameField, l, rho, nRadialBins, weightFunction);
      
      int ind = 0;
      for (int i = 0; i < res; i++)
      {
         for (int j = 0; j < res; j++)
         {
            echoV(l, ind) = F(i, j);
            ind++;
         }
      }
      
   }
   
   std::vector<Eigen::MatrixXd> out;
   out.push_back (echoV); out.push_back (hksM);
   return out;
}

Eigen::MatrixXd computeError( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& pred, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& truth)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();
   
   float surfaceArea = 0.0f;
   for (Face f : mesh->faces()) 
   {
      surfaceArea += geometry->faceArea(f);
   }
   
   double aNrm = std::sqrt(1.0 / surfaceArea);
  
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0, true);
   
   Eigen::MatrixXd error(pred.rows(), 1);
   
   for (int l = 0; l < pred.rows(); l++)
   {
      Vertex p = mesh->vertex( pred(l, 0) );
      
      Vertex t = mesh->vertex( truth(l, 0) );
      
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      //error(l, 0) = geo[p];
      //double d = geo[p];
      
      error(l, 0) = geo[p] * aNrm;
      
   }
   
   return error;
   
}

/*
Eigen::MatrixXd errorToPlot( Eigen::MatrixXd& error, double maxE, int nSamples)
{

   std::vector<double> E;
   
   for (int i = 0; i < error.rows(); i++)
   {
      for (int j = 0; j < error.cols(); j++)
      {
         E.push_back (error(i, j));
      }
   }
   
   std::sort(E.begin(), E.end());
   
   std::vector<double> ET;
   for (int i = 0; i < E.size(); i++)
   {
      if ( E[i] < maxE )
      {
         ET.push_back(E[i]);
      }
      else
      {
         ET.push_back(E[i]);
         break;
      }
   }
   
   int nE = ET.size();
   
   double maxPCT = ( (double) nE) / E.size();
   
   double step = ( (double) nE) / nSamples;
   double stepPCT = 1.0 / nSamples;
   
   Eigen::MatrixXd plot (nSamples, 2);
   plot.setZero();
   plot(nSamples-1, 0) = ET.back();
   plot(nSamples-1, 1) = maxPCT;
   
   for (int l = 1; l < nSamples; l++)
   {
      int sF = std::floor(step * l) - 1;
      int sC = std::ceil(step * l) - 1;
      
      double t;
      if (sF != sC)
      {
         t = (step*l - sF) / (sC - sF);
      }
      else
      {
         t = 0;
      }
      
      plot(l-1, 0) = ET[sF] + (ET[sC] - ET[sF])*t;
      
      plot(l - 1, 1) = maxPCT * ( l * stepPCT);
   }
   
   return plot;

}
*/

Eigen::MatrixXd errorToPlot( Eigen::MatrixXd& error, double maxE, int nSamples=5000)
{

   std::vector<double> E;
   
   for (int i = 0; i < error.rows(); i++)
   {
      for (int j = 0; j < error.cols(); j++)
      {
         E.push_back (error(i, j));
      }
   }
   
   std::sort(E.begin(), E.end());
   
   double step = maxE / nSamples;
   
   Eigen::MatrixXd plot (nSamples + 1, 1);
   plot.setZero();
   
   double totalP = (double) E.size();
   
   for (int l = 1; l <= nSamples; l++)
   {
      int sCount = 0;
      for (int i = 0; i < E.size(); i++)
      {  
         if ( E[i] <= step * l )
         {
            sCount ++;
         }
         else
         {
            break;
         }
      }
      
      plot(l, 0) = sCount / totalP;
   }
   
   return plot;
}


   
   
// ============================================================================
// Compute triangle metrics
// ============================================================================
Eigen::MatrixXd triMetric(Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces)
{

   Eigen::MatrixXd metDet (faces.rows (), 4);
   
   for (int l = 0; l < faces.rows (); l++)
   {
      double d1 = biharmonicDist ( faces(l, 0), faces(l, 1), spec);
      double d2 = biharmonicDist ( faces(l, 0), faces(l, 2), spec);
      double d3 = biharmonicDist ( faces(l, 1), faces(l, 2), spec);

      
      double a = d1 * d1;
      double b = ( d1 * d1 + d2 * d2 - d3 * d3 ) / 2.0;
      double c = d2 * d2;
      
      double det = a * c - b*b;
      
      if (det < 1.0e-12) 
      {
         //std::cout << "Warning: dangerously small metric determinant" << std::endl;
         //fflush(stdout);
      }
      
      metDet(l, 0) = a; 
      metDet(l, 1) = b;
      metDet(l, 2) = c;
      metDet(l, 3) = det;
   }
   
   return metDet;
}


//==============================================================================
// Biharmonic FPS
//==============================================================================
Eigen::MatrixXd fpsBiharmonic (Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& set, int nSamples)
{
   Eigen::MatrixXd spec = LBSpectrum(pos, faces, 200);
   
   std::vector<int> samples;
   std::vector<bool> isSampled;
   isSampled.resize(set.rows(), false);
   
   std::srand( (unsigned)time(NULL) );
   int ind0 = std::rand() % set.rows();
   samples.push_back (ind0);
   isSampled[ind0] = true;
   
   for (int l = 0; l < set.size(); l++)
   {
      if (set(l,0) == set(ind0,0))
      {
         isSampled[l] = true;
      }
   }
   
   std::vector<double> prevMin;
   prevMin.resize(set.rows(), 1.0e12);
   
   for (int l = 1; l < nSamples; l++)
   {
      double maxD = 0.0;
      int maxInd;
      
      for (int i = 0; i < set.size(); i++)
      {
         if (!isSampled[i] )
         {
            
            double minDist = std::min(biharmonicDist( set(i, 0), set(samples.back(), 0), spec), prevMin[i]);
                           
            prevMin[i] = minDist;
            
            if (minDist > maxD)
            {
               maxD = minDist;
               maxInd = i;
            }
         }
      }
      
      samples.push_back (maxInd);
      isSampled[maxInd] = true;
      
      for (int j = 0; j < set.size(); j++)
      {
         if ( set(j, 0) == set(maxInd, 0) )
         {
            isSampled[j] = true;
         }
      }
      
   }
   
   //std::cout<< "after sampling" << std::endl;
   Eigen::MatrixXd sOut(samples.size(), 1);
   
   for (int l = 0; l < nSamples; l++)
   {
      sOut(l, 0) = samples[l];
   }
   
   return sOut;
}

//==============================================================================
// Geodesic FPS
//==============================================================================
Eigen::MatrixXd fpsGeodesic (Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& set, int nSamples)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
   std::vector<int> samples;
   std::vector<bool> isSampled;
   isSampled.resize(set.rows(), false);
   
   std::srand( (unsigned)time(NULL) );
   int ind0 = std::rand() % set.rows();
   samples.push_back (ind0);
   isSampled[ind0] = true;
   
   for (int l = 0; l < set.size(); l++)
   {
      if (set(l,0) == set(ind0,0))
      {
         isSampled[l] = true;
      }
   }
   
   

   //Eigen::MatrixXd distMat(nSamples-1, set.rows());
   //distMat.setZero();
   
   std::vector<double> minDistances;
   minDistances.resize(set.size(), 1e12);
   
   for (int l = 1; l < nSamples; l++)
   {
      
      Vertex t = mesh->vertex( set(samples.back(), 0) );
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      double maxMinD = 0;
      int sInd = 0;

      for (int i = 0; i < set.size(); i++)
      {
         if (!isSampled[i] )
         {
               
            Vertex p = mesh->vertex( set(i, 0) );
            double d = geo[p];
            
            minDistances[i] = std::min(d, minDistances[i]);
            
            if (minDistances[i] > maxMinD)
            {
               maxMinD = minDistances[i];
               sInd = i;
            }
         }

      }


      samples.push_back (sInd);
      isSampled[sInd] = true;
      
      for (int j = 0; j < set.size(); j++)
      {
         if ( set(j, 0) == set(sInd, 0) )
         {
            isSampled[j] = true;
         }
      }
 
   }
   
  geometry->unrequireVertexIndices();
   //std::cout<< "after sampling" << std::endl;
   Eigen::MatrixXd sOut(samples.size(), 1);
   
   for (int l = 0; l < nSamples; l++)
   {
      sOut(l, 0) = samples[l];
   }
   
   return sOut;
}


Eigen::MatrixXd biharmonicFPSBatch (Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& batch, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& set, int nSamples)
{
   std::vector<int> samples;
   std::vector<bool> isSampled;
   isSampled.resize(set.rows(), false);
   
   std::srand( (unsigned)time(NULL) );
   int ind0 = std::rand() % set.rows();
   samples.push_back (ind0);
   isSampled[ind0] = true;
   
   std::vector<double> prevMin;
   prevMin.resize(set.rows(), 1.0e12);
   
   for (int l = 1; l < nSamples; l++)
   {
      double maxD = 0.0;
      int maxInd;
      
      for (int i = 0; i < set.size(); i++)
      {
         if (!isSampled[i] && batch(i, 0) == batch(ind0, 0) )
         {
            
            double minDist = std::min(biharmonicDist( set(i, 0), set(samples.back(), 0), spec), prevMin[i]);
                           
            prevMin[i] = minDist;
            
            if (minDist > maxD)
            {
               maxD = minDist;
               maxInd = i;
            }
         }
      }
      
      samples.push_back (maxInd);
      isSampled[maxInd] = true;
      
   }
   
   //std::cout<< "after sampling" << std::endl;
   Eigen::MatrixXd sOut(samples.size(), 1);
   
   for (int l = 0; l < nSamples; l++)
   {
      sOut(l, 0) = samples[l];
   }
   
   return sOut;
}

Eigen::MatrixXd biharmonicFPS (Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& set, int nSamples)
{
   Eigen::Matrix<size_t, Eigen::Dynamic, 1> batch = set;
   
   batch.setZero();
   
   return biharmonicFPSBatch(spec, batch, set, nSamples);
   
}

   
   
   
// =============================================================================
// Tri Areas 
// =============================================================================
std::vector<Eigen::MatrixXd> computeAreas (Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces )
{
   //printf("spec = [%d, %d], faces = [%d, %d]\n", (int) spec.rows (), (int) spec.cols(), (int) faces.rows (), (int) faces.cols () );
   

   Eigen::MatrixXd triAreas (faces.rows (), 1);
   Eigen::MatrixXd ringAreas(pos.rows(), 1);
   ringAreas.setZero();
   
   std::vector<Point3D<float>> vertices;
   std::vector<TriangleIndex> triangles;
   
   importMesh(pos, faces, vertices, triangles);
   
   for (int l = 0; l < triangles.size(); l++)
   {      
      double d1 = std::sqrt( (vertices[triangles[l][1]] - vertices[triangles[l][0]]).squareNorm ());
      double d2 = std::sqrt( (vertices[triangles[l][2]] - vertices[triangles[l][0]]).squareNorm ());
      double d3 = std::sqrt( (vertices[triangles[l][2]] - vertices[triangles[l][1]]).squareNorm ());
      
      double s = (d1 + d2 + d3) / 2;
      
      double tArea = std::sqrt ( s * (s - d1) * (s - d2) * (s - d3) );
      
      triAreas(l, 0) = tArea;
      
      for (int i = 0; i < 3; i++)
      {
         ringAreas(triangles[l][i], 0) += triAreas(l, 0);
      }
   }
   
   return std::vector<Eigen::MatrixXd> {triAreas, ringAreas};
}

// =============================================================================
// Biharmonic pooling neighbors 
// =============================================================================

Eigen::MatrixXd poolNeighbors (Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& selected)
{
   py::gil_scoped_acquire acquire;
    
   Eigen::MatrixXd pContrib (samples.rows (), 1);
   
   #pragma omp paralell
   for (int l = 0; l < samples.rows (); l++)
   {
      double minDist = 1.0e10;
      int minLabel = 0;
      
      for (int i = 0; i < selected.rows (); i++)
      {
        
         double d = biharmonicDist (samples(l, 0), selected(i, 0), spec);
      
         if ( d < minDist)
         {
            minDist = d;
            minLabel = i;
         }
      }
      
      pContrib(l, 0) = minLabel;
   }
   
   return pContrib;
   
}



// ===================================================================================
// Computes edges contributiing to convolutions
// ===================================================================================

Eigen::MatrixXd gradEdgesBatch (Eigen::MatrixXd& pos, Eigen::MatrixXd& normals, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& indices, double maxD, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& batch)
{
  // py::gil_scoped_acquire acquire;
   //py::scoped_interpreter guard{};
   std::vector<std::vector<int>> rings;
   
   rings.resize(pos.rows());
   
   #pragma omp paralell
   for (int l = 0; l < pos.rows(); l++)
   {
      Eigen::Vector3d v0;
      v0 << pos(l, 0), pos(l, 1), pos(l, 2);
      
 
      std::vector<std::pair<Point2D<double>, int>> coord_ind;
      coord_ind.push_back(std::pair<Point2D<double>, int>(Point2D<double>(0.0, 0.0), indices(l, 0)));

      
      Eigen::Vector3d nVec;
      nVec << normals(indices(l, 0), 0), normals(indices(l, 0), 1), normals(indices(l, 0), 2);
      Eigen::Vector3d X = unitOrthogonal(nVec);
      Eigen::Vector3d Y = nVec.cross(X);
      Eigen::Matrix3d projN = Eigen::MatrixXd::Identity(3, 3) - nVec * nVec.transpose();
      
      for (int j = 0; j < pos.rows(); j++)
      {
         if (batch(j, 0) == batch(l, 0) && j != l)
         {
            Eigen::Vector3d v;
            v << pos(j, 0), pos(j, 1), pos(j, 2);
            
            Eigen::Vector3d ln = (projN * (v - v0)).normalized();
            
            double d = (v - v0).norm();
            
            ln *= d;
            

            double lnX = X.dot(ln);
            double lnY = Y.dot(ln);
            
            coord_ind.push_back(std::pair<Point2D<double>, int>(Point2D<double>(lnX, lnY), indices(j, 0)));

            
         }
      }
      
      
      std::sort(coord_ind.begin() + 1, coord_ind.end(), sortLogs);
      Eigen::MatrixXd lnV (12, 2);
      Eigen::MatrixXi lnF;
      
      if (coord_ind[0].second != indices(l, 0) )
      {
         std::cout << "BAD COORD SORT" << std::endl;
      }
      
      for (int j = 0; j < 12; j++)
      {
         lnV(j, 0) = coord_ind[j].first[0];
         lnV(j, 1) = coord_ind[j].first[1];
      }
      
      //py::print("Running Delaunay for vertex ", l, "flush"=true);
      //std::cout << "Running Delaunay for vertex " << l << std::endl;
      //igl::delaunay_triangulation(lnV, orient2dPredicates, inCirclePredicates, lnF);
      py::object tri = delTri(lnV, "qhull_options"_a="QJ");
      //std::cout << "Ran Delaunay, getting simps" << std::endl;
      //py::object tFaces = tri.attr("simplices")();
      //std::cout << "Ran Delaunay, casting..." << std::endl;
      lnF = tri.attr("simplices").cast<Eigen::MatrixXi>();
      
      if (lnF.rows() == 0) {std::cout<< "ZERO FACES IN TRIANGULATION, #input points = " << lnV.rows() << std::endl;}
      
      for (int j = 0; j < lnF.rows(); j++)
      {
         for (int i = 0; i < 3; i++)
         {
            if ( lnF(j, i) == 0 )
            {
               int i1 = lnF(j, (i+1)%3);
               int i2 = lnF(j, (i+2)%3);
               
               rings[l].push_back ( coord_ind[i1].second );
               rings[l].push_back ( coord_ind[i2].second );
               
               if (coord_ind[i1].second == indices(l, 0) || coord_ind[i2].second == indices(l, 0))
               {
                  std::cout << "Bad triangle: " << lnF(j, 0) << " " << lnF(j, 1) << " " << lnF(j, 2) << std::endl;
               }
             
               break;
            }
         }
      }
      
      if (rings[l].size() == 0) {std::cout << "POST TRIANGULATION RING SIZE IS ZERO" << std::endl;}
      
   }
   
   

   std::vector<Point2D<int>> edges;
   
   for (int l = 0; l < rings.size(); l++)
   {
      std::sort( rings[l].begin(), rings[l].end() );     
      rings[l].erase( std::unique( rings[l].begin(), rings[l].end() ), rings[l].end() );
   
      for (int j = 0; j < rings[l].size (); j++)
      {
         edges.push_back ( Point2D<int> (indices(l, 0), rings[l][j]) );
      }
   }
   
   Eigen::MatrixXd outEdges (2, (int) edges.size ());
   
   for (int l = 0; l < edges.size (); l++)
   {
      outEdges(0, l) = edges[l][0];
      outEdges(1, l) = edges[l][1];
   }
   
   return outEdges;
   
}

Eigen::MatrixXd gradEdges (Eigen::MatrixXd& pos, Eigen::MatrixXd& normals, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& indices, double maxD)
{
   Eigen::Matrix<size_t, Eigen::Dynamic, 1> batch = indices;
   
   batch.setZero();
   

   return gradEdgesBatch (pos, normals, indices, maxD, batch );

}
   

/*
Eigen::MatrixXd precomputeLog(Eigen::MatrixXd& pos, Eigen::Vector3d normal, int sourceInd, Eigen::Matrix<size_t, Eigen::Dynamic, 1> targetInds ) 
{
   Eigen::MatrixXd log (targetInds.rows (), 3); 


   Eigen::Vector3d v0;
   v0(0) = pos(sourceInd, 0), v0(1) = pos(sourceInd, 1), v0(2) = pos(sourceInd, 2);

   Eigen::Matrix3d projN = Eigen::MatrixXd::Identity(3, 3) - normal * normal.transpose();
   
   // Compute log_j(i)
   for (int l = 0; l < targetInds.rows(); l++)
   {
      Eigen::Vector3d v;
      v << pos(targetInds(l, 0), 0), pos(targetInds(l, 0), 1), pos(targetInds(l, 0), 2);
      
      // Compute log direction
      Eigen::Vector3d ln = ( projN * (v - v0) ).normalized ();
     
      log(l, 0) = ln(0);
      log(l, 1) = ln(1);
      log(l, 2) = ln(2);
      
   }
   
   return log;

}

// ===========================================================================================
// // Precomputes log_j (i), gradient_contributions for one ring
// (j, i) j (source) --> i (target) (log_j(i))
// ===========================================================================================

Eigen::MatrixXd precompute (Eigen::MatrixXd& pos, Eigen::MatrixXd& normals, Eigen::Matrix<size_t, Eigen::Dynamic, 2>& edge_index, Eigen::Matrix<size_t, Eigen::Dynamic, 1> degree) 
{

   
   // Store logs 
   Eigen::MatrixXd log(edge_index.rows(), 3); // log (3) + grad_contrib (3)

   size_t index = 0;
   // For each sampled point:
   for (size_t row = 0; row < degree.rows(); row++) 
   {
      int sampleV = edge_index(index, 0);
      
      Eigen::Vector3d nVec;
      nVec << normals(sampleV, 0), normals(sampleV, 1), normals(sampleV, 2);
      
      // Compute logs in one-ring    
      log.block(index, 0, degree(row), 3) = precomputeLog(pos, nVec, sampleV, edge_index.block(index, 1, degree(row), 1));
      index += degree(row);
   }
   
   return log;
}
*/

//=============================================================================
// Compute local parallel transport from faces in one ring to source vertex
// Assume all triangles in one ring are ordered CCW with source vertex in 0th index
//=============================================================================

   //
//=============================================================================
// Precomputes log_j (i)
// (j, i) j (source) --> i (target) (log_j(i))
//=============================================================================
Eigen::MatrixXd getLogs ( Eigen::MatrixXd& spec, Eigen::MatrixXd& pos, int j, Eigen::Vector3d& normal, std::vector<TriangleIndex>& tris, std::vector<int>& targetInds)
{
   Eigen::Matrix3d projN = Eigen::MatrixXd::Identity(3, 3) - normal * normal.transpose();
   
   Eigen::MatrixXd ln( targetInds.size(), 3);
   
   ln.setZero();

   for (int l = 0; l < tris.size(); l++)
   {
      // Compute metric + area
      double l1 = biharmonicDist(tris[l][0], tris[l][1], spec);
      double l2 = biharmonicDist(tris[l][0], tris[l][2], spec);
      double l3 = biharmonicDist(tris[l][1], tris[l][2], spec);
      
      double a = l1 * l1;
      double b = ( l1 * l1 + l2 * l2 - l3 * l3 ) / 2.0;
      double c = l2 * l2;
      
      double det = a * c - b*b;
      
      double s = (l1 + l2 + l3) / 2;
      
      double tArea = std::sqrt ( s * (s - l1) * (s - l2) * (s - l3) );

      
      Eigen::Vector3d v1, v2, v3;
      v1 << pos(tris[l][0], 0), pos(tris[l][0], 1), pos(tris[l][0], 2);
      v2 << pos(tris[l][1], 0), pos(tris[l][1], 1), pos(tris[l][1], 2);
      v3 << pos(tris[l][2], 0), pos(tris[l][2], 1), pos(tris[l][2], 2);
      
      Eigen::Vector3d X = biharmonicDist(tris[l][0], tris[l][1], spec)*(v2 - v1).normalized();
      Eigen::Vector3d Y = biharmonicDist(tris[l][2], tris[l][0], spec)*(v3 - v1).normalized();
      
      for (int i = 0; i <  targetInds.size(); i++)
      {
         // Compute contribution to log direction
         double d1 = biharmonicDist( tris[l][0], targetInds[i] , spec);
         double d2 = biharmonicDist( tris[l][1], targetInds[i], spec);
         double d3 = biharmonicDist( tris[l][2], targetInds[i], spec);
         
         double e1 = d2 - d1;
         double e2 = d3 - d1;
         
         Eigen::Vector2d tGrad;
         tGrad(0) = (c * e1 - b * e2) / det;
         tGrad(1) = (a * e2 - b * e1) / det;
         
         Eigen::Vector3d log3D = -1.0*(tGrad(0) * X  + tGrad(1) * Y);
         
         Eigen::Vector3d lnCont =  tArea * (projN * log3D).normalized();
         
         ln(i, 0) = lnCont(0);
         ln(i, 1) = lnCont(1);
         ln(i, 2) = lnCont(2);
       }      
   }
   
   for (int i = 0; i <  targetInds.size(); i++)
   {
      double mag = std::sqrt( ln(i, 0) * ln(i, 0) + ln(i, 1) * ln(i, 1) + ln(i, 2) * ln(i, 2) );
      
      double dist = biharmonicDist(j, targetInds[i], spec);
      
      if (mag > 1.0e-12)
      {
         ln(i, 0) *= dist/mag;
         ln(i, 1) *= dist/mag;
         ln(i, 2) *= dist/mag;
      }
   }
   
   return ln;
}


// ============================================================================
// Precomputes gradient_contributions
// (j, i) j (source) <-- i (target) (gradient)
// ============================================================================
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> precomputeLogGrad(Eigen::MatrixXd& spec, Eigen::MatrixXd& pos,  int sourceInd, Eigen::Vector3d normal,  std::vector<TriangleIndex>& tris,  std::vector<int>& targetInds ) 
{
   Eigen::MatrixXd logGrad (targetInds.size (), 6); 
   Eigen::MatrixXd selfGrad (1, 3);

   logGrad.setZero();
   selfGrad.setZero();
   
   logGrad.block(0, 0, targetInds.size(), 3) = getLogs(spec, pos,  sourceInd, normal, tris, targetInds);
   
   // First sort one ring   
   Eigen::Matrix3d projN = Eigen::MatrixXd::Identity(3, 3) - normal * normal.transpose();
   Eigen::Matrix3d N = Eigen::MatrixXd::Zero(3, 3);
   N(0, 1) = -normal(2); N(0, 2) = normal(1);
   N(1, 0) = normal(2); N(1, 2) = -normal(0);
   N(2, 0) = -normal(1); N(2, 1) = normal(0);
   
   Eigen::Vector3d log1X;
   log1X << logGrad(0, 0), logGrad(0, 1), logGrad(0, 2);
   
   Eigen::Vector3d log1Y = N * log1X;
  
   
   std::vector<std::pair<double, int>> ringInd;
   
   ringInd.push_back (std::pair<double, int>(0.0, 0));
   
   if (targetInds.size() == 0)
   {
      std::cout<<"TARGET INDS ROWS = 0"<<std::endl;
   }
   // Compute logs + ring theta
   for (int l = 1; l <  targetInds.size(); l++)
   {
      Eigen::Vector3d ln;
      ln << logGrad(l, 0), logGrad(l, 1), logGrad(l, 2);
      
      // Compute log direction
      
      if ( std::isnan(ln(0)) || std::isnan(ln(1)) || std::isnan(ln(2)))
      {
         std::cout << "NAN LOG" << std::endl;
      }
      

      double theta = std::atan2( log1Y.dot( ln ), log1X.dot (ln) );
         
      if (theta < 0) { theta = 2 * PI + theta; }
         
      ringInd.push_back ( std::pair<double, int> (theta, l) );
         
   }
   
   // Order one-ring
   std::sort(ringInd.begin (), ringInd.end (), sortOneRing);
   
   std::vector<std::pair<Eigen::Vector3d, int>> ringVs;
   
   for (int l = 0; l < ringInd.size (); l++)
   {
      int ind = ringInd[l].second;
      Eigen::Vector3d u;
      u << pos(targetInds[ind], 0), pos(targetInds[ind], 1), pos(targetInds[ind], 2);
      
      ringVs.push_back ( std::pair<Eigen::Vector3d, int> (u, ind) );
      
   }   
   
   // Iterate over one ring and accumulate ring vertex contributions to gradient
   double A = 0.0;
   int n_verts = ringVs.size ();
   int num_triangles = 0;
   Eigen::Vector3d v0;
   v0 << pos(sourceInd, 0), pos(sourceInd, 1), pos(sourceInd, 2);
   
   double aTotal = 0.0;
   for (int l = 0; l < ringVs.size (); l++)
   {
      double delTheta; 
      if (l != n_verts - 1) 
      {
         delTheta =  ringInd[(l+1)%n_verts].first - ringInd[l].first;
      }
      else
      {
         delTheta = 2*M_PI - ringInd[l].first;
      }
      
      
      if ( 0 < delTheta && delTheta < M_PI)
      {
         num_triangles++;
         int ind1 = ringVs[l].second;
         int ind2 = ringVs[(l+1)%n_verts].second;
         
         // Compute triangle metric + area
         double l1 = biharmonicDist(sourceInd, targetInds[ind1], spec);
         double l2 = biharmonicDist(sourceInd, targetInds[ind2], spec);
         double l3 = biharmonicDist(targetInds[ind1], targetInds[ind2], spec);
      
         double a = l1 * l1;
         double b = ( l1 * l1 + l2 * l2 - l3 * l3 ) / 2.0;
         double c = l2 * l2;
      
         double det = a * c - b*b;
      
         double s = (l1 + l2 + l3) / 2;
      
         double tArea = std::sqrt ( s * (s - l1) * (s - l2) * (s - l3) );
         
         aTotal += tArea;
          
         Eigen::Vector3d v1 = ringVs[l].first;
         Eigen::Vector3d v2 = ringVs[(l+1)%n_verts].first;

      
         Eigen::Vector3d X = biharmonicDist(sourceInd, targetInds[ind1], spec)*(v1 - v0).normalized();
         Eigen::Vector3d Y = biharmonicDist(sourceInd, targetInds[ind2], spec)*(v2 - v0).normalized();

         Eigen::MatrixXd to3D(3, 2);
         to3D << X(0), Y(0), X(1), Y(1), X(2), Y(2);
         
         Eigen::MatrixXd lift = tArea * projN * to3D;
         //Eigen::MatrixXd lift = tArea * to3D;
         
         Eigen::Vector2d cont0, cont1, cont2;
         
         cont0(0) = -(c * 1.0  - b * 1.0) / det;
         cont0(1) = -(a * 1.0 - b * 1.0) / det;
         
         cont1(0) = (c) / det;
         cont1(1) = (-b) / det;
         
         cont2(0) = (- b) / det;
         cont2(1) = (a) / det;
         
         Eigen::Vector3d c0 = lift * cont0;
         Eigen::Vector3d c1 = lift * cont1;
         Eigen::Vector3d c2 = lift * cont2;
         
         for (int i = 0; i < 3; i++)
         {
            selfGrad(0, i) += c0(i);
            logGrad(ind1, 3 + i) += c1(i);
            logGrad(ind2, 3 + i) += c2(i);
         }
      }
      
   }
   
   selfGrad /= aTotal;
   logGrad.block(0, 3,  targetInds.size(), 3) = logGrad.block(0, 3,  targetInds.size(), 3) / aTotal;

   if (std::isnan(selfGrad.norm()) || std::isnan(logGrad.block(0, 3,  targetInds.size(), 3).norm()) ) 
   { 
   
      std::cout<< "NAN  GRAD ERROR" << std::endl;
      selfGrad.setZero (); 
      logGrad.setZero();
      std::cout<<"RING V SIZE = " << ringVs.size() << std::endl;
      std::cout<< "num tris = " << num_triangles << std::endl;
      std::cout<< "offending node = " <<  sourceInd << ", edges:";
      for (int k = 0; k < ringVs.size(); k++)
      {
         std::cout<< " " << targetInds[ringVs[k].second];
      }
      std::cout<<std::endl;
      
      
   }
   
   if ( std::isnan(logGrad.block(0, 3,  targetInds.size(), 3).norm()) ){ logGrad.block(0, 3,  targetInds.size(), 3).setZero (); }
   
   return std::pair<Eigen::MatrixXd, Eigen::MatrixXd> (logGrad, selfGrad);

}

// ===========================================================================================
// // Precomputes log_j (i), gradient_contributions for one ring
// (j, i) j (source) --> i (target) (log_j(i))
// (j, i) j (source) <-- i (target) (gradient) | Resorted to j (source) --> i (target)
// ===========================================================================================

std::vector<Eigen::MatrixXd> precompute (Eigen::MatrixXd& spec, Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sampleInds, Eigen::MatrixXd& normals, Eigen::Matrix<size_t, Eigen::Dynamic, 2>& edge_index, Eigen::Matrix<size_t, Eigen::Dynamic, 2>& conv_index, Eigen::Matrix<size_t, Eigen::Dynamic, 1> degree) 
{

   
   // Store logs 
   Eigen::MatrixXd logGrad(conv_index.rows(), 6); // log (3) + grad_contrib (3)
   Eigen::MatrixXd nodeGrad(sampleInds.rows(), 3);
   nodeGrad.setZero();

   
   size_t index = 0;
   // For each sampled point:
   for (size_t row = 0; row < degree.rows(); row++) 
   {
      int sampleV = conv_index(index, 0);
      
      Eigen::Vector3d nVec;
      nVec << normals(sampleV, 0), normals(sampleV, 1), normals(sampleV, 2);
      
      int sourceInd = sampleInds(sampleV, 0);
      
      std::vector<int> targetInds;
      
      for (int i = 0; i < degree(row); i++)
      {
         targetInds.push_back ( sampleInds(conv_index(index + i, 1), 0) );
      }
      
      // Construct adjacent faces
      std::vector<TriangleIndex> sourceTris;
      
      for (int l = 0; l < faces.rows(); l++)
      {
         for (int i = 0; i < 3; i++)
         {
            if ( faces(l, i) == sourceInd )
            {
               TriangleIndex tri;
               tri[0] = faces(l, i);
               tri[1] = faces(l, (i+1)%3);
               tri[2] = faces(l, (i+2)%3);
              
               sourceTris.push_back (tri);
               break;
            }
         }
      }
      
      
      std::pair<Eigen::MatrixXd, Eigen::MatrixXd> lnG = precomputeLogGrad(spec, pos, sourceInd, nVec, sourceTris, targetInds);
      
      logGrad.block(index, 0, degree(row), 6) = lnG.first;
      nodeGrad.block(sampleV, 0, 1, 3) = lnG.second;
      
      bool nanFound = false;
      for (int i = 0; i < 3; i++)
      {

         if ( std::isnan(lnG.second(0, i)) )
         {
            nanFound = true;
         }
         for (int k = 0; k < lnG.first.rows(); k++)
         {
            if (std::isnan(lnG.first(k, i)))
            {
                  nanFound = true;
            }
            if (std::isnan(lnG.first(k, 3 + i)))
            {
               nanFound = true;
            }
         }
         
        
      }
      
      if (nanFound)
      {
         std::cout << "NAN FOUND IN LOG PRECOMP but not CAUGHT " << std::endl; fflush(stdout);
      }
      
      index += degree(row);
   }
   
   // Map conv_edge data to global edge data
   Eigen::MatrixXd edgeGrad (edge_index.rows(), 3);
   Eigen::MatrixXd log(edge_index.rows(), 3);
   
   edgeGrad.setZero();
   log.setZero();
   
   #pragma omp paralell 
   for (int l = 0; l < conv_index.rows (); l++)
   {
      int e0 = conv_index(l, 0);
      int e1 = conv_index(l, 1);
      
      bool found = false;
      //bool found_recip = false;
      for (int i = 0; i < edge_index.rows(); i++)
      {

         if (e0 == edge_index(i, 0) && e1 == edge_index(i, 1))
         {
            found = true;
            log.block(i, 0, 1, 3) = logGrad.block(l, 0, 1, 3);
            edgeGrad.block(i, 0, 1, 3) = logGrad.block(l, 3, 1, 3);
            break;
         }
         
      }
      
      if (!found) {std::cout << "ERROR: Could not find conv. edge!" << std::endl; }
      
   }
     
   std::vector<Eigen::MatrixXd> out {log, edgeGrad, nodeGrad};
   return out;
}


// ===================================================================
// Computes weights associated with each sampled vertex
// ===================================================================

Eigen::MatrixXd weights(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sample_points, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labels) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();

  // We use short-time heat diffusion to retrieve geodesic nearest neighbors.
  VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);

  // Set up indices of sampled points to diffuse.
  std::vector<std::tuple<SurfacePoint, double>> points;
  for (size_t row = 0; row < sample_points.rows(); row++) {
    points.emplace_back(SurfacePoint(mesh->vertex(sample_points(row))), labels(row));
  }

  // Solve heat diffusion.
  VertexData<double> scalarExtension = vhmSolver.extendScalar(points);

  // Store the results in an Eigen matrix, which can be accessed as a NumPy array.
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(sample_points.rows(), 1);
  // For each vertex:
  for (size_t row = 0; row < pos.rows(); row++) {
    size_t to_idx = std::lround(scalarExtension[mesh->vertex(row)]);

    // Clamp nearest neighbor index from heat diffusion to range [0, n_vertices]
    if (to_idx >= sample_points.rows()) {
      to_idx = sample_points.rows() - 1;
    } else if (to_idx < 0) {
      to_idx = 0;
    }

    // Add vertex lumped mass to nearest sampled vertex.
    res(to_idx) += geometry->vertexLumpedMassMatrix.coeff(row, row);
  }
  
  geometry->unrequireVertexLumpedMassMatrix();
  geometry->unrequireVertexIndices();

  return res;
}

// ====================================================================================
// Compute HKS at nodes for all input diffusion times
// ====================================================================================
Eigen::MatrixXd hksSamples (Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& nodes, Eigen::MatrixXd diff)
{
   if (diff.rows () < diff.cols () ) {diff.transposeInPlace ();}
   
   Eigen::MatrixXd hks (nodes.rows (),  diff.rows () );
   hks.setZero();
   
   int n_t = diff.rows();
   for (int t = 0; t < diff.rows (); t++)
   {
      for (int l = 0; l < nodes.rows (); l++)
      {   
         for ( int k = 0; k < spec.rows (); k++)
         {
            hks(l,  t) += (double) exp( -spec(k, 0) * diff(t, 0) ) * spec(k, nodes(l, 0) + 1) * spec(k, nodes(l, 0) + 1);
         }
       
      }
   }
   
   return hks;
}

Eigen::MatrixXd wksSamples (Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& nodes, int nSamples)
{
   int sDim = spec.rows();
   
   for (int l = 1; l < sDim; l++)
   {
      if ( std::isnan(std::log(spec(l, 0))) )
      {
         std::cout << "NaN WKS log at " << l << std::endl;
      }
   }
   
   double eDel = (std::log(spec(5, 0))- std::log(spec(sDim-1, 0)))/ nSamples;
  

   
   double sigma = (7 * (std::log(spec(5, 0))- std::log(spec(sDim-1, 0))) / 100);

   if (std::isnan(sigma) ) { std::cout << "BAD SIGMA" << std::endl;
   std::cout<< "log1 = " << std::log(spec(5, 0)) << std::endl;
   std::cout << "log2 = " << std::log(spec(sDim, 0)) << std::endl;
   }
   
   Eigen::MatrixXd wks (nodes.rows(), nSamples);
   wks.setZero();

   double e = 0.0;
   for (int t = 0; t < nSamples; t++)
   {
      e += eDel;
      double Ce = 0.0;
      for ( int k = 1; k < spec.rows (); k++)
      {
       
         double coeff = (double) std::exp( -1.0 * (e - std::log(spec(k, 0))) * (e - std::log(spec(k, 0))) / (2.0 * sigma * sigma));
         
         /*
         if (std::isnan(coeff) || std::isinf(coeff))
         {
            std::cout << "BAD WKS COEFF" << std::endl;
            if (std::isnan(sigma) ) {std::cout << "culprit: sigma" << std::endl;}
         }
         */
         
         Ce += coeff;
         

       
      }
      
      for (int l = 0; l < nodes.rows(); l++)
      {
         wks(l, t) /= Ce;
         
      }
      
   }
   
   return wks;
}

Eigen::MatrixXd specEmbed (Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& nodes, int topN)
{
   
   Eigen::MatrixXd embed (nodes.rows (),  topN );
   embed.setZero();
   

   for (int l = 0; l < nodes.rows (); l++)
   {   
      for ( int k = 1; k <= topN; k++)
      {
         embed(l,  k-1) += (double) spec(k, nodes(l, 0) + 1) / std::sqrt(spec(k, 0));
      }
    
   }
   

   return embed;
}

// ====================================================================================
// SHREC 19 Utils
// ====================================================================================

int findClosestNode( Eigen::MatrixXd& vertices, Eigen::Vector3d x)
{

   double minDist = 1e10;
   int ind;
   
   for (int l = 0; l < vertices.rows() ; l++)
   {


         double d = (vertices.block(l, 0, 1, 3) - x).norm();
      
         if (d < minDist)
         {
            minDist = d;
            ind = l;
         }
   }
   
   return ind;
}


void transferSHRECGT(void)
{
   std::string lrPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC19_lores";
   std::string subPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC19_5K";

   std::string gt0Path = lrPath + "/gt";
   
   std::vector<std::string> gtFiles;
   
   GetFilesInDirectory(gtFiles, gt0Path);
   
   // scan_x_scan_y: v_y : t_x b1 b2 b3
   for (int l = 0; l < gtFiles.size(); l++)
   {
      std::string sourceID = gtFiles[l].substr ( gtFiles[l].find ("n_") + 2, 3 );
      
      std::string targetID = gtFiles[l].substr ( gtFiles[l].rfind("n_") + 2, 3);
      
      Eigen::MatrixXd VS0, VT0, VSd, VTd;
      Eigen::MatrixXi FS0, FT0, FSd, FTd;
      
      // Read oringal meshes
      igl::readOBJ (lrPath + "/models/scan_" + sourceID + ".obj", VS0, FS0);
      igl::readOBJ (lrPath + "/models/scan_" + targetID + ".obj", VT0, FT0);
      
      // Read subsampled meshes
      igl::readPLY (subPath + "/models/scan_" + sourceID + ".ply", VSd, FSd);
      igl::readPLY (subPath + "/models/scan_" + targetID + ".ply", VTd, FTd);     
      
      std::vector<std::pair<int, Point3D<float>>> gt;
      
      // Read gt file
      {
         FILE* gtF = fopen( gtFiles[l].c_str () ,"r");
         char line[200];

         while ( fgets ( line, 200, gtF) )
         {
            std::pair<int, Point3D<float> > index;
            sscanf ( line, "%d %f %f %f", &index.first, &index.second[0], &index.second[1], &index.second[2]);

            gt.push_back (index);
         }

         fclose (gtF);
      }
      
      std::vector<Eigen::Vector3d> sourceGT0;
      
      for (int l = 0; l < gt.size(); l++)
      {  

         int tInd = gt[l].first-1;
         
         Eigen::Vector3d vS = VS0.block(FS0(tInd, 0), 0, 1, 3) * gt[l].second[0] + VS0.block(FS0(tInd, 1), 0, 1, 3) * gt[l].second[1] + VS0.block(FS0(tInd, 2), 0, 1, 3) * gt[l].second[2];

         sourceGT0.push_back(vS);
      }
      
      std::vector<int> gt0;
      
      for (int l = 0; l < VTd.rows(); l++)
      {
         Eigen::Vector3d vTd = VTd.block(l, 0, 1, 3);
         
         int vT0 = findClosestNode(VT0, vTd);
         
         Eigen::Vector3d vS0 = sourceGT0[vT0];
         
         gt0.push_back (findClosestNode(VSd, vS0));
         
      }
      
      std::string gtOut = subPath + "/gt/scan_" + sourceID + ".scan_" + targetID + ".gt.txt";
      
      vector2File(gtOut, gt0);
   }
   
   return;
}

void SHRECGTtoVertex(void)
{
   std::string lrPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC19_lores";
   std::string subPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC19_10K";

   std::string gt0Path = lrPath + "/gt";
   
   std::vector<std::string> gtFiles;
   
   GetFilesInDirectory(gtFiles, gt0Path);
   
   // scan_x_scan_y: v_y : t_x b1 b2 b3
   for (int l = 0; l < gtFiles.size(); l++)
   {
      std::string sourceID = gtFiles[l].substr ( gtFiles[l].find ("n_") + 2, 3 );
      std::string targetID = gtFiles[l].substr ( gtFiles[l].rfind("n_") + 2, 3);
      
      
      Eigen::MatrixXd VS0;
      Eigen::MatrixXi FS0;
      
      // Read oringal meshes
      igl::readOBJ (lrPath + "/models/scan_" + sourceID + ".obj", VS0, FS0);
   
      
      std::vector<std::pair<int, Point3D<float>>> gt;
      
      // Read gt file
      {
         FILE* gtF = fopen( gtFiles[l].c_str () ,"r");
         char line[200];

         while ( fgets ( line, 200, gtF) )
         {
            std::pair<int, Point3D<float> > index;
            sscanf ( line, "%d %f %f %f", &index.first, &index.second[0], &index.second[1], &index.second[2]);

            gt.push_back (index);
         }

         fclose (gtF);
      }
      
      std::vector<int> gtV;
      
      for (int l = 0; l < gt.size(); l++)
      {  

         int tInd = gt[l].first-1;
         
         Eigen::Vector3d v0, v1, v2;
         v0 = VS0.block(FS0(tInd, 0), 0, 1, 3);
         v1 = VS0.block(FS0(tInd, 1), 0, 1, 3);
         v2 = VS0.block(FS0(tInd, 2), 0, 1, 3);
         
         
         Eigen::Vector3d vS = v0 * gt[l].second[0] + v1 * gt[l].second[1] + v2 * gt[l].second[2];
         
         double d0 = (v0 - vS).norm();
         double d1 = (v1 - vS).norm();
         double d2 = (v2 - vS).norm();
         
         if (d0 <= d1 && d0 <= d2)
         {
            gtV.push_back ( FS0(tInd, 0) );
         }
         else if (d1 <= d0 && d1 <= d2)
         {
            gtV.push_back(FS0(tInd, 1));
         }
         else
         {
            gtV.push_back(FS0(tInd, 2));
         }


         
      }

      std::string gtOut = subPath + "/gt/scan_" + sourceID + ".scan_" + targetID + ".gt.txt";
      
      vector2File(gtOut, gtV);
   }
   
   return;
}


void splitsSHREC19(void)
{
   std::string listPath = "/home/tommy/Dropbox/specialMath/Harmonic/ECHONet/V7/data/SHREC19PR/raw/";

   std::vector<std::string> setFiles {"pairs/figure_pairs.txt", "pairs/glove_pairs.txt", "pairs/hand_pairs.txt"};
   
   //std::vector<std::string> setFiles {"pairs/figure_pairs.txt"};
   
   
   std::vector<std::pair<std::string, std::string>> test, train;
   std::srand ( unsigned ( std::time(0) ) );
   
   for (int l = 0; l < setFiles.size(); l++)
   {
      std::cout << listPath + setFiles[l] << std::endl;
      
      FILE* setFile = fopen ( (listPath + setFiles[l]).c_str (), "r");

      char line[300];

      std::vector< std::pair<std::string, std::string> > pairs;
      
      while (fgets( line, 300, setFile) )
      {

         //printf("line = %c%c%c%c%c%c%c|\n", line[0], line[1], line[2], line[3],line[4],line[5], line[6]);
         std::string sID, tID;
         sID.push_back ( line[0] ); sID.push_back ( line[1] ); sID.push_back (line[2]);
         tID.push_back ( line[4] ); tID.push_back ( line[5] ); tID.push_back (line[6]);
         
         pairs.push_back (std::pair<std::string, std::string> (sID, tID));
         
      }
      
      fclose(setFile);
      
      //std::cout << "read file and got pairs: " << (int)pairs.size() << std::endl;
      int nTest = int(std::ceil( pairs.size() * 0.2 ));
      
      std::random_shuffle ( pairs.begin(), pairs.end() );
      
      for (int i = 0; i < pairs.size(); i++)
      {
         if (i < nTest)
         {
            test.push_back (pairs[i]);
         }
         else
         {
            train.push_back(pairs[i]);
         }
      }
      
      
   }
   
   // Save test
   //std::cout<< "num test pairs: " << (int) test.size() << ", num train pairs: " << (int) train.size() << std::endl;
   std::string testFile = listPath + "split/test_pairs.txt";
   std::string trainFile = listPath + "split/train_pairs.txt";
      
   std::ofstream teF;
   teF.open ( testFile.c_str () );

   for ( int i = 0; i < test.size (); i++)
   {
      teF << test[i].first << "," << test[i].second << std::endl;
   }
   
   teF.close ();
   
   std::ofstream trF;
   trF.open ( trainFile.c_str () );

   for ( int i = 0; i < train.size (); i++)
   {
      trF << train[i].first << "," << train[i].second << std::endl;
   }
   
   trF.close ();
   
   return;
   
}

Eigen::MatrixXd readSplit(void)
{
   std::string listPath = "/home/tommy/Dropbox/specialMath/Harmonic/ECHONet/V7/data/SHREC19PR/raw/split/";
   
   std::vector<std::pair<std::string, std::string>> test, train;

   FILE* testFile = fopen ( (listPath + "test_pairs.txt").c_str (), "r");

   char line[300];

   std::vector< std::pair<int, int> > test_pairs;
   
   while (fgets( line, 300, testFile) )
   {

      //printf("line = %c%c%c%c%c%c%c|\n", line[0], line[1], line[2], line[3],line[4],line[5], line[6]);
      
      std::string sID, tID;
      if (std::string(1, line[1]).compare("0") == 0)
      {
         sID.push_back(line[2]);
      }
      else
      {
         sID.push_back(line[1]); sID.push_back(line[2]);
      }
      
      if (std::string(1, line[5]).compare("0") == 0)
      {
         tID.push_back(line[6]);
      }
      else
      {
         tID.push_back(line[5]); tID.push_back(line[6]);
      }

      
      test_pairs.push_back (std::pair<int, int> (std::stoi(sID), std::stoi(tID)));
      
   }
   
   fclose(testFile);


   FILE* trainFile = fopen ( (listPath + "train_pairs.txt").c_str (), "r");


   std::vector< std::pair<int, int> > train_pairs;
   
   while (fgets( line, 300, trainFile) )
   {

      //printf("line = %c%c%c%c%c%c%c|\n", line[0], line[1], line[2], line[3],line[4],line[5], line[6]);
      
      std::string sID, tID;
      
      if (std::string(1, line[1]).compare("0") == 0)
      {
         sID.push_back(line[2]);
      }
      else
      {
         sID.push_back(line[1]); sID.push_back(line[2]);
      }
      
      if (std::string(1, line[5]).compare("0") == 0)
      {
         tID.push_back(line[6]);
      }
      else
      {
         tID.push_back(line[5]); tID.push_back(line[6]);
      }

      
      train_pairs.push_back (std::pair<int, int> (std::stoi(sID), std::stoi(tID)));
      
   }
   
   fclose(trainFile);
   
   Eigen::MatrixXd pairsM(test_pairs.size() + train_pairs.size(), 3);
   pairsM.setZero();
   
   for (int l = 0; l < train_pairs.size(); l++)
   {
      pairsM(l, 0) = train_pairs[l].first;
      pairsM(l, 1) = train_pairs[l].second;
      
   }
   
   for (int l = 0; l < test_pairs.size(); l++)
   {
      pairsM(l + train_pairs.size(), 0) = test_pairs[l].first;
      pairsM(l + train_pairs.size(), 1) = test_pairs[l].second;
      pairsM(l + train_pairs.size(), 2) = 1;
   }

   
   return pairsM;
}
   

std::vector<Eigen::MatrixXd> sampleGeoMat( Eigen::MatrixXd& posS, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& facesS, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samplesTS, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samplesS)
{
   std::tie(mesh, geometry) = loadMesh_np(posS, facesS);
   
   geometry->requireVertexIndices();



   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   

   std::vector<std::pair<int, int>> pairs;
   std::vector<double> dist;

   Eigen::MatrixXd geoDense(samplesTS.rows(), samplesS.rows());
   geoDense.setZero();
   
   //std::cout<< "beginning distance calc" << std::endl;
   for (int i = 0; i < samplesTS.rows(); i++)
   {
   

      Vertex t = mesh->vertex( samplesTS(i, 0) );
            
      //std::cout << "init solver" << std::endl;
      VertexData<double> geo = heatSolver.computeDistance(t);
      //std::cout << "done" << std::endl;
      for (int j = 0; j < samplesS.rows(); j++)
      {

         Vertex p = mesh->vertex( samplesS(j, 0) );
      
         pairs.push_back(std::pair<int, int>(i, j));
         double d = geo[p];
         dist.push_back ( d);
         geoDense(i, j) = d;
      
      }
   }
      
      //std::cout << "got all distances" << std::endl;
   Eigen::MatrixXd geoMat(pairs.size(), 3);
   geoMat.setZero();
   
   
   for (int l = 0; l < pairs.size(); l++)
   {
      geoMat(l, 0) = pairs[l].first;
      geoMat(l, 1) = pairs[l].second;
      geoMat(l, 2) = dist[l];
   }
   
   std::vector<Eigen::MatrixXd> out;
   out.push_back (geoDense);
   out.push_back (geoMat);
   
   geometry->unrequireVertexIndices();

   return out;
   
   
}

Eigen::MatrixXd getNullPairs( Eigen::MatrixXd& posS, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& facesS, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samplesTS, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samplesS, double rMin)
{
   std::tie(mesh, geometry) = loadMesh_np(posS, facesS);
   
   geometry->requireVertexIndices();

   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
   std::vector<std::pair<int, int>> pairs;

   for (int i = 0; i < samplesTS.rows(); i++)
   {
   
      Vertex t = mesh->vertex( samplesTS(i, 0) );
            
      VertexData<double> geo = heatSolver.computeDistance(t);

      for (int j = 0; j < samplesS.rows(); j++)
      {

         Vertex p = mesh->vertex( samplesS(j, 0) );
      
         if (geo[p] > rMin)
         {
            pairs.push_back(std::pair<int, int>(i, j));
         }
         
      }
   }
      
      //std::cout << "got all distances" << std::endl;
   Eigen::MatrixXd nullPairs(pairs.size(), 2);
   nullPairs.setZero();
   
   
   for (int l = 0; l < pairs.size(); l++)
   {
      nullPairs(l, 0) = pairs[l].first;
      nullPairs(l, 1) = pairs[l].second;

   }
   
   return nullPairs;
   
   
}




/*
Eigen::MatrixXd getMatchInd( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples,  double rM)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, facesS);
   
   geometry->requireVertexIndices();

   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
   std::vector<int> valid;

   for (int i = 0; i < samples.rows(); i++)
   {
   
      Vertex t = mesh->vertex( samples(i, 0) );
            
      VertexData<double> geo = heatSolver.computeDistance(t);

      for (int j = 0; j < pos.rows(); j++)
      {

         Vertex p = mesh->vertex( j );
      
         if (geo[p] < rM)
         {
            valid.push_back (j);
         }
         
      }
   }
      
      //std::cout << "got all distances" << std::endl;
   std::sort( valid.begin(), valid.end() );
   valid.erase( std::unique( valid.begin(), valid.end() ), valid.end() );

   Eigen::MatrixXd validV(valid.size());
   
   for (int l = 0; l < valid.size(); l++)
   {
      validV(l, 0) = valid[l];

   }
   
   return validV;
   
   
}
*/

Eigen::MatrixXd getNearestDes( Eigen::MatrixXd& desS,  Eigen::MatrixXd& desT, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samplesS)
{

   Eigen::MatrixXd pNearest(samplesS.rows(), 1);
   pNearest.setZero();
   
   int desDim = desT.cols();
   
   for (int l = 0; l < desT.rows(); l++)
   {
      
      double minDist = 1e12;
      int minI;
      
      for (int i = 0; i < desS.rows(); i++)
      {
         double d = (desT.block(l, 0, 1, desDim) - desS.block(i, 0, 1, desDim)).norm();
         
         if (d < minDist)
         {
            minDist = d;
            minI = i;
         }
      }
      
      pNearest(l, 0) = samplesS(minI, 0);
   }
   
   return pNearest;
      
}

Eigen::MatrixXd getGeoError( Eigen::MatrixXd& pos,  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 2>& compInd)
{

   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
   Eigen::MatrixXd error(compInd.rows(), 1);
   error.setZero();
   

   for (int i = 0; i < compInd.rows(); i++)
   {
   
      Vertex t = mesh->vertex( compInd(i, 0) );
      Vertex p = mesh->vertex( compInd(i, 1) );
            
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      error(i, 0) = geo[p];

   }
   
   return error;
      
}


Eigen::MatrixXd getAdjacentDist( Eigen::MatrixXd& pos,  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sample_ind)
{

   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
   Eigen::MatrixXd dist(sample_ind.rows(), sample_ind.rows());
   dist.setZero();
   
   for (int i = 0; i < sample_ind.rows(); i++)
   {
   
      Vertex t = mesh->vertex( sample_ind(i, 0) );
            
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      for (int j = 0; j < sample_ind.rows(); j++)
      {
         Vertex p = mesh->vertex( sample_ind(j, 0) );
         dist(i, j) = geo[p]; 
      }

   }
   
   return dist;
      
}



// ====================================================================================
// SHREC 20 Utils
// ====================================================================================
std::vector<int> labelsToNearest(Eigen::MatrixXd& pos, Eigen::MatrixXi& faces, std::vector<int>& labels, std::vector<std::pair<int, Point3D<float>>>& coords)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();
   geometry->requireFaceIndices();
   
   // Geodesic Nearest Neighbors
   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   // Geodesic nearest nieghbors
   //VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);
   
   //std::vector<std::tuple<SurfacePoint, double>> points;
   
   std::vector<SurfacePoint> points;
   
   for (int l = 0; l < labels.size(); l++)
   {

      Face f = mesh->face( (int) coords[l].first );
      Vector3 bary;
      bary[0] = coords[l].second[0];
      bary[1] = coords[l].second[1];
      bary[2] = coords[l].second[2];
      
      points.push_back(SurfacePoint(f, bary));
      
   }
   
   std::vector<int> denseL;
      
   for (int l = 0; l < pos.rows(); l++)
   {
      Vertex t = mesh->vertex( l );
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      int minL = 0;
      double minD = 1.0e12;
      
      for (int i = 0; i < points.size(); i++)
      {
         double d = points[i].interpolate(geo);
         
         if (d < minD)
         {
            minD = d;
            minL = i;
         }
      }
      
      denseL.push_back ( labels[minL] );
   }
   
   geometry->unrequireVertexIndices();
   geometry->unrequireFaceIndices();
   
   return denseL;
   
}

std::vector<int> labelsToNearest(Eigen::MatrixXd& pos, Eigen::MatrixXi& faces, std::vector<int>& labels, std::vector<int>& indices)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   
   // Geodesic Nearest Neighbors
   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   // Geodesic nearest nieghbors
   //VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);
   
   //std::vector<std::tuple<SurfacePoint, double>> points;
   
   std::vector<Vertex> points;
   
   for (int l = 0; l < labels.size(); l++)
   {

      points.push_back ( mesh->vertex( (int) indices[l] ) );

   }
   
   std::vector<int> denseL;
      
   for (int l = 0; l < pos.rows(); l++)
   {
      Vertex t = mesh->vertex( l );
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      int minL = 0;
      double minD = 1.0e12;
      
      for (int i = 0; i < points.size(); i++)
      {
         double d = geo[points[i]];
         
         if (d < minD)
         {
            minD = d;
            minL = i;
         }
      }
      
      denseL.push_back ( labels[minL] );
   }
   
   geometry->unrequireVertexIndices();
   
   return denseL;
   
}

/*
Eigen::MatrixXd samplesToNearest(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   
   // Geodesic Nearest Neighbors
   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   // Geodesic nearest nieghbors
   //VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);
   
   //std::vector<std::tuple<SurfacePoint, double>> points;
   

   Eigen::MatrixXd denseL(pos.rows(), 1);
   denseL.setZero();
   
   std::vector<double> minDist;
   minDist.resize(pos.rows(), 1e12);
   
   for (int l = 0; l < samples.size(); l++)
   {
      VertexData<double> geo = heatSolver.computeDistance(mesh->vertex( (int) samples(l , 0) ));
      
      for (int i = 0; i < pos.rows(); i++)
      {
         Vertex p = mesh->vertex ( (int) i);
         
         double d = geo[p];
         
         if (minDist[i] > d)
         {
            denseL(i, 0) = l;
            minDist[i] = d;
         }
      }
   }


   geometry->unrequireVertexIndices();
   
   return denseL;
   
}
*/
Eigen::MatrixXd samplesToNearest(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   
   // Geodesic Nearest Neighbors
   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   // Geodesic nearest nieghbors
   //VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);
   
   //std::vector<std::tuple<SurfacePoint, double>> points;
   
   std::vector<Vertex> points;
   
   for (int l = 0; l < samples.size(); l++)
   {

      points.push_back ( mesh->vertex( (int) samples[l] ) );

   }
   
   Eigen::MatrixXd denseL(pos.rows(), 1);
      
   for (int l = 0; l < pos.rows(); l++)
   {
      Vertex t = mesh->vertex( l );
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      int minL = 0;
      double minD = 1.0e12;
      
      for (int i = 0; i < points.size(); i++)
      {
         double d = geo[points[i]];
         
         if (d < minD)
         {
            minD = d;
            minL = i;
         }
      }
      
      denseL(l, 0) = minL;
   }
   
   geometry->unrequireVertexIndices();
   
   return denseL;
   
}

void denseGTS20b(void)
{

   std::string lrPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC20b_lores/models/";
   std::string gt0Path = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC20b_lores/gt";
   std::string gtDPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC20b_lores/gt_nearest/";

   
   std::vector<std::string> gtFiles;
   
   GetFilesInDirectory(gtFiles, gt0Path);
   
   // scan_x_scan_y: v_y : t_x b1 b2 b3
   for (int l = 0; l < gtFiles.size(); l++)
   {
      std::string modelID = gtFiles[l].substr ( gtFiles[l].rfind("/") + 1, gtFiles[l].length());
      modelID = modelID.substr(0, modelID.find ("."));
      
      Eigen::MatrixXd V;
      Eigen::MatrixXi F;
            //std::cout<< lrPath +  modelID + ".obj" << std::endl;
            //std::cout<< " " << std::endl;
      // Read oringal meshes
      igl::readOBJ (lrPath +  modelID + ".obj", V, F);

      //std::cout<< lrPath +  modelID + ".obj" << std::endl;
      std::vector<int> labels;
      std::vector<std::pair<int, Point3D<float>>> gt;
      
      // Read gt file
      {
         FILE* gtF = fopen( gtFiles[l].c_str () ,"r");
         char line[200];

         while ( fgets ( line, 200, gtF) )
         {
            std::pair<int, Point3D<float> > index;
            int lab;
            sscanf ( line, "%d %d %f %f %f", &lab, &index.first, &index.second[0], &index.second[1], &index.second[2]);

            labels.push_back(lab);
            index.first = index.first -1;
            gt.push_back (index);
         }

         fclose (gtF);
      }
      
      std::vector<int> denseL = labelsToNearest(V, F, labels, gt);
      
      std::string gtOut = gtDPath + modelID + ".gt.txt";
      
      vector2File(gtOut, denseL);
   }
   
   return;
}

void denseGTS20a(void)
{

   std::string lrPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC20a_lores/models/";
   std::string gt0Path = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC20a_lores/gt";
   std::string gtDPath = "/home/tommy/Dropbox/specialMath/Harmonic/Data/external/SHREC20a_lores/gt_nearest/";

   
   std::vector<std::string> gtFiles;
   
   GetFilesInDirectory(gtFiles, gt0Path);
   
   // scan_x_scan_y: v_y : t_x b1 b2 b3
   for (int l = 0; l < gtFiles.size(); l++)
   {
      std::string modelID = gtFiles[l].substr ( gtFiles[l].rfind("/") + 1, gtFiles[l].length());
      modelID = modelID.substr(0, modelID.find ("."));
      
      Eigen::MatrixXd V;
      Eigen::MatrixXi F;
            //std::cout<< lrPath +  modelID + ".obj" << std::endl;
            //std::cout<< " " << std::endl;
      // Read oringal meshes
      igl::readOBJ (lrPath +  modelID + ".obj", V, F);

      //std::cout<< lrPath +  modelID + ".obj" << std::endl;
      std::vector<int> labels;
      std::vector<int> gt;
      
      // Read gt file
      {
         FILE* gtF = fopen( gtFiles[l].c_str () ,"r");
         char line[200];

         while ( fgets ( line, 200, gtF) )
         {
            int index;
            int lab;
            sscanf ( line, "%d %d", &lab, &index);

            labels.push_back(lab);
            index = index - 1;
            gt.push_back (index);
         }

         fclose (gtF);
      }
      
      std::vector<int> denseL = labelsToNearest(V, F, labels, gt);
      
      std::string gtOut = gtDPath + modelID + ".gt.txt";
      
      vector2File(gtOut, denseL);
   }
   
   return;
}

Eigen::MatrixXd computePR(Eigen::MatrixXd& dist, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& matches)
{
   int nMatches = matches.rows();
   
   std::vector<std::pair<double, int>> info;
   
   std::vector<int> mtch;
   
   for (int l = 0; l < matches.rows(); l++)
   {
      mtch.push_back ( (int) matches[l]);
   }
   
   for (int l = 0; l < dist.rows(); l++)
   {
      if ( std::find(mtch.begin(), mtch.end(), l) != mtch.end() )
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), 1 ) );
      }
      else
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), 0 ) );
      }
      
   }
   
   std::sort(info.begin(), info.end(), sortOneRing);
   
   std::vector<double> pre;
   std::vector<double> re;

   int mCount = 0;
   for (int l = 0; l < dist.rows(); l++)
   {
      
      mCount += info[l].second;
      
      pre.push_back( ( (double) mCount ) / (l+1) );
      re.push_back ( ( (double) mCount ) / nMatches);
      
   }
   
   
   Eigen::MatrixXd PR (2, pre.size());
   PR.setZero();
   
   for (int l = 0; l < pre.size(); l++)
   {
      PR(0, l) = pre[l];
      PR(1, l) = re[l];
   }
   
   return PR;
}

Eigen::MatrixXd firstValidMatches(Eigen::MatrixXd& dist, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& matches)
{
   int nMatches = matches.rows();
   
   std::vector<std::pair<double, int>> info;
   
   std::vector<int> mtch;
   
   for (int l = 0; l < matches.rows(); l++)
   {
      mtch.push_back ( (int) matches[l]);
   }
   
   for (int l = 0; l < dist.rows(); l++)
   {
      if ( std::find(mtch.begin(), mtch.end(), l) != mtch.end() )
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), l ) );
      }
      else
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), -1 ) );
      }
      
   }
   
   std::sort(info.begin(), info.end(), sortOneRing);


   std::vector<int> valid;
   
   for (int l = 0; l < dist.rows(); l++)
   {
      
      if (info[l].second != -1)
      {
         valid.push_back ( info[l].second );
      }
      else
      {
         break;
      }
   }

   
   if (valid.size() == 0)
   {
      Eigen::MatrixXd validM(1, 1);
      validM(0, 0) = -1;
      return validM;
   }
   else
   {
      Eigen::MatrixXd validM(valid.size(), 1);
      
      for (int l = 0; l < valid.size(); l++)
      {
         validM(l, 0) = valid[l];
      }
      return validM;
   }


}

int firstValidMatch(Eigen::MatrixXd& dist, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& matches)
{
   int nMatches = matches.rows();
   
   std::vector<std::pair<double, int>> info;
   
   std::vector<int> mtch;
   
   for (int l = 0; l < matches.rows(); l++)
   {
      mtch.push_back ( (int) matches[l]);
   }
   
   for (int l = 0; l < dist.rows(); l++)
   {
      if ( std::find(mtch.begin(), mtch.end(), l) != mtch.end() )
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), l ) );
      }
      else
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), -1 ) );
      }
      
   }
   
   std::sort(info.begin(), info.end(), sortOneRing);



   
   if (info[0].second == -1)
   {
      return -1;
   }
   else
   {
      return info[0].second;
   }


}

double hitRate(Eigen::MatrixXd& dist, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& matches)
{
   int nMatches = matches.rows();
   
   std::vector<std::pair<double, int>> info;
   
   std::vector<int> mtch;
   
   for (int l = 0; l < matches.rows(); l++)
   {
      mtch.push_back ( (int) matches[l]);
   }
   
   for (int l = 0; l < dist.rows(); l++)
   {
      if ( std::find(mtch.begin(), mtch.end(), l) != mtch.end() )
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), l ) );
      }
      else
      {
         info.push_back ( std::pair<double, int> ( dist(l, 0), -1 ) );
      }
      
   }
   
   std::sort(info.begin(), info.end(), sortOneRing);


   int nHits = 0;
   
   for (int l = 0; l < mtch.size(); l++)
   {
      if (info[l].second != -1)
      {
         nHits ++;
      }
   }
   
   return ( (double) nHits ) / nMatches; 
   

}



      
//==========================================================================================
// SHREC Connectivity Dataset
//==========================================================================================


void splitsSHRECConn(void)
{
   std::string listPath = "/home/tommy/Dropbox/specialMath/Harmonic/ECHONet/V7/data/SHRECConn/raw/gt";

   std::string splitPath = "/home/tommy/Dropbox/specialMath/Harmonic/ECHONet/V7/data/SHRECConn/raw/split";
   
   std::vector<std::string> gtFiles;
   
   GetFilesInDirectory(gtFiles, listPath);
   
   
   std::vector<std::pair<std::string, std::string>> pairs, test, train;
   std::srand ( unsigned ( std::time(0) ) );
   
   for (int l = 0; l < gtFiles.size(); l++)
   {
      std::string ID = gtFiles[l].substr(gtFiles[l].rfind("/") + 1, gtFiles[l].length() );
      
      ID = ID.substr(0, ID.find("."));
      
      std::string p0 = ID.substr(0, ID.find("_"));
      std::string p1 = ID.substr(ID.find("_") + 1, ID.length());
      
      pairs.push_back( std::pair<std::string, std::string>(p0, p1));
   }
   
   int nTest = int(std::ceil( pairs.size() * 0.2 ));
      
   std::random_shuffle ( pairs.begin(), pairs.end() );
      
   for (int i = 0; i < pairs.size(); i++)
   {
      if (i < nTest)
      {
         test.push_back (pairs[i]);
      }
      else
      {
         train.push_back(pairs[i]);
      }
   }

   // Save test
   //std::cout<< "num test pairs: " << (int) test.size() << ", num train pairs: " << (int) train.size() << std::endl;
   std::string testFile = splitPath + "/test_pairs.txt";
   std::string trainFile = splitPath + "/train_pairs.txt";
      
   std::ofstream teF;
   teF.open ( testFile.c_str () );

   for ( int i = 0; i < test.size (); i++)
   {
      teF << test[i].first << " " << test[i].second << std::endl;
   }
   
   teF.close ();
   
   std::ofstream trF;
   trF.open ( trainFile.c_str () );

   for ( int i = 0; i < train.size (); i++)
   {
      trF << train[i].first << " " << train[i].second << std::endl;
   }
   
   trF.close ();
   
   return;
   
}

Eigen::MatrixXd readSplitConn(void)
{
   std::string listPath = "/home/tommy/Dropbox/specialMath/Harmonic/ECHONet/V7/data/SHRECConn/raw/split/";
   

  

   std::string testFile = listPath + "test_pairs.txt";
   
   std::vector<std::vector<int>> test_pairs;
   
   file2Vectors(testFile, test_pairs);
   
   std::string trainFile = listPath + "train_pairs.txt";
   std::vector<std::vector<int>> train_pairs;
   file2Vectors(trainFile, train_pairs);
   
   
   
   Eigen::MatrixXd pairsM(test_pairs.size() + train_pairs.size(), 3);
   pairsM.setZero();
   
   for (int l = 0; l < train_pairs.size(); l++)
   {
      pairsM(l, 0) = train_pairs[l][0];
      pairsM(l, 1) = train_pairs[l][1];
      
   }
   
   for (int l = 0; l < test_pairs.size(); l++)
   {
      pairsM(l + train_pairs.size(), 0) = test_pairs[l][0];
      pairsM(l + train_pairs.size(), 1) = test_pairs[l][1];
      pairsM(l + train_pairs.size(), 2) = 1;
   }

   
   return pairsM;
}


// =======================================================================================
// == Visualization 
// =======================================================================================

RegularGrid<float, 2> resampleDes (Eigen::MatrixXd& des, int outH, int outW)
{
   RegularGrid<float, 2> D;
   
   D.resize(des.rows(), 1);
   
   int inH = des.rows();
   //int inW = des.cols();
   
   for (int i = 0; i < des.rows(); i++)
   {
      //for (int j = 0; j < des.cols(); j++)
      //{
         D(i, 0) = (float) des(i, 0);
      //}
   }
   
   
   RegularGrid<float, 2> F;
   F.resize(outH, outW);
   
   for (unsigned int i = 0; i < outH; i++)
   {
      float x = (float)( (((double)i)/(outH - 1)) * (inH-1) );
      float y = 0.f;
      for (unsigned int j = 0; j < outW; j++)
      {
         //float x = (float)( (((double)i)/(outH - 1)) * (inH-1) );
         //float y = (float)( (((double)j)/(outW - 1)) * (inW-1) );
         
         F(i, j) = D(x, y);
      }
   }
   
   return F;
}
      
      
RegularGrid< float , 2 > TransposeSignal( const RegularGrid< float , 2 > &in )
{
    RegularGrid< float , 2 > out;
    out.resize( in.res(1) , in.res(0) );
    for( unsigned int i=0 ; i<in.res(0) ; i++ ) for( unsigned int j=0 ; j<in.res(1) ; j++ ) out(j,i) = in(i,j);
    return out;
}
   
      

Eigen::MatrixXd vizDescriptor( Eigen::MatrixXd& des, double dValMax, double dValMin, int outRes, std::string fileName)
{

   int outW = outRes;
   int outH = int(outW /2);

   RegularGrid<float, 2> F = TransposeSignal(resampleDes(des, outH, outW));
   
  
   //std::cout << "resampled des" << std::endl;
   unsigned char *pixels = new unsigned char[ F.resolution()*3 ];

   int count = 0;
   double dev = 0;
   double sumD = 0;
   for( int i=0 ; i<F.resolution() ; i++ ) 
   {
      if( F[i]<std::numeric_limits< float >::infinity() )
      { 
         sumD += F[i];
         count++;
         
         dev += F[i] * F[i];
         
      }
   }
         

   
   dev = sqrt( dev/count );
   double hue , saturation;
   if( dValMax<=0 )
   {
       saturation = 0.;
       hue = 0;

   }
   else
   {
       saturation = 1.;
       hue = 4.*M_PI/3. * (dev - dValMin) / (dValMax - dValMin);
       //hue = 4 * M_PI / 3 * dev / dValMax;
   }
   
   for( int i=0 ; i<F.resolution() ; i++ )
   {
       if( F[i]==std::numeric_limits< float >::infinity() ) pixels[3*i+0] = pixels[3*i+1] = pixels[3*i+2] = 255;
       else
       {
           double d = std::max( 0. , std::min( F[i] / (3.*dev) , 1. ) );
           Point3D< double > rgb , hsv( hue , saturation , d );
           Miscellany::HSVtoRGB( &hsv[0] , &rgb[0] );
           for( int c=0 ; c<3 ; c++ ) pixels[3*i+c] = (unsigned char)(int)floor( rgb[c] * 255 );
       }
   }
   
   ImageWriter::Write( fileName.c_str() , pixels , outH, outW , 3 );
   delete[] pixels;
   
   Eigen::MatrixXd outQ(2, 1);
   
   outQ(0, 0) = dev;
   outQ(1, 0) = sumD;
   
   return outQ;
}



Eigen::MatrixXd getNearestSamples( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sel)
{
   Eigen::MatrixXd ind (sel.rows(), 1);
   

   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   
   // Geodesic Nearest Neighbors
   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);

   std::vector<Vertex> points;
   
   for (int l = 0; l < samples.size(); l++)
   {

      points.push_back ( mesh->vertex( (int) samples[l] ) );

   }
   
      
   for (int l = 0; l < sel.rows(); l++)
   {
      Vertex t = mesh->vertex( sel(l, 0) );
      VertexData<double> geo = heatSolver.computeDistance(t);
      
      int minL = 0;
      double minD = 1.0e12;
      
      for (int i = 0; i < points.size(); i++)
      {
         double d = geo[points[i]];
         
         if (d < minD)
         {
            minD = d;
            minL = i;
         }
      }
      
      ind(l, 0) = minL;
   }
   
   geometry->unrequireVertexIndices();
   
   return ind;
}

Eigen::MatrixXd extendPrecision( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples, Eigen::MatrixXd& values, double time = 1.0, bool nrm = false)
{

   // Load mesh
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);

   geometry->requireVertexIndices();

   // Sclar extensions solver
   VectorHeatMethodSolver vhmSolver(*geometry, time);
  

   std::vector<std::tuple<Vertex, double>> points;
   
   for (int l = 0; l < samples.rows(); l++)
   {
      Vertex v = mesh->vertex( (int) samples(l, 0) );
      double val = values(l, 0);
      
      points.emplace_back(v, val);
   }
   
   VertexData<double> ext = vhmSolver.extendScalar(points);
   
   
   Eigen::MatrixXd pre (pos.rows(), 1);
   
   for (int l = 0; l < pos.rows(); l++)
   {
      Vertex t = mesh->vertex( l );
      
      double val = ext[t];
      
      pre(l, 0) = val;
   }
   
   if (nrm)
   {
      return pre / pre.maxCoeff();
   }
   else
   {
      return pre;
   }
}


Eigen::MatrixXd extendColors( Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples, Eigen::MatrixXd& values, double time = 1.0)
{

   // Load mesh
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);

   geometry->requireVertexIndices();

   // Sclar extensions solver
   VectorHeatMethodSolver vhmSolver(*geometry, time);
  
   Eigen::MatrixXd color (pos.rows(), 3);
   
   
   for (int j = 0; j < 3; j++)
   {
      std::vector<std::tuple<Vertex, double>> points;
      
      for (int l = 0; l < samples.rows(); l++)
      {
         Vertex v = mesh->vertex( (int) samples(l, 0) );
         double val = values(l, j);
         
         points.emplace_back(v, val);
      }
      
      VertexData<double> ext = vhmSolver.extendScalar(points);
      
      for (int l = 0; l < pos.rows(); l++)
      {
         Vertex t = mesh->vertex( l );
      
         double val = ext[t];
      
         if (val > 1.0)
         {
            val = 1.0;
         }
         else if (val < 0.0)
         {
            val = 0.0;
         }
         
         color(l, j) = val;
      }
   }
   
   return color;
}



Eigen::MatrixXd colorPrecision( Eigen::MatrixXd& pre, double mag=1.0)
{

   double saturation = 1.;

   
   Eigen::MatrixXd cMat (pre.rows(), 3);
   
   for (int l = 0; l < pre.rows(); l++)
   {
      double hue = 4.*M_PI/3. * pre(l, 0);
      
      Point3D< double > rgb , hsv( hue , saturation , mag);
      
      Miscellany::HSVtoRGB( &hsv[0] , &rgb[0] );
      
      for (int i = 0; i < 3; i++)
      {
         cMat(l, i) = rgb[i];
      }
   }
   
   return cMat;
}


Vector3 getBarycentric (Vector2 P, Vector2 v1, Vector2 v2, Vector2 v3)
{
   double detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * ( v1[1] - v3[1]);

   Vector3 coords;

   coords[0] = ( (v2[1] - v3[1]) * (P[0] - v3[0]) + (v3[0] - v2[0]) * ( P[1] - v3[1]) ) / detT;
   coords[1] = ( (v3[1] - v1[1]) * (P[0] - v3[0]) + (v1[0] - v3[0]) * ( P[1] - v3[1]) ) / detT;
   coords[2] = 1.0 - coords[0] - coords[1];

   return coords;
}

Eigen::MatrixXd constructTriUpsample(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   
   // Geodesic Nearest Neighbors
   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
   VectorHeatMethodSolver vhmSolver(*geometry, 0.01);
   
   // Geodesic nearest nieghbors
   //VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);
   
   //std::vector<std::tuple<SurfacePoint, double>> points;
   
   std::vector<int> sInds;
   
   for (int l = 0; l < samples.rows(); l++)
   {
      sInds.push_back(samples(l, 0));
   }
   
   std::vector<std::vector<std::pair<double, int>>> dist;
   dist.resize(pos.rows());
   for (int l = 0; l < pos.rows(); l++)
   {
      dist[l].resize(samples.rows());
   }
   
   std::vector<Vertex> points;
   
   for (int l = 0; l < samples.size(); l++)
   {

      points.push_back ( mesh->vertex( (int) samples[l] ) );

   }
   
   for (int l = 0; l < points.size(); l++)
   {
      VertexData<double> geo = heatSolver.computeDistance(points[l]);
      
      for (int i = 0; i < pos.rows(); i++)
      {
         dist[i][l] = std::pair<double, int>(geo[mesh->vertex( i )], l);
      }
   }
   
   
   Eigen::MatrixXd interp (pos.rows(), 6);
   interp.setZero();
   double eps = 1.0e-8;
   for (int l = 0; l < pos.rows(); l++)
   {
      auto it = std::find(sInds.begin(), sInds.end(), l);
      if ( it != sInds.end() )
      {
         interp(l, 0) = (int)(it - sInds.end());
         interp(l, 3) = 1.0;
      }
      else
      {
         std::vector<std::pair<double, int>> sNear = dist[l];
         std::sort(sNear.begin(), sNear.end(), sortOneRing);
         
         Eigen::MatrixXd lnV (12, 2);
         Eigen::MatrixXi lnF;
         
         VertexData<Vector2> logM = vhmSolver.computeLogMap(mesh->vertex(l));
         
         for (int j = 0; j < 12; j++)
         {
            Vector2 coord = logM[points[sNear[j].second]];
            
            lnV(j, 0) = coord[0];
            lnV(j, 1) = coord[1];
      
         }

         py::object tri = delTri(lnV, "qhull_options"_a="QJ");
         lnF = tri.attr("simplices").cast<Eigen::MatrixXi>();

         bool found = false;
         
         Vector2 p0;
         p0[0] = p0[1] = 0.0;
         
         for (int j = 0; j < lnF.rows(); j++)
         {
            Vector2 v1, v2, v3;
            
            v1[0] = lnV(lnF(j, 0), 0);
            v1[1] = lnV(lnF(j, 0), 1);
            
            v2[0] = lnV(lnF(j, 1), 0);
            v2[1] = lnV(lnF(j, 1), 1);
            
            v3[0] = lnV(lnF(j, 2), 0);
            v3[1] = lnV(lnF(j, 2), 1);
            
            Vector3 baryC = getBarycentric(p0, v1, v2, v3);
            
            if ( baryC[0] >= -eps && baryC[0] <= 1.0 + eps && baryC[1] >= -eps && baryC[1] <= 1 + eps && baryC[2] >= -eps && baryC[2] <= 1 + eps)
            {
               found = true;
               
               interp(l, 0) = sNear[lnF(j, 0)].second;
               interp(l, 1) = sNear[lnF(j, 1)].second;
               interp(l, 2) = sNear[lnF(j, 2)].second;
               interp(l, 3) = baryC[0];
               interp(l, 4) = baryC[1];
               interp(l, 5) = baryC[2];
               
               break;
            }

         }
         
         if (!found)
         {
            interp(l, 0) = sNear[0].second;
            interp(l, 3) = 1.0;
         }
         
      }
   }
   
   geometry->unrequireVertexIndices();
   return interp;
   
}

      
   
   
   
  

PYBIND11_MODULE(trimesh, m) {
    m.doc() = R"pbdoc(
        Extended Convolution precomputation module.
        -----------------------

        .. currentmodule:: precomputation

        .. autosummary::
           :toctree: _generate

    )pbdoc";
    
     m.def("constructTriUpsample", &constructTriUpsample, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
     m.def("colorPrecision", &colorPrecision, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
    m.def("hitRate", &hitRate, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
   m.def("extendColors", &extendColors, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
     m.def("extendPrecision", &extendPrecision, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
    m.def("firstValidMatch", &firstValidMatch, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
    m.def("firstValidMatches", &firstValidMatches, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
    m.def("vizDescriptor", &vizDescriptor, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
    m.def("getNearestSamples", &getNearestSamples, py::return_value_policy::copy, R"pbdoc(
        Descriptor visualization.
    )pbdoc");
    
   m.def("fpsGeodesic", &fpsGeodesic, py::return_value_policy::copy, R"pbdoc(
        Fixes stupid hippo.
    )pbdoc");
    
     m.def("fpsBiharmonic", &fpsBiharmonic, py::return_value_policy::copy, R"pbdoc(
        Fixes stupid hippo.
    )pbdoc");
        
    m.def("samplesToNearest", &samplesToNearest, py::return_value_policy::copy, R"pbdoc(
        Fixes stupid hippo.
    )pbdoc");
    
     m.def("readSplitConn", &readSplitConn, py::return_value_policy::copy, R"pbdoc(
        Fixes stupid hippo.
    )pbdoc");
    
    
     m.def("splitsSHRECConn", &splitsSHRECConn, py::return_value_policy::copy, R"pbdoc(
        Fixes stupid hippo.
    )pbdoc");
    
    m.def("computePR", &computePR, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
    m.def("denseGTS20b", &denseGTS20b, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
  
   m.def("denseGTS20a", &denseGTS20a, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
    m.def("getGeoError", &getGeoError, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
     m.def("getNearestDes", &getNearestDes, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    /*
        m.def("getMatchInd", &getMatchInd, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    */
    
    
      m.def("getNullPairs", &getNullPairs, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
     
      m.def("sampleGeoMat", &sampleGeoMat, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");

     m.def("transferSHRECGT", &transferSHRECGT, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
      m.def("SHRECGTtoVertex", &SHRECGTtoVertex, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
      m.def("splitsSHREC19", &splitsSHREC19, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
     m.def("readSplit", &readSplit, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
     m.def("composeMap", &composeMap, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
     m.def("LBSpectrum", &LBSpectrum, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
     m.def("computeECHO", &computeECHO, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
    m.def("computeError", &computeError, py::return_value_policy::copy, R"pbdoc(
        Computes geodesic errors.
    )pbdoc");
    
    m.def("errorToPlot", &errorToPlot, py::return_value_policy::copy, R"pbdoc(
        Computes geodesic errors.
    )pbdoc");
    
    m.def("biharmonicFPSBatch", &biharmonicFPSBatch, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
    m.def("biharmonicFPS", &biharmonicFPS, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    

    m.def("getAdjacentDist", &getAdjacentDist, py::return_value_policy::copy, R"pbdoc(
        Computes areas of mesh triangles w.r.t. biharmonic distance.
    )pbdoc");
    
    
    m.def("computeAreas", &computeAreas, py::return_value_policy::copy, R"pbdoc(
        Computes areas of mesh triangles w.r.t. biharmonic distance.
    )pbdoc");

    m.def("computeNormals", &computeNormals, py::return_value_policy::copy, R"pbdoc(
        Computes robust normals at vertices.
    )pbdoc");
    
    m.def("meshArea", &meshArea, py::return_value_policy::copy, R"pbdoc(
        Computes surface area of mesh
    )pbdoc");
    
         m.def("weights", &weights, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
          m.def("lumpedMass", &lumpedMass, py::return_value_policy::copy, R"pbdoc(
        Computes LB spectral decomposition.
    )pbdoc");
    
        m.def("labelSampling", &labelSampling, py::return_value_policy::copy, R"pbdoc(
       Assign labels to subsampling of mesh
    )pbdoc");
    
    
    m.def("poolNeighbors", &poolNeighbors, py::return_value_policy::copy, R"pbdoc(
        Computes pooling neighbors w.r.t. to biharmonic distance.
    )pbdoc");
    
    
        m.def("gradEdgesBatch", &gradEdgesBatch, py::return_value_policy::copy, R"pbdoc(
        Computes edges contributing to response of convolution at triangles. Assumes compact support. Biharmonic distances. 
    )pbdoc");

    m.def("gradEdges", &gradEdges, py::return_value_policy::copy, R"pbdoc(
        Computes edges contributing to response of convolution at triangles. Assumes compact support. Biharmonic distances. 
    )pbdoc");
   
    
    /*
        m.def("poolNeighbors", [](Eigen::MatrixXd& spec, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& selected){ py::gil_scoped_release;
        return poolNeighbors(spec, samples, selected);}, py::return_value_policy::copy, R"pbdoc(
        Computes pooling neighbors w.r.t. to biharmonic distance.
    )pbdoc");
    
    
        m.def("convEdgesBatch", [](Eigen::MatrixXd& pos, Eigen::MatrixXd& normals, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& indices, double maxD, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& batch){ py::gil_scoped_release; return convEdgesBatch( pos, normals, indices,maxD, batch);}, py::return_value_policy::copy, R"pbdoc(
        Computes edges contributing to response of convolution at triangles. Assumes compact support. Biharmonic distances. 
    )pbdoc");


    m.def("convEdges", [](Eigen::MatrixXd& pos, Eigen::MatrixXd& normals, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& indices, double maxD){ py::gil_scoped_release; return convEdges(pos, normals, indices, maxD);}, py::return_value_policy::copy, R"pbdoc(
        Computes edges contributing to response of convolution at triangles. Assumes compact support. Biharmonic distances. 
    )pbdoc");
    
 */

    
    m.def("precompute", &precompute, py::return_value_policy::copy, R"pbdoc(
        Precomputes logs associated with convolution edges.
    )pbdoc");
    
    m.def("hksSamples", &hksSamples, py::return_value_policy::copy, R"pbdoc(
        Computes HKS at vertices of sample triangles at input diffusion times.
    )pbdoc");
    
    m.def("wksSamples", &wksSamples, py::return_value_policy::copy, R"pbdoc(
        Computes HKS at vertices of sample triangles at input diffusion times.
    )pbdoc");
    
    m.def("specEmbed", &specEmbed, py::return_value_policy::copy, R"pbdoc(
        Computes HKS at vertices of sample triangles at input diffusion times.
    )pbdoc");

    


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
