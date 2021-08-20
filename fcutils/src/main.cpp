#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/halfedge_factories.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <sstream>
#include <chrono>


// Adapted from https://github.com/rubenwiersma/hsn


using namespace geometrycentral;
using namespace geometrycentral::surface;
namespace py = pybind11;

// Geometry-central data
std::unique_ptr<HalfedgeMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

// Algorithm parameters for Vector Heat method
float tCoef = 1.0;
std::unique_ptr<VectorHeatMethodSolver> solver;

// HELPER FUNCTIONS ------------------------------------------------------------

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


Eigen::MatrixXd precomputeLogXP(Vertex& sourceV, Eigen::Matrix<size_t, Eigen::Dynamic, 1> targetVs,
                                  Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sample_points) {
  if (solver == nullptr) {
    solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
  }

   std::vector<std::tuple<SurfacePoint, Vector2>> points;
   points.emplace_back(sourceV, Vector2{1.0, 0.0});


  VertexData<Vector2> connection = solver->transportTangentVectors(points);

  // And compute the logarithmic map giving the position of i w.r.t. j: log_j(i)

  VertexData<Vector2> logmap = solver->computeLogMap(sourceV);


  Eigen::MatrixXd res(targetVs.rows(), 4);


  for (size_t i = 0; i < targetVs.rows(); i++) {
    size_t v = sample_points(targetVs(i));

    Vector2 targetCoords = logmap[v];

    Vector2 targetConnection = connection[v];

    // Store the parallel transport (connection) and logarithmic map.
    res(i, 0) = targetConnection.x;
    res(i, 1) = targetConnection.y;
    res(i, 2) = targetCoords.x;
    res(i, 3) = targetCoords.y;
  }

  return res;
}



// PRECOMPUTATION------------------------------------------------------------

// Precomputes the logarithmic map and parallel 
Eigen::MatrixXd precompute(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces,
        Eigen::Matrix<size_t, Eigen::Dynamic, 2>& edge_index, Eigen::Matrix<size_t, Eigen::Dynamic, 1> degree,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1>& sample_points) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();
  geometry->requireVertexPrincipalCurvatureDirections();


  // Setup solver for Vector Heat Method.
  solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));

  // Store the results in an Eigen matrix, which can be accessed as a NumPy array.
  Eigen::MatrixXd res(edge_index.rows(), 4);
  size_t index = 0;
  // For each sampled point:
  for (size_t row = 0; row < sample_points.rows(); row++) {
    Vertex v = mesh->vertex(sample_points(row));

    // Compute parallel transport and logarithmic map for neighborhood.
    res.block(index, 0, degree(row), 4) = precomputeLogXP(v, edge_index.block(index, 1, degree(row), 1), sample_points);
    index += degree(row);
  }

  geometry->unrequireVertexPrincipalCurvatureDirections();
  geometry->unrequireVertexLumpedMassMatrix();
  geometry->unrequireVertexIndices();
  return res;
}

// Computes the vertex lumped mass matrix for each sampled vertex,
// automatically adding the weights from nearest geodesic neighbors.
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





// UTILITIES ------------------------------------------------------------

// Compute the surface area of a mesh.
double surface_area(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  float surfaceArea = 0.0f;
  for (Face f : mesh->faces()) {
    surfaceArea += geometry->faceArea(f);
  }

  return surfaceArea;
}

// Compute geodesic nearest neighbors
Eigen::MatrixXd nearest(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1>& selected_points, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labels) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();

  // We use short-time heat diffusion to retrieve geodesic nearest neighbors.
  VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);

  // Set up indices of sampled points to diffuse.
  std::vector<std::tuple<SurfacePoint, double>> points;
  for (size_t row = 0; row < selected_points.rows(); row++) {
    points.emplace_back(SurfacePoint(mesh->vertex(selected_points(row))), labels(row));
  }

  // Solve heat diffusion
  VertexData<double> scalarExtension = vhmSolver.extendScalar(points);

  // Store the results in an Eigen matrix, which can be accessed as a NumPy array.
  Eigen::MatrixXd res(pos.rows(), 1);
  for (size_t row = 0; row < pos.rows(); row++) {
    res(row) = scalarExtension[mesh->vertex(row)];
  }
  
  geometry->unrequireVertexIndices();

  return res;
}

//==========================
// Data Manipulation
//==========================

// Invert and compose ground truth labelings
Eigen::MatrixXd composeMap (Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labelsTem2Tar, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& labelsTem2Sour, Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces)
{
   
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

  
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);
   
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

// Given a set of subsampled points, for each point on the original mesh find the nearest sampled point
Eigen::MatrixXd samplesToNearest(Eigen::MatrixXd& pos, Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic>& faces, Eigen::Matrix<size_t, Eigen::Dynamic, 1>& samples)
{
   std::tie(mesh, geometry) = loadMesh_np(pos, faces);
   
   geometry->requireVertexIndices();

   
   // Geodesic Nearest Neighbor   
   HeatMethodDistanceSolver heatSolver(*geometry, 1.0);

   
   std::vector<Vertex> points;
   
   for (int l = 0; l < samples.rows(); l++)
   {

      points.push_back ( mesh->vertex( (int) samples(l, 0) ) );

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

void splitSHREC19(std::string rawDir)
{
   std::vector<std::string> setFiles {"/pairs/figure_pairs.txt", "/pairs/glove_pairs.txt", "/pairs/hand_pairs.txt"};
   
   
   std::vector<std::pair<std::string, std::string>> test, train;
   std::srand ( unsigned ( std::time(0) ) );
   
   for (int l = 0; l < setFiles.size(); l++)
   {
      std::cout << rawDir + setFiles[l] << std::endl;
      
      FILE* setFile = fopen ( (rawDir + setFiles[l]).c_str (), "r");

      char line[300];

      std::vector< std::pair<std::string, std::string> > pairs;
      
      while (fgets( line, 300, setFile) )
      {


         std::string sID, tID;
         sID.push_back ( line[0] ); sID.push_back ( line[1] ); sID.push_back (line[2]);
         tID.push_back ( line[4] ); tID.push_back ( line[5] ); tID.push_back (line[6]);
         
         pairs.push_back (std::pair<std::string, std::string> (sID, tID));
         
      }
      
      fclose(setFile);
      

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
   std::string testFile = rawDir + "/test_pairs.txt";
   std::string trainFile = rawDir + "/train_pairs.txt";
      
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

Eigen::MatrixXd readSplit(std::string rawDir)
{
   
   std::vector<std::pair<std::string, std::string>> test, train;

   FILE* testFile = fopen ( (rawDir + "/test_pairs.txt").c_str (), "r");

   char line[300];

   std::vector< std::pair<int, int> > test_pairs;
   
   while (fgets( line, 300, testFile) )
   {
      
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


   FILE* trainFile = fopen ( (rawDir + "/train_pairs.txt").c_str (), "r");


   std::vector< std::pair<int, int> > train_pairs;
   
   while (fgets( line, 300, trainFile) )
   {
      
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
   
   




PYBIND11_MODULE(fcutils, m) {
    m.doc() = R"pbdoc(
        Precomputation module.
        -----------------------

        .. currentmodule:: precomputation

        .. autosummary::
           :toctree: _generate

           add
           precompute
           diameter
    )pbdoc";

    m.def("precompute", &precompute, py::return_value_policy::copy, R"pbdoc(
        Precompute parallel transport and logarithmic map for meshes given by pos, face, edges and degree.
    )pbdoc");

    m.def("surface_area", &surface_area, py::return_value_policy::copy, R"pbdoc(
        Computes surface area of the given mesh.
    )pbdoc");

    m.def("weights", &weights, py::return_value_policy::copy, R"pbdoc(
        Computes vertex lumped mass matrix for sampled points.
    )pbdoc");

    m.def("nearest", &nearest, py::return_value_policy::copy, R"pbdoc(
        Returns a mapping from all vertices to the nearest sampled points.
    )pbdoc");
        
    m.def("composeMap", &composeMap, py::return_value_policy::copy, R"pbdoc(
        Inverts and composes ground-truth labelings.
    )pbdoc");
    
    m.def("samplesToNearest", &samplesToNearest, py::return_value_policy::copy, R"pbdoc(
        Given a set of subsampled points, for each point on the original mesh find the nearest sampled point.
    )pbdoc");
    
     m.def("splitSHREC19", &splitSHREC19, py::return_value_policy::copy, R"pbdoc(
       Generate random test and train splits for the SHREC19 dataset.
    )pbdoc");
    
     m.def("readSplit", &readSplit, py::return_value_policy::copy, R"pbdoc(
       Read the generated test and train splits for the SHREC19 dataset.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
