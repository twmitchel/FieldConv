#pragma once

#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/halfedge_mesh.h"

#include "gtest/gtest.h"

// A mesh used for testing
struct MeshAsset {
  MeshAsset(){};
  MeshAsset(std::string localPath);

  std::string name = "Unnamed_Mesh_Asset";
  std::string sourcePath = "unknown";
  std::unique_ptr<geometrycentral::surface::HalfedgeMesh> mesh;
  std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry;
  bool hasBoundary = false;
  bool isTriangular = true;
  bool isPolygonalComplex = true;

  MeshAsset copy() const;
  void printThyName();
};

// Loads test meshes from disk
class MeshAssetSuite : public ::testing::Test {
public:
  MeshAsset getAsset(std::string name);

  // Get various groups for meshes
  // if includeNoGeom=true, will also include meshes which have connectivity but not geometry
  std::vector<MeshAsset> allMeshes(bool includeNoGeom = false);
  std::vector<MeshAsset> closedMeshes(bool includeNoGeom = false);           // have no boundary
  std::vector<MeshAsset> boundaryMeshes(bool includeNoGeom = false);         // have boundary
  std::vector<MeshAsset> triangularMeshes(bool includeNoGeom = false);       // faces have three sides
  std::vector<MeshAsset> polygonalComplexMeshes(bool includeNoGeom = false); // not a general delta complex

protected:
  static void SetUpTestSuite();
  // static void TearDownTestSuite();
  // virtual void SetUp();
  // virtual void TearDown();

private:
  static std::vector<MeshAsset> allMeshAssets;
};
