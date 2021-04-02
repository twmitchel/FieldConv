#pragma once

#include "geometrycentral/surface/halfedge_containers.h"
#include "geometrycentral/surface/halfedge_element_types.h"
#include "geometrycentral/surface/halfedge_iterators.h"
#include "geometrycentral/utilities/utilities.h"

#include <list>
#include <memory>
#include <vector>

// NOTE: ipp includes at bottom of file

namespace geometrycentral {
namespace surface {
// Foward declare some return types from below
// template<typename T> class VertexData;
// template<> class VertexData<size_t>;

class HalfedgeMesh {

public:
  HalfedgeMesh();

  // Build a halfedge mesh from polygons, with a list of 0-indexed vertices incident on each face, in CCW order.
  // Assumes that the vertex listing in polygons is dense; all indices from [0,MAX_IND) must appear in some face.
  // (some functions, like in meshio.h preprocess inputs to strip out unused indices).
  // The output will preserve the ordering of vertices and faces.
  HalfedgeMesh(const std::vector<std::vector<size_t>>& polygons, bool verbose = false);
  ~HalfedgeMesh();


  // Number of mesh elements of each type
  size_t nHalfedges() const;
  size_t nInteriorHalfedges() const;
  size_t nCorners() const;
  size_t nVertices() const;
  size_t nInteriorVertices(); // warning: O(n)
  size_t nEdges() const;
  size_t nFaces() const;
  size_t nBoundaryLoops() const;
  size_t nExteriorHalfedges() const;

  // Methods for range-based for loops
  // Example: for(Vertex v : mesh.vertices()) { ... }
  HalfedgeSet halfedges();
  HalfedgeInteriorSet interiorHalfedges();
  HalfedgeExteriorSet exteriorHalfedges();
  CornerSet corners();
  VertexSet vertices();
  EdgeSet edges();
  FaceSet faces();
  BoundaryLoopSet boundaryLoops();

  // Methods for accessing elements by index
  // only valid when the  mesh is compressed
  Halfedge halfedge(size_t index);
  Corner corner(size_t index);
  Vertex vertex(size_t index);
  Edge edge(size_t index);
  Face face(size_t index);
  BoundaryLoop boundaryLoop(size_t index);

  // Methods that mutate the mesh. Note that these occasionally trigger a resize, which invaliates
  // any outstanding Vertex or MeshData<> objects. See the guide (currently in docs/mutable_mesh_docs.md).

  // Flip an edge. Unlike all the other mutation routines, this _does not_ invalidate pointers, though it does break the
  // canonical ordering.
  // Return true if the edge was actually flipped (can't flip boundary or non-triangular edges)
  bool flip(Edge e);

  // Adds a vertex along an edge, increasing degree of faces. Returns ptr along the edge, with he.vertex() as new
  // vertex and he.edge().halfedge() == he. Preserves canonical direction of edge.halfedge() for both halves of new
  // edge. The original edge is repurposed as one of the two new edges (same for original halfedges).
  Halfedge insertVertexAlongEdge(Edge e);

  // Split an edge, also splitting adjacent faces. Returns new halfedge which points away from the new vertex (so
  // he.vertex() is new vertex), and is the same direction as e.halfedge() on the original edge. The halfedge
  // direction of the other part of the new split edge is also preserved.
  Halfedge splitEdgeTriangular(Edge e);

  // Add vertex inside face and triangulate. Returns new vertex.
  Vertex insertVertex(Face f);

  // The workhorse version of connectVertices(). heA.vertex() will be connected to heB.vertex().
  // Returns new halfedge with vA at tail. he.twin().face() is the new face.
  Halfedge connectVertices(Halfedge heA, Halfedge heB);

  // Collapse an edge. Returns the vertex adjacent to that edge which still exists. Returns Vertex() if not
  // collapsible.
  // Vertex collapseEdge(Edge e); TODO

  // Remove a face which is adjacent to the boundary of the mesh (along with its edge on the boundary).
  // Face must have exactly one boundary edge.
  // Returns true if could remove
  // bool removeFaceAlongBoundary(Face f); TODO


  // Triangulate in a face, returns all subfaces
  std::vector<Face> triangulate(Face face);


  // Methods for obtaining canonical indices for mesh elements
  // (Note that in some situations, custom indices might instead be needed)
  VertexData<size_t> getVertexIndices();
  VertexData<size_t> getInteriorVertexIndices();
  HalfedgeData<size_t> getHalfedgeIndices();
  CornerData<size_t> getCornerIndices();
  EdgeData<size_t> getEdgeIndices();
  FaceData<size_t> getFaceIndices();
  BoundaryLoopData<size_t> getBoundaryLoopIndices();

  // == Utility functions
  bool hasBoundary() const;        // does the mesh have boundary? (aka not closed)
  bool isTriangular();             // returns true if and only if all faces are triangles [O(n)]
  int eulerCharacteristic() const; // compute the Euler characteristic [O(1)]
  int genus() const;               // compute the genus [O(1)]
  size_t nConnectedComponents();   // compute number of connected components [O(n)]
  void printStatistics() const;    // print info about element counts to std::cout

  std::vector<std::vector<size_t>> getFaceVertexList();
  std::unique_ptr<HalfedgeMesh> copy() const;

  // Compress the mesh
  bool isCompressed() const;
  void compress();

  // Canonicalize the element ordering to be the same indexing convention as after construction from polygon soup.
  bool isCanonical() const;
  void canonicalize();

  // == Callbacks that will be invoked on mutation to keep containers/iterators/etc valid.

  // Expansion callbacks
  // Argument is the new size of the element list. Elements up to this index may now be used (but _might_ not be
  // immediately)
  std::list<std::function<void(size_t)>> vertexExpandCallbackList;
  std::list<std::function<void(size_t)>> faceExpandCallbackList;
  std::list<std::function<void(size_t)>> edgeExpandCallbackList;
  std::list<std::function<void(size_t)>> halfedgeExpandCallbackList;

  // Compression callbacks
  // Argument is a permutation to a apply, such that d_new[i] = d_old[p[i]]
  // TODO FIXME there's a big problem here, in that this format of callbacks does not allow DynamicElement update in
  // O(1)
  // TODO think about capacity rules with callbacks: old rule was that new size may be smaller
  std::list<std::function<void(const std::vector<size_t>&)>> vertexPermuteCallbackList;
  std::list<std::function<void(const std::vector<size_t>&)>> facePermuteCallbackList;
  std::list<std::function<void(const std::vector<size_t>&)>> edgePermuteCallbackList;
  std::list<std::function<void(const std::vector<size_t>&)>> halfedgePermuteCallbackList;

  // Mesh delete callbacks
  // (this unfortunately seems to be necessary; objects which have registered their callbacks above
  // need to know not to try to de-register them if the mesh has been deleted)
  std::list<std::function<void()>> meshDeleteCallbackList;

  // Check capacity. Needed when implementing expandable containers for mutable meshes to ensure the contain can
  // hold a sufficient number of elements before the next resize event.
  size_t nHalfedgesCapacity() const;
  size_t nVerticesCapacity() const;
  size_t nEdgesCapacity() const;
  size_t nFacesCapacity() const;
  size_t nBoundaryLoopsCapacity() const;


  // == Debugging, etc

  // Performs a sanity checks on halfedge structure; throws on fail
  void validateConnectivity();


private:
  // = Core arrays which hold the connectivity
  // Note: it should always be true that heFace.size() == nHalfedgesCapacityCount, but any elements after
  // nHalfedgesFillCount will be valid indices (in the std::vector sense), but contain uninitialized data. Similarly,
  // any std::vector<> indices corresponding to deleted elements will hold meaningless values.
  std::vector<size_t> heNext;    // he.next()
  std::vector<size_t> heVertex;  // he.vertex()
  std::vector<size_t> heFace;    // he.face()
  std::vector<size_t> vHalfedge; // v.halfedge()
  std::vector<size_t> fHalfedge; // f.halfedge()

  // Implicit connectivity relationships
  static size_t heTwin(size_t iHe);   // he.twin()
  static size_t heEdge(size_t iHe);   // he.edge()
  static size_t eHalfedge(size_t iE); // e.halfedge()

  // Other implicit relationships
  bool heIsInterior(size_t iHe) const;
  bool faceIsBoundaryLoop(size_t iF) const;
  size_t faceIndToBoundaryLoopInd(size_t iF) const;
  size_t boundaryLoopIndToFaceInd(size_t iF) const;

  // Auxilliary arrays which cache other useful information

  // Track element counts (can't rely on rawVertices.size() after deletions have made the list sparse). These are the
  // actual number of valid elements, not the size of the buffer that holds them.
  size_t nHalfedgesCount = 0;
  size_t nInteriorHalfedgesCount = 0;
  size_t nVerticesCount = 0;
  size_t nFacesCount = 0;
  size_t nBoundaryLoopsCount = 0;

  // == Track the capacity and fill size of our buffers.
  // These give the capacity of the currently allocated buffer.
  // Note that this is _not_ defined to be std::vector::capacity(), it's the largest size such that arr[i] is legal (aka
  // arr.size()).
  size_t nVerticesCapacityCount = 0;
  size_t nHalfedgesCapacityCount = 0; // will always be even
  size_t nFacesCapacityCount = 0;     // capacity for faces _and_ boundary loops

  // These give the number of filled elements in the currently allocated buffer. This will also be the maximal index of
  // any element (except the weirdness of boundary loop faces). As elements get marked dead, nVerticesCount decreases
  // but nVertexFillCount does not (etc), so it denotes the end of the region in the buffer where elements have been
  // stored.
  size_t nVerticesFillCount = 0;
  size_t nHalfedgesFillCount = 0; // must always be even
  size_t nEdgesFillCount() const;
  size_t nFacesFillCount = 0;         // where the real faces stop, and empty/boundary loops begin
  size_t nBoundaryLoopsFillCount = 0; // remember, these fill from the back of the face buffer

  bool isCanonicalFlag = true;
  bool isCompressedFlag = true;

  // Hide copy and move constructors, we don't wanna mess with that
  HalfedgeMesh(const HalfedgeMesh& other) = delete;
  HalfedgeMesh& operator=(const HalfedgeMesh& other) = delete;
  HalfedgeMesh(HalfedgeMesh&& other) = delete;
  HalfedgeMesh& operator=(HalfedgeMesh&& other) = delete;


  // Implementation note: the getNew() and delete() functions below cannot operate on a single halfedge or edge. We must
  // simultaneously create or delete the triple of an edge and both adjacent halfedges. This constraint arises because
  // of the implicit indexing convention.

  // Used to resize the halfedge mesh. Expands and shifts vectors as necessary.
  Vertex getNewVertex();
  Halfedge getNewEdgeTriple(bool onBoundary); // returns e.halfedge() from the newly created edge
  Face getNewFace();
  BoundaryLoop getNewBoundaryLoop();

  // Detect dead elements
  bool vertexIsDead(size_t iV) const;
  bool halfedgeIsDead(size_t iHe) const;
  bool edgeIsDead(size_t iE) const;
  bool faceIsDead(size_t iF) const;
  bool boundaryLoopIsDead(size_t iBl) const;

  // Deletes leave tombstones, which can be cleaned up with compress()
  void deleteEdgeTriple(Halfedge he);
  void deleteElement(Vertex v);
  void deleteElement(Face f);
  void deleteElement(BoundaryLoop bl);

  // Compression helpers
  void compressHalfedges();
  void compressEdges();
  void compressFaces();
  void compressVertices();


  // Helpers for mutation methods
  void ensureVertexHasBoundaryHalfedge(Vertex v); // impose invariant that v.halfedge is start of half-disk
  Vertex collapseEdgeAlongBoundary(Edge e);


  // Elements need direct access in to members to traverse
  friend class Vertex;
  friend class Halfedge;
  friend class Corner;
  friend class Edge;
  friend class Face;
  friend class BoundaryLoop;

  // friend class VertexRangeIterator;
  friend struct VertexRangeF;
  friend struct HalfedgeRangeF;
  friend struct HalfedgeInteriorRangeF;
  friend struct HalfedgeExteriorRangeF;
  friend struct CornerRangeF;
  friend struct EdgeRangeF;
  friend struct FaceRangeF;
  friend struct BoundaryLoopRangeF;
};

} // namespace surface
} // namespace geometrycentral

// clang-format off
// preserve ordering
#include "geometrycentral/surface/halfedge_containers.ipp"
#include "geometrycentral/surface/halfedge_iterators.ipp"
#include "geometrycentral/surface/halfedge_element_types.ipp"
#include "geometrycentral/surface/halfedge_logic_templates.ipp"
#include "geometrycentral/surface/halfedge_mesh.ipp"
// clang-format on
