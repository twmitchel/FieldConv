These routines allow modification of the mesh connectivity and insertion/deletion of elements.

Geometry-central is designed from the ground up to have good support for mesh mutation. The underlying `HalfedgMesh` data structure is index-based, with lazy expansion and deletion, so all operations run in (amortized) constant time with respect to the number of mesh elements, and usually do not incur any memory allocations. [Containers](containers.md) automatically update after mesh operations.

As much as possible, these routines will check for validity before executing and throw an exception if something isn't right. The `NGC_SAFETY_CHECKS` define disables this behavior for a modest increase in performance, but checks are enabled by default even in release builds.

Note that aggressive use of these routines may reduce a mesh from a _simplicial complex_ to a _$\Delta$-complex_. For instance, flipping enough edges in a mesh might create self-edges, which connect a vertex to itself. See the [$\Delta$-complex](delta_complex.md) section for details, and an explanation of why these complexes are important.

## In-place modifications

??? func "`#!cpp bool HalfedgeMesh::flip(Edge e)`"

    Flip an edge by rotating counter-clockwise. 

    An edge cannot be combinatorially flipped if it is:

      - a boundary edge
      - incident on a degree-1 vertex.

    **Return:** true if the edge was actually flipped 


## Insertions

These routines modify a mesh by inserting new elements. Element references remain valid, and [containers](containers.md) will automatically resize themselves to accommodate the new elements. 

Note that some operations my re-use existing elements to create their output. For instance, `splitEdge()` turns a single edge in to two; the input edge will be re-used as one of the two output edges, and data along that edge will be unchanged in any containers.

??? warning "Boundary loop invalidation"

    There is one tiny exceptional invalidation behavior related to insertion. `Face` handles which actually point to boundary loops are invalidated after any operation which adds faces to the mesh. This is a consequence of the way we index boundary loops separate from faces, even though they are essentially faces in practice (see [Boundaries](boundaries.md) and [Internals](internals.md)) for details.

---

??? func "`#!cpp Halfedge HalfedgeMesh::insertVertexAlongEdge(Edge e)`"

    Adds a degree 2 vertex along an edge. Unlike `splitEdge()`, this _does not_ triangulate the adjacent faces; the degree of adjacent faces will be increased by 1. Works as expected on boundary edges.

    Returns a halfedge `he` along the newly created edge, which points in the same direction as `e.halfedge()`, and such that `he.vertex()` is the newly inserted vertex.

    Preserves canonical direction of edge.halfedge() for both halves of new edge. The original edge is repurposed as one of the two new edges (same for original halfedges).


??? func "`#!cpp Halfedge HalfedgeMesh::splitEdge(Edge e)`"

    Inserts a vertex along an edge, and triangulates the adjacent faces. On a triangle mesh, the newly inserted vertex will be a degree 4 vertex.  Works as expected on boundary edges.

    Returns a halfedge `he` along the newly created edge, which points in the same direction as `e.halfedge()`, and such that `he.vertex()` is the newly inserted vertex.

    Preserves canonical direction of edge.halfedge() for both halves of new edge. The original edge is repurposed as one of the new edges (same for original halfedges).
    

??? func "`#!cpp Vertex HalfedgeMesh::insertVertex(Face f)`"

    TODO
    // Add vertex inside face and triangulate. Returns new vertex.
    Vertex insertVertex(Face f);


??? func "`#!cpp Halfedge connectVertices(Halfedge heA, Halfedge heB)`"

    Call to add an edge to a face, splitting it to two faces.

    Creates a new edge connecting `heA.vertex()` to `heB.vertex()`. The initial shared face will be repurposed as one of the two resulting faces.
    
    `heA` and `heB` must be distinct halfedges in the same face, and their vertices must not already be adjacent in that face.

    Returns new halfedge with `heA.vertex()` at tail, and `he.twin().face()` is the new face.


??? func "`#!cpp std::vector<Face> HalfedgeMesh::triangulate(Face face)`"

    Triangulate a face in the mesh, returning all of the resulting faces.
    
    One of the returned faces will be the input face, repurposed as a face in the triangulation.


### Trimming storage
    
To amortize the cost of allocation, mesh buffers are resized sporadically in large increments; these resized buffers might significantly increase (e.g., double) the storage size of a mesh and the associated containers. Calling `trimStorage()` frees up any unused storage space to reduce memory usage. 

??? func "`#!cpp void HalfedgeMesh::trimStorage()`"

    Free any additional storage associated with the mesh. Does not invalidate elements.

    Trimming storage *does not* put the mesh in _compressed mode_, though compressing the mesh does trim storage.

    This function costs $O(n)$ and should not be called in a tight loop.


## Deletions

These routines delete mesh elements. Elements (like `Vertex`) and containers (like `VertexData<>`) will remain valid through deletions. However, performing any deletion will cause the mesh to no longer be [compressed](#compressed-mode).

??? func "`#!cpp Vertex HalfedgeMesh::collapseEdge(Edge e)`"

    // Collapse an edge. Returns the vertex adjacent to that edge which still exists. Returns Vertex() if not
    // collapsible.
    Vertex collapseEdge(Edge e);

??? func "`#!cpp bool HalfedgeMesh::removeFaceAlongBoundary(Face f)`"

    // Remove a face which is adjacent to the boundary of the mesh (along with its edge on the boundary).
    // Face must have exactly one boundary edge.
    // Returns true if could remove
    bool removeFaceAlongBoundary(Face f);


### Compressed mode

Internally, the halfedge mesh is represented by dense arrays of indices which are lazily expanded (see [interals](internals.md) for details). To support fast deletion operations, we simply mark elements as deleted, without re-packing the index space. We say that the mesh is _compressed_ if the index space is dense and there are no such marked elements. When a mesh is not compressed, the `index` of a mesh element no longer serves as a proper enumeration from `[0,N)`, but merely as a unique ID.

There are two consequences to being non-compressed:

  - Some operations cannot be implemented efficiently/correctly (e.g., random access of the i'th vertex)
  - Storage space is wasted by deleted elements


**All meshes are compressed after construction, and only become non-compressed if the user performs a deletion operation.**  The `makeCompressed()` function can be called to re-index the elements of the mesh as a proper enumeration from `[0,N)`.

The `makeCompressed()` function invalidates pointers, and incurs an update of existing containers. As such, it is recommended to be called sporadically, after a sequence of operations is completed.

??? func "`#!cpp bool HalfedgeMesh::isCompressed()`"

    Returns true if the mesh is compressed.

??? func "`#!cpp void HalfedgeMesh::makeCompressed()`"

    Re-index the elements of the mesh to yield a dense enumeration. Invalidates all Vertex (etc) objects.

    Does nothing if the mesh is already compressed.

### Dynamic pointer types

A few of the operations listed below invalidate outstanding element references (like `Halfedge`) by re-indexing the elements of the mesh. [Containers](containers.md) automatically update after re-indexing, and often code can be structured such that no element references need to be maintained across an invalidation.

However, if it is necessary to keep a reference to an element through a re-indexing, the `DynamicHalfedge` can be used. These types behave like a `Halfedge`, with the exception that they automatically update to remain valid when a mesh is re-indexed. These types should only be used when necessary, because they are expensive to maintain.

