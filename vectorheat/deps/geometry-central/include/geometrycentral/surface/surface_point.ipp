#pragma once

namespace geometrycentral {
namespace surface {


// == Constructors
inline SurfacePoint::SurfacePoint() : type(SurfacePointType::Vertex) {}
inline SurfacePoint::SurfacePoint(Vertex v) : type(SurfacePointType::Vertex), vertex(v) {}
inline SurfacePoint::SurfacePoint(Edge e, double tEdge_) : type(SurfacePointType::Edge), edge(e), tEdge(tEdge_) {}
inline SurfacePoint::SurfacePoint(Face f, Vector3 faceCoords_)
    : type(SurfacePointType::Face), face(f), faceCoords(faceCoords_) {}


// == Methods

inline std::ostream& operator<<(std::ostream& output, const SurfacePoint& p) {
  switch (p.type) {
  case SurfacePointType::Vertex: {
    output << "[SurfacePoint: type=Vertex, vertex= " << p.vertex << "]";
    break;
  }
  case SurfacePointType::Edge: {
    output << "[SurfacePoint: type=Edge, edge= " << p.edge << " tEdge= " << p.tEdge << "]";
    break;
  }
  case SurfacePointType::Face: {
    output << "[SurfacePoint: type=Face, face= " << p.face << " faceCoords= " << p.faceCoords << "]";
    break;
  }
  }

  return output;
}


inline SurfacePoint SurfacePoint::inSomeFace() const {

  switch (type) {
  case SurfacePointType::Vertex: {

    Halfedge he = vertex.halfedge();
    Face inFace = he.face();
    Halfedge targetHe = inFace.halfedge();

    // Find the appropriate barycentric coordinate and return
    if (he == targetHe) {
      return SurfacePoint(inFace, Vector3{1., 0., 0.});
    }
    he = he.next();
    if (he == targetHe) {
      return SurfacePoint(inFace, Vector3{0., 0., 1.});
    }
    return SurfacePoint(inFace, Vector3{0., 1., 0.});

    break;
  }
  case SurfacePointType::Edge: {

    Halfedge he = edge.halfedge();
    Face inFace = he.face();
    Halfedge targetHe = inFace.halfedge();

    // Find the appropriate barycentric coordinate and return
    if (he == targetHe) {
      return SurfacePoint(inFace, Vector3{1. - tEdge, tEdge, 0.});
    }
    he = he.next();
    if (he == targetHe) {
      return SurfacePoint(inFace, Vector3{tEdge, 0., 1. - tEdge});
    }
    return SurfacePoint(inFace, Vector3{0., 1. - tEdge, tEdge});

    break;
  }
  case SurfacePointType::Face: {
    return *this;
    break;
  }
  }

  throw std::logic_error("bad switch");
  return *this;
}


inline Vertex SurfacePoint::nearestVertex() const {

  switch (type) {
  case SurfacePointType::Vertex: {
    return vertex;
    break;
  }
  case SurfacePointType::Edge: {
    if (tEdge < .5) return edge.halfedge().vertex();
    return edge.halfedge().twin().vertex();
    break;
  }
  case SurfacePointType::Face: {
    if (faceCoords.x >= faceCoords.y && faceCoords.x >= faceCoords.z) {
      return face.halfedge().vertex();
    }
    if (faceCoords.y >= faceCoords.x && faceCoords.y >= faceCoords.z) {
      return face.halfedge().next().vertex();
    }
    return face.halfedge().next().next().vertex();
    break;
  }
  }

  throw std::logic_error("bad switch");
  return vertex;
}

template <typename T>
inline T SurfacePoint::interpolate(const VertexData<T>& data) const {

  switch (type) {
  case SurfacePointType::Vertex: {
    return data[vertex];
    break;
  }
  case SurfacePointType::Edge: {
    T valTail = data[edge.halfedge().vertex()];
    T valTip = data[edge.halfedge().twin().vertex()];
    return (1. - tEdge) * valTail + tEdge * valTip;
    break;
  }
  case SurfacePointType::Face: {
    T valA = data[face.halfedge().vertex()];
    T valB = data[face.halfedge().next().vertex()];
    T valC = data[face.halfedge().next().next().vertex()];

    return (faceCoords.x * valA) + (faceCoords.y * valB) + (faceCoords.z * valC);
    break;
  }
  }

  throw std::logic_error("bad switch");
  return data[vertex];
}

inline void SurfacePoint::validate() const {
  const double EPS = 1e-4;

  switch (type) {
  case SurfacePointType::Vertex: {
    if (vertex == Vertex()) throw std::logic_error("surface point with Type::Vertex has invalid vertex ref");
    break;
  }
  case SurfacePointType::Edge: {
    if (edge == Edge()) throw std::logic_error("surface point with Type::Edge has invalid edge ref");
    if (!std::isfinite(tEdge)) throw std::logic_error("surface point with Type::Edge has non-finite tEdge");
    if (tEdge < -EPS || tEdge > (1. + EPS))
      throw std::logic_error("surface point with Type::Edge has tEdge outside of [0,1] = " + std::to_string(tEdge));
    break;
  }
  case SurfacePointType::Face: {
    if (face == Face()) throw std::logic_error("surface point with Type::Face has invalid face ref");
    if (!isfinite(faceCoords)) throw std::logic_error("surface point with Type::Face has non-finite face coords");
    if (faceCoords.x < -EPS || faceCoords.y < -EPS || faceCoords.z < -EPS)
      throw std::logic_error("surface point with Type::Face has negative bary coord " + std::to_string(faceCoords));

    if ((faceCoords.x + faceCoords.y + faceCoords.z) > (1. + EPS))
      throw std::logic_error("surface point with Type::Face has bary coord that sum to > 1 " +
                             std::to_string(faceCoords));
    break;
  }
  }
}

} // namespace surface
} // namespace geometrycentral

namespace std {
inline std::string to_string(geometrycentral::surface::SurfacePoint p) {
  ostringstream output;
  output << p;
  return output.str();
}
} // namespace std

