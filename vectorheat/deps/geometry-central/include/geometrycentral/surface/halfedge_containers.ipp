#pragma once

// === Implementations for datatypes which hold data stored on the mesh

namespace geometrycentral {
namespace surface {

// === Actual function implementations

template <typename E, typename T>
MeshData<E, T>::MeshData() {}

template <typename E, typename T>
MeshData<E, T>::MeshData(HalfedgeMesh& parentMesh) : mesh(&parentMesh) {
  data.resize(elementCapacity<E>(mesh));
  fill(defaultValue);

  registerWithMesh();
}

template <typename E, typename T>
MeshData<E, T>::MeshData(HalfedgeMesh& parentMesh, T initVal) : mesh(&parentMesh), defaultValue(initVal) {
  data.resize(elementCapacity<E>(mesh));
  fill(defaultValue);

  registerWithMesh();
}

template <typename E, typename T>
MeshData<E, T>::MeshData(HalfedgeMesh& parentMesh, const Eigen::Matrix<T, Eigen::Dynamic, 1>& vector)
    : MeshData(parentMesh) {
  fromVector(vector);
}

template <typename E, typename T>
MeshData<E, T>::MeshData(HalfedgeMesh& parentMesh, const Eigen::Matrix<T, Eigen::Dynamic, 1>& vector,
                         const MeshData<E, size_t>& indexer)
    : MeshData(parentMesh) {
  fromVector(vector, indexer);
}

template <typename E, typename T>
MeshData<E, T>::MeshData(const MeshData<E, T>& other)
    : mesh(other.mesh), defaultValue(other.defaultValue), data(other.data) {
  registerWithMesh();
}

template <typename E, typename T>
MeshData<E, T>::MeshData(MeshData<E, T>&& other) noexcept
    : mesh(other.mesh), defaultValue(other.defaultValue), data(std::move(other.data)) {
  registerWithMesh();
}

template <typename E, typename T>
MeshData<E, T>& MeshData<E, T>::operator=(const MeshData<E, T>& other) {
  deregisterWithMesh();
  mesh = other.mesh;
  defaultValue = other.defaultValue;
  data = other.data;
  registerWithMesh();

  return *this;
}

template <typename E, typename T>
MeshData<E, T>& MeshData<E, T>::operator=(MeshData<E, T>&& other) noexcept {
  deregisterWithMesh();
  mesh = other.mesh;
  defaultValue = other.defaultValue;
  data = std::move(other.data);
  registerWithMesh();

  return *this;
}

template <typename E, typename T>
MeshData<E, T>::~MeshData() {
  deregisterWithMesh();
}


template <typename E, typename T>
void MeshData<E, T>::registerWithMesh() {

  // Used during default initialization
  if (mesh == nullptr) return;

  // Callback function on expansion
  std::function<void(size_t)> expandFunc = [&](size_t newSize) {
    size_t oldSize = data.size();
    data.resize(newSize);
    for (size_t i = oldSize; i < data.size(); i++) {
      data[i] = defaultValue;
    }
  };


  // Callback function on compression
  std::function<void(const std::vector<size_t>&)> permuteFunc = [this](const std::vector<size_t>& perm) {
    data = applyPermutation(data, perm);
  };


  // Callback function on mesh delete
  std::function<void()> deleteFunc = [this]() {
    // Ensures that we don't try to remove with iterators on deconstruct of this object
    mesh = nullptr;
  };

  expandCallbackIt = getExpandCallbackList<E>(mesh).insert(getExpandCallbackList<E>(mesh).begin(), expandFunc);
  permuteCallbackIt = getPermuteCallbackList<E>(mesh).insert(getPermuteCallbackList<E>(mesh).end(), permuteFunc);
  deleteCallbackIt = mesh->meshDeleteCallbackList.insert(mesh->meshDeleteCallbackList.end(), deleteFunc);
}

template <typename E, typename T>
void MeshData<E, T>::deregisterWithMesh() {

  // Used during destruction of default-initializated object, for instance
  if (mesh == nullptr) return;

  getExpandCallbackList<E>(mesh).erase(expandCallbackIt);
  getPermuteCallbackList<E>(mesh).erase(permuteCallbackIt);
  mesh->meshDeleteCallbackList.erase(deleteCallbackIt);
}

template <typename E, typename T>
void MeshData<E, T>::fill(T val) {
  std::fill(data.begin(), data.end(), val);
}

template <typename E, typename T>
inline void MeshData<E, T>::clear() {
  deregisterWithMesh();
  mesh = nullptr;
  defaultValue = T();
  data.clear();
}

template <typename E, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> MeshData<E, T>::toVector() const {
  Eigen::Matrix<T, Eigen::Dynamic, 1> result(nElements<E>(mesh));
  size_t i = 0;
  for (E e : iterateElements<E>(mesh)) {
    result(i) = (*this)[e];
    i++;
  }
  return result;
}

template <typename E, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> MeshData<E, T>::toVector(const MeshData<E, size_t>& indexer) const {
  size_t outSize = 0;
  for (E e : iterateElements<E>(mesh)) {
    if (indexer[e] != std::numeric_limits<size_t>::max()) outSize++;
  }
  Eigen::Matrix<T, Eigen::Dynamic, 1> result(outSize);
  for (E e : iterateElements<E>(mesh)) {
    if (indexer[e] != std::numeric_limits<size_t>::max()) {
      result(indexer[e]) = (*this)[e];
    }
  }
  return result;
}

template <typename E, typename T>
void MeshData<E, T>::fromVector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vector) {
  if ((size_t)vector.rows() != nElements<E>(mesh)) throw std::runtime_error("Vector size does not match mesh size.");
  size_t i = 0;
  for (E e : iterateElements<E>(mesh)) {
    (*this)[e] = vector(i);
    i++;
  }
}

template <typename E, typename T>
void MeshData<E, T>::fromVector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vector, const MeshData<E, size_t>& indexer) {
  for (E e : iterateElements<E>(mesh)) {
    if (indexer[e] != std::numeric_limits<size_t>::max()) {
      (*this)[e] = vector(indexer[e]);
    }
  }
}

template <typename E, typename T>
inline T& MeshData<E, T>::operator[](E e) {
#ifndef NDEBUG
  // These checks are a bit much to do on every access, so disable in release mode.
  assert(mesh != nullptr && "MeshData is uninitialized.");
  assert(e.getMesh() == mesh && "Attempted to access MeshData with member from wrong mesh");
#endif
  size_t i = dataIndexOfElement(mesh, e);
  return data[i];
}

template <typename E, typename T>
inline const T& MeshData<E, T>::operator[](E e) const {
#ifndef NDEBUG
  // These checks are a bit much to do on every access, so disable in release mode.
  assert(mesh != nullptr && "MeshData is uninitialized.");
  assert(e.getMesh() == mesh && "Attempted to access MeshData with member from wrong mesh");
#endif
  size_t i = dataIndexOfElement(mesh, e);
  return data[i];
}

template <typename E, typename T>
inline T& MeshData<E, T>::operator[](size_t i) {
#ifndef NDEBUG
  assert(i < size() && "Attempted to access MeshData with out of bounds index");
#endif
  return data[i];
}

template <typename E, typename T>
inline const T& MeshData<E, T>::operator[](size_t i) const {
#ifndef NDEBUG
  assert(i < size() && "Attempted to access MeshData with out of bounds index");
#endif
  return data[i];
}

template <typename E, typename T>
inline size_t MeshData<E, T>::size() const {
  if (mesh == nullptr) return 0;
  return nElements<E>(mesh);
}


template <typename E, typename T>
inline MeshData<E, T> MeshData<E, T>::reinterpretTo(HalfedgeMesh& targetMesh) {
  GC_SAFETY_ASSERT(nElements<E>(mesh) == nElements<E>(&targetMesh),
                   "meshes must have same number of elements to reinterpret");
  MeshData<E, T> newData(targetMesh, defaultValue);
  newData.data = data;
  return newData;
}

} // namespace surface
} // namespace geometrycentral
