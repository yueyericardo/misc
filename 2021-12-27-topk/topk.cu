#include <torch/extension.h>
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>

#include <c10/macros/Macros.h>

using at::round_up;
using at::TensorBase;
using torch::Tensor;
using namespace at::native;

namespace {

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const& lhs, T const& rhs) {
    return (lhs + rhs);
  }
};

template <typename T, typename IndexType, int Dim, bool Order>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK(
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`

    IndexType numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType numTopKSlices,
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
#if defined(USE_ROCM)
  __shared__ int smem[64];
#else
  __shared__ int smem[32]; // one per each warp, up to warp limit
#endif
  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex = at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex = at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex = at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T topKValue = static_cast<T>(0);
  radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>(
      inputSliceStart, outputSliceSize, inputSliceSize, inputWithinSliceStride, smem, &topKValue);
  const auto topKConverted = at::native::TopKTypeConfig<T>::convert(topKValue);

  // Every value that is strictly less/greater than `pattern`
  // (depending on sort dir) in sorted int format is in the top-K.
  // The top-K value itself might not be unique.
  //
  // Since there are a variable number of elements that we see that
  // are within the top-k, we don't know at what index to write out
  // the resulting values.
  // In order to get this, we perform an exclusive prefix sum of
  // `hasTopK`. This will return the resulting index into which we
  // need to write the result, if a thread has a result.

  // All threads need to participate in the loop and the prefix sum,
  // but not necessarily in the load; hence loop bounds being rounded
  // up to a multiple of the block dim.
  IndexType numIterations = round_up(inputSliceSize, (IndexType)blockDim.x);
  IndexType writeIndexStart = 0;

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : static_cast<T>(0);
    const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
    bool hasTopK;
    if (Order) {
      hasTopK = inRange && (convertedV > topKConverted);
    } else {
      hasTopK = inRange && (convertedV < topKConverted);
    }

    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    writeIndexStart += carry;
  }

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize -
  // writeIndexStart. There might be more than that number available,
  // in which case we have to choose the first seen set. We do this
  // via a prefix sum to calculate indices for writing results.
  CUDA_KERNEL_ASSERT(outputSliceSize >= writeIndexStart);
  IndexType topKRemaining = (outputSliceSize - writeIndexStart);

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride]) : static_cast<T>(0);
    const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
    bool hasTopK = inRange && (convertedV == topKConverted);

    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    if (carry >= topKRemaining) {
      break;
    }

    topKRemaining -= carry;
    writeIndexStart += carry;
  }
};

} // namespace

template <typename T, typename IndexType, int Dim, bool Order>
void dispatchGatherTopK(
    at::cuda::detail::TensorInfo<T, IndexType> input,
    IndexType inputSliceSize,
    IndexType outputSliceSize, // aka `k`

    IndexType numInputSlices,
    IndexType inputWithinSliceStride,

    at::cuda::detail::TensorInfo<T, IndexType> topK,
    IndexType numTopKSlices,
    IndexType topKWithinSliceStride,

    at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
    IndexType indicesWithinSliceStride) {
  dim3 grid;
  TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices to sort");
  dim3 block(
      std::min(at::ceil_div((int64_t)inputSliceSize, (int64_t)C10_WARP_SIZE) * (int64_t)C10_WARP_SIZE, (int64_t)1024));

  gatherTopK<T, IndexType, Dim, Order><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input,
      inputSliceSize,
      outputSliceSize,
      numInputSlices,
      inputWithinSliceStride,
      topK,
      numTopKSlices,
      topKWithinSliceStride,
      indices,
      indicesWithinSliceStride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_topk_kernel(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    const Tensor& values,
    const Tensor& indices) {
  int numDims = self.dim();
  numDims = numDims == 0 ? 1 : numDims;
  TORCH_CHECK(numDims <= MAX_DIMS, "input tensor has too many dimensions");
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);

  auto input = self.contiguous();
  // static_cast is required to ensure that the correct type (INDEX_T)
  // is provided to the kernel for the arguments.

#define RUN_K(INDEX_T, DIM, DIR)                                                                                     \
  dispatchGatherTopK<scalar_t, INDEX_T, DIM, DIR>(                                                                   \
      inputInfo,                                                                                                     \
      static_cast<INDEX_T>(sliceSize),                                                                               \
      static_cast<INDEX_T>(k),                                                                                       \
      static_cast<INDEX_T>(inputSlices), /* The actual dimension that the k-selection is running in may have changed \
                                            from collapseDims() */                                                   \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]),                                                     \
      topKInfo,                                                                                                      \
      static_cast<INDEX_T>(topKSlices),                                                                              \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),                                                       \
      indicesInfo,                                                                                                   \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]));

#define RUN_DIR(INDEX_T, DIM)   \
  if (largest) {                \
    RUN_K(INDEX_T, DIM, true);  \
  } else {                      \
    RUN_K(INDEX_T, DIM, false); \
  }

#define RUN_DIM(INDEX_T)     \
  if (allDims == 1) {        \
    RUN_DIR(INDEX_T, 1);     \
  } else if (allDims == 2) { \
    RUN_DIR(INDEX_T, 2);     \
  } else if (allDims == 3) { \
    RUN_DIR(INDEX_T, 3);     \
  } else {                   \
    RUN_DIR(INDEX_T, -1);    \
  }

#define RUN_T(INDEX_T)                                                     \
  do {                                                                     \
    using scalar_t = float;                                                \
    at::cuda::detail::TensorInfo<scalar_t, INDEX_T> inputInfo =            \
        at::cuda::detail::getTensorInfo<scalar_t, INDEX_T>(input);         \
    at::cuda::detail::TensorInfo<scalar_t, INDEX_T> topKInfo =             \
        at::cuda::detail::getTensorInfo<scalar_t, INDEX_T>(values);        \
    at::cuda::detail::TensorInfo<int64_t, INDEX_T> indicesInfo =           \
        at::cuda::detail::getTensorInfo<int64_t, INDEX_T>(indices);        \
    /* tensorInfoLegacyIfScalar*/                                          \
    if (!input.dim()) {                                                    \
      inputInfo.dims = 1;                                                  \
      inputInfo.sizes[0] = 1;                                              \
      inputInfo.strides[0] = 1;                                            \
      topKInfo.dims = 1;                                                   \
      topKInfo.sizes[0] = 1;                                               \
      topKInfo.strides[0] = 1;                                             \
      indicesInfo.dims = 1;                                                \
      indicesInfo.sizes[0] = 1;                                            \
      indicesInfo.strides[0] = 1;                                          \
    }                                                                      \
    /* We use these structures solely to find the offset to */             \
    /* each slice we are operating on */                                   \
    inputInfo.sizes[dim] = 1;                                              \
    topKInfo.sizes[dim] = 1;                                               \
    indicesInfo.sizes[dim] = 1;                                            \
    /* stash the stride of dim because it can be accidentally collapsed */ \
    auto strideTopK = topKInfo.strides[dim];                               \
    auto strideIndices = indicesInfo.strides[dim];                         \
    /* Collapse all other dims */                                          \
    int collapseInputDim = inputInfo.collapseDims(dim);                    \
    int collapseTopKDim = topKInfo.collapseDims(dim);                      \
    int collapseIndicesDim = indicesInfo.collapseDims(dim);                \
    /* restore stride in case it was collapsed */                          \
    topKInfo.strides[collapseTopKDim] = strideTopK;                        \
    indicesInfo.strides[collapseIndicesDim] = strideIndices;               \
    int64_t inputSlices = 1;                                               \
    for (int i = 0; i < inputInfo.dims; ++i) {                             \
      inputSlices *= inputInfo.sizes[i];                                   \
    }                                                                      \
    int64_t topKSlices = 1;                                                \
    for (int i = 0; i < topKInfo.dims; ++i) {                              \
      topKSlices *= topKInfo.sizes[i];                                     \
    }                                                                      \
                                                                           \
    /* This is used as a template parameter to calculate indices. */       \
    /* We only specialize it if all collapsed dim sizes are the */         \
    /* same; otherwise, we use -1 which is the specialization */           \
    /* parameter for arbitrary dimensions */                               \
    int allDims = inputInfo.dims;                                          \
    if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {         \
      allDims = -1;                                                        \
    }                                                                      \
                                                                           \
    RUN_DIM(INDEX_T);                                                      \
  } while (0)

  // the below is safe with 0-dimensional tensors because it is based on
  // TensorInfo which implicitly expands to 1-dimensional.
  if (input.numel() > 0) {
    // Based on required index size, run the algorithm with the
    // appropriate index type
    if (at::cuda::detail::canUse32BitIndexMath(input) && at::cuda::detail::canUse32BitIndexMath(values) &&
        at::cuda::detail::canUse32BitIndexMath(indices)) {
      RUN_T(uint32_t);
    } else {
      RUN_T(uint64_t);
    }
  }
#undef RUN_T
#undef RUN_DIM
#undef RUN_DIR
#undef RUN_K
}

TORCH_LIBRARY(mbtopk, m) {
  m.def("multiBlockTopK", launch_gather_topk_kernel);
}

TORCH_LIBRARY_IMPL(mbtopk, CUDA, m) {
  m.impl("multiBlockTopK", launch_gather_topk_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
