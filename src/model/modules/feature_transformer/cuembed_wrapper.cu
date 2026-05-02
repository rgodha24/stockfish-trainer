#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

#include "cuembed/include/embedding_lookup.cuh"
#include "cuembed/include/index_transforms.cuh"

namespace {

void check_i32_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::ScalarType::Int,
              name,
              " must be int32");
}

void check_f32_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::ScalarType::Float,
              name,
              " must be float32");
}

void cuembed_backward_checked(torch::Tensor flat_indices,
                              torch::Tensor offsets,
                              torch::Tensor grad_output,
                              torch::Tensor grad_weight) {
  check_i32_cuda_contiguous(flat_indices, "flat_indices");
  check_i32_cuda_contiguous(offsets, "offsets");
  check_f32_cuda_contiguous(grad_output, "grad_output");
  check_f32_cuda_contiguous(grad_weight, "grad_weight");
  TORCH_CHECK(flat_indices.dim() == 1, "flat_indices must be rank-1");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be rank-1");
  TORCH_CHECK(offsets.numel() >= 1, "offsets must include the final offset");
  TORCH_CHECK(grad_output.dim() == 2, "grad_output must be rank-2");
  TORCH_CHECK(grad_weight.dim() == 2, "grad_weight must be rank-2");
  TORCH_CHECK(grad_weight.size(1) == grad_output.size(1),
              "grad_weight width must match grad_output width");

  const int batch_size = static_cast<int>(offsets.numel() - 1);
  const int embed_width = static_cast<int>(grad_output.size(1));
  const int nnz = static_cast<int>(flat_indices.numel());

  if (nnz == 0) {
    grad_weight.zero_();
    return;
  }

  auto sample_ids = torch::empty({nnz}, flat_indices.options());
  cuembed::ExtractRowIdsFromCSR<int32_t, int32_t>(
      offsets.data_ptr<int32_t>(),
      batch_size,
      sample_ids.mutable_data_ptr<int32_t>(),
      at::cuda::getCurrentCUDAStream());

  auto transpose_indices = torch::empty({nnz}, flat_indices.options());
  auto transpose_sample_ids = torch::empty({nnz}, flat_indices.options());

  size_t lwork = 0;
  cuembed::Transpose<int32_t, float>(
      sample_ids.data_ptr<int32_t>(),
      flat_indices.data_ptr<int32_t>(),
      nullptr,
      nnz,
      transpose_indices.mutable_data_ptr<int32_t>(),
      transpose_sample_ids.mutable_data_ptr<int32_t>(),
      nullptr,
      nullptr,
      &lwork,
      at::cuda::getCurrentCUDAStream());
  auto work = torch::empty(
      {static_cast<int64_t>(lwork)},
      torch::TensorOptions().device(flat_indices.device()).dtype(torch::kUInt8));
  cuembed::Transpose<int32_t, float>(
      sample_ids.data_ptr<int32_t>(),
      flat_indices.data_ptr<int32_t>(),
      nullptr,
      nnz,
      transpose_indices.mutable_data_ptr<int32_t>(),
      transpose_sample_ids.mutable_data_ptr<int32_t>(),
      nullptr,
      reinterpret_cast<char*>(work.mutable_data_ptr<uint8_t>()),
      &lwork,
      at::cuda::getCurrentCUDAStream());

  cuembed::EmbeddingBackward<float, int32_t>(
      grad_output.data_ptr<float>(),
      embed_width,
      static_cast<int>(grad_weight.size(0)),
      nnz,
      transpose_indices.data_ptr<int32_t>(),
      transpose_sample_ids.data_ptr<int32_t>(),
      nullptr,
      nullptr,
      false,
      grad_weight.mutable_data_ptr<float>(),
      nullptr,
      at::cuda::getCurrentCUDAStream());
}

}  // namespace

torch::Tensor cuembed_forward(torch::Tensor flat_indices,
                              torch::Tensor offsets,
                              torch::Tensor weight) {
  check_i32_cuda_contiguous(flat_indices, "flat_indices");
  check_i32_cuda_contiguous(offsets, "offsets");
  check_f32_cuda_contiguous(weight, "weight");
  TORCH_CHECK(flat_indices.dim() == 1, "flat_indices must be rank-1");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be rank-1");
  TORCH_CHECK(offsets.numel() >= 1, "offsets must include the final offset");
  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");

  const int batch_size = static_cast<int>(offsets.numel() - 1);
  const int embed_width = static_cast<int>(weight.size(1));
  auto output = torch::empty({batch_size, embed_width}, weight.options());

  cuembed::EmbeddingForward<float, float, int32_t, int32_t, false>(
      weight.data_ptr<float>(),
      embed_width,
      flat_indices.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(),
      nullptr,
      batch_size,
      0,
      cuembed::CombineMode::kSum,
      output.mutable_data_ptr<float>(),
      at::cuda::getCurrentCUDAStream());

  return output;
}

torch::Tensor cuembed_backward(torch::Tensor flat_indices,
                               torch::Tensor offsets,
                               torch::Tensor grad_output,
                               int64_t num_features) {
  TORCH_CHECK(num_features > 0, "num_features must be positive");
  const int embed_width = static_cast<int>(grad_output.size(1));
  auto grad_weight = torch::empty({num_features, embed_width}, grad_output.options());
  cuembed_backward_checked(flat_indices, offsets, grad_output, grad_weight);
  return grad_weight;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuembed_forward", &cuembed_forward, "cuEmbed CSR embedding forward");
  m.def("cuembed_backward", &cuembed_backward, "cuEmbed COO embedding backward");
}
