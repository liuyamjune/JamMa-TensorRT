/*
 * selective_scan_plugin.cu
 * Calls mamba_ssm's selective_scan_fwd directly — NO Python, NO dlopen.
 * Links against selective_scan_cuda.so at compile time.
 *
 * Key fix: cudaMemcpy isolates TRT memory from PyTorch memory.
 * Unlike at::from_blob (shared), at::empty + cudaMemcpy creates independent tensors.
 */

#include <cuda_runtime.h>
#include <ATen/ATen.h>

// Exact declaration matching the exported symbol in selective_scan_cuda.so:
//   _Z18selective_scan_fwdRKN2at6TensorES2_S2_S2_S2_RKSt8optionalIS0_ES6_S6_b
// Demangled:
//   std::vector<at::Tensor> selective_scan_fwd(
//       at::Tensor const&, at::Tensor const&, at::Tensor const&,
//       at::Tensor const&, at::Tensor const&,
//       std::optional<at::Tensor> const&, std::optional<at::Tensor> const&,
//       std::optional<at::Tensor> const&, bool);
//
// In C++17 with PyTorch 2.x, c10::optional == std::optional.
// Use std::optional to match the exact mangled symbol.
#include <optional>
std::vector<at::Tensor> selective_scan_fwd(
    const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C,
    const std::optional<at::Tensor>& D, const std::optional<at::Tensor>& z,
    const std::optional<at::Tensor>& delta_bias, bool delta_softplus);


namespace jam {

int selective_scan_cuda_fwd_wrapper(
    const void* u, const void* delta, const void* A,
    const void* B, const void* C, const void* D_ptr,
    const void* z, const void* delta_bias,
    void* out, int B_dim, int D_dim, int L_dim, int N_dim,
    int delta_softplus, cudaStream_t /*stream*/) {

    auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

    // CRITICAL: at::empty + cudaMemcpy, NOT at::from_blob.
    // at::from_blob creates tensors that share TRT's memory — PyTorch
    // may modify them in-place, corrupting TRT's data.
    auto copy_in = [&](const void* src, at::IntArrayRef shape) {
        at::Tensor t = at::empty(shape, opts);
        cudaMemcpy(t.data_ptr(), src, t.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        return t;
    };

    at::Tensor u_t  = copy_in(u, {B_dim, D_dim, L_dim});
    at::Tensor delta_t = copy_in(delta, {B_dim, D_dim, L_dim});
    at::Tensor A_t  = copy_in(A, {D_dim, N_dim});
    // B and C: (B, N, L) → need unsqueeze(1) to (B, 1, N, L)
    at::Tensor B_t  = copy_in(B, {B_dim, N_dim, L_dim}).unsqueeze(1);
    at::Tensor C_t  = copy_in(C, {B_dim, N_dim, L_dim}).unsqueeze(1);
    at::Tensor D_t  = copy_in(D_ptr, {D_dim});
    at::Tensor z_t  = copy_in(z, {B_dim, D_dim, L_dim});
    at::Tensor delta_bias_t = copy_in(delta_bias, {D_dim});

    // Call the real mamba_ssm kernel
    auto result = selective_scan_fwd(
        u_t, delta_t, A_t, B_t, C_t,
        std::optional<at::Tensor>(D_t),
        std::optional<at::Tensor>(z_t),
        std::optional<at::Tensor>(delta_bias_t),
        (bool)delta_softplus);

    // selective_scan_cuda.fwd returns (out, x, out_z) when z is provided.
    // selective_scan_fn returns out_z (the gated output), which is result[2].
    // JamMa's Mamba always provides z → we need result[2], not result[0].
    at::Tensor out_t = result[2].contiguous();
    cudaMemcpy(out, out_t.data_ptr(),
               B_dim * D_dim * L_dim * sizeof(float),
               cudaMemcpyDeviceToDevice);

    return cudaGetLastError() != cudaSuccess;
}

}  // namespace jam
