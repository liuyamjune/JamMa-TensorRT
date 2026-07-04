/*
 * selective_scan_plugin.cpp
 * TensorRT Plugin for Mamba Selective Scan — ONNX Parser & Builder
 */

#include "selective_scan_plugin.hpp"
#include <cstring>
#include <cassert>
#include <cuda_fp16.h>

namespace jam {

// ============================================================================
// Plugin implementation
// ============================================================================

SelectiveScanPlugin::SelectiveScanPlugin(int delta_softplus)
    : mDeltaSoftplus(delta_softplus) {}

SelectiveScanPlugin::SelectiveScanPlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    mDeltaSoftplus = *reinterpret_cast<const int32_t*>(d);
}

const char* SelectiveScanPlugin::getPluginVersion() const noexcept {
    return "1";
}

const char* SelectiveScanPlugin::getPluginType() const noexcept {
    return "SelectiveScan";
}

int32_t SelectiveScanPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SelectiveScanPlugin::initialize() noexcept { return 0; }
void SelectiveScanPlugin::terminate() noexcept {}

size_t SelectiveScanPlugin::getSerializationSize() const noexcept {
    return sizeof(int32_t);
}

void SelectiveScanPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int32_t*>(d) = mDeltaSoftplus;
}

void SelectiveScanPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* SelectiveScanPlugin::clone() const noexcept {
    auto* p = new SelectiveScanPlugin(mDeltaSoftplus);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

void SelectiveScanPlugin::setPluginNamespace(
    const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* SelectiveScanPlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace.c_str();
}

nvinfer1::DataType SelectiveScanPlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept {
    // Output same dtype as u (input 0)
    return inputTypes[0];
}

// --------------------------------------------------------------------------
// Output shape: (B, D, L)  — same as u (input 0)
// --------------------------------------------------------------------------
nvinfer1::DimsExprs SelectiveScanPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];  // B
    ret.d[1] = inputs[0].d[1];  // D
    ret.d[2] = inputs[0].d[2];  // L
    return ret;
}

// --------------------------------------------------------------------------
// All 8 inputs + 1 output must be float (kFLOAT) and linear (kNCHW not used,
// we use plain 3D tensor layout that TRT passes through as is).
// --------------------------------------------------------------------------
bool SelectiveScanPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {

    // All inputs and outputs must be float and linear
    const nvinfer1::PluginTensorDesc& desc = inOut[pos];
    if (desc.type != nvinfer1::DataType::kFLOAT) return false;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    return true;
}

void SelectiveScanPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
    // Nothing to configure — shapes are dynamic
}

// --------------------------------------------------------------------------
// No extra workspace needed — selective_scan_cuda manages its own memory
// --------------------------------------------------------------------------
size_t SelectiveScanPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
    return 0;
}

// --------------------------------------------------------------------------
// Forward declaration — implemented in .cu file
// --------------------------------------------------------------------------
int selective_scan_cuda_fwd_wrapper(
    const void* u,        // (B, D, L)  float
    const void* delta,    // (B, D, L)  float
    const void* A,        // (D, N)     float
    const void* B,        // (B, N, L)  float  (or (B,1,N,L))
    const void* C,        // (B, N, L)  float  (or (B,1,N,L))
    const void* D,        // (D,)       float
    const void* z,        // (B, D, L)  float
    const void* delta_bias,  // (D,)     float
    void* out,             // (B, D, L)  float
    int B_dim, int D_dim, int L_dim,
    int N_dim,
    int delta_softplus,
    cudaStream_t stream);

// --------------------------------------------------------------------------
// Main execution — call the CUDA kernel
// --------------------------------------------------------------------------
int32_t SelectiveScanPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    // Input layout (all linear):
    // 0: u          (B, D, L)
    // 1: delta      (B, D, L)
    // 2: A          (D, N)      — static/constant
    // 3: B          (B, N, L)
    // 4: C          (B, N, L)
    // 5: D          (D,)
    // 6: z          (B, D, L)
    // 7: delta_bias (D,)

    int B = inputDesc[0].dims.d[0];
    int D = inputDesc[0].dims.d[1];
    int L = inputDesc[0].dims.d[2];
    int N = inputDesc[2].dims.d[1];

    return selective_scan_cuda_fwd_wrapper(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], inputs[6], inputs[7],
        outputs[0],
        B, D, L, N, mDeltaSoftplus, stream);
}

// ============================================================================
// Plugin Creator
// ============================================================================

nvinfer1::PluginFieldCollection SelectiveScanPluginCreator::mFC{};
std::vector<nvinfer1::PluginField>
    SelectiveScanPluginCreator::mPluginAttributes;

SelectiveScanPluginCreator::SelectiveScanPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("delta_softplus", nullptr,
                              nvinfer1::PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SelectiveScanPluginCreator::getPluginName() const noexcept {
    return "SelectiveScan";
}

const char* SelectiveScanPluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection*
SelectiveScanPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* SelectiveScanPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
    int32_t delta_softplus = 0;
    for (int32_t i = 0; i < fc->nbFields; i++) {
        if (std::strcmp(fc->fields[i].name, "delta_softplus") == 0) {
            delta_softplus = *static_cast<const int32_t*>(fc->fields[i].data);
        }
    }
    auto* p = new SelectiveScanPlugin(delta_softplus);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::IPluginV2* SelectiveScanPluginCreator::deserializePlugin(
    const char* name, const void* serialData,
    size_t serialLength) noexcept {
    auto* p = new SelectiveScanPlugin(serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void SelectiveScanPluginCreator::setPluginNamespace(
    const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* SelectiveScanPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// ============================================================================
// Plugin Registration (called at library load time)
// ============================================================================

REGISTER_TENSORRT_PLUGIN(SelectiveScanPluginCreator);

} // namespace jam
