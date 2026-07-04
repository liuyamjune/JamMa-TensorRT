/*
 * selective_scan_plugin.hpp
 * TensorRT Plugin for Mamba Selective Scan
 *
 * Wraps mamba_ssm's selective_scan_cuda.fwd() CUDA kernel
 * as a TensorRT custom plugin.
 *
 * ONNX node: jam::SelectiveScan
 *   Inputs:  u(B,D,L), delta(B,D,L), A(D,N), B(B,N,L), C(B,N,L),
 *            D(D,), z(B,D,L), delta_bias(D,)
 *   Output: out(B,D,L)
 *   Attrs:  delta_softplus (int)
 */

#ifndef JAM_SELECTIVE_SCAN_PLUGIN_HPP
#define JAM_SELECTIVE_SCAN_PLUGIN_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace jam {

// ==========================================================================
// Plugin class
// ==========================================================================

class SelectiveScanPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    SelectiveScanPlugin(int delta_softplus);
    SelectiveScanPlugin(const void* data, size_t length);
    ~SelectiveScanPlugin() override = default;

    // --- IPluginV2 ---
    const char* getPluginVersion() const noexcept override;
    const char* getPluginType() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index, const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const noexcept override;

    // --- IPluginV2Ext ---
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // --- IPluginV2DynamicExt ---
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;

private:
    int32_t mDeltaSoftplus;
    std::string mNamespace;
    std::string mPluginNamespace;
};

// ==========================================================================
// Plugin Creator (for ONNX parser integration)
// ==========================================================================

class SelectiveScanPluginCreator : public nvinfer1::IPluginCreator {
public:
    SelectiveScanPluginCreator();
    ~SelectiveScanPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace jam

#endif // JAM_SELECTIVE_SCAN_PLUGIN_HPP
