// #pragma once

// #include <bits/stdint-intn.h>
// #include <cstdint>
// #include <vector>

// #include "ops/atb_ops/atb_ops.h"

// #include "ops/aclnn_ops/acl_nn_operation.h"

// #include "utils/tensor_utils.h"


// namespace dicp {

// class CustomAllToAllVOperation : public atb::Operation {
// public:
//     explicit CustomAllToAllVOperation(const std::string& name, atb::infer::AllToAllVParam param);
//     ~CustomAllToAllVOperation() override;

//     std::string GetName() const override;
//     atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
//     atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
//     atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
//     uint32_t GetInputNum() const override;
//     uint32_t GetOutputNum() const override;

// protected:
//     mutable int64_t unit = 1;
//     atb::infer::AllToAllVParam param_;
//     std::string opName_;

// private:
//     int GetDataFromAtbtensor(const atb::Tensor& tensor, std::vector<int64_t>& counts, std::vector<int64_t> displs);
// };

// }  // namespace dicp