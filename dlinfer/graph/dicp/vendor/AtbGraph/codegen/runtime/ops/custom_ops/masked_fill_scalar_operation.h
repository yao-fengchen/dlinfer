#pragma once

#include "ops/aclnn_ops/acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {
class MaskedFillScalarOperation : public atb::Operation {
public:
    explicit MaskedFillScalarOperation(const std::string& name, float value, const std::string& dtype);
    ~MaskedFillScalarOperation() override;
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspaceSize, atb::Context* context) override;
    atb::Status Execute(const atb::VariantPack& variantPack, uint8_t* workspace, uint64_t workspaceSize, atb::Context* context) override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;

private:
    aclTensor* CreateAclTensor(const AclNnTensor& aclNnTensor);
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
    int CreateAclTensors(const atb::VariantPack& variantPack);

private:
    std::string opName_;
    DICPScalar value_;
    aclScalar* aclValue_ = nullptr;
    DICPScalar one_;
    aclScalar* aclOne_ = nullptr;
    aclOpExecutor* aclMulsExecutor_ = nullptr;
    aclOpExecutor* aclInplaceMaskedFillScalarExecutor_ = nullptr;
    uint64_t mulsWorkspaceSize_ = 0;
    uint64_t inplaceMaskedFillScalarWorkspaceSize_ = 0;

private:
    atb::SVector<AclNnTensor> aclInTensors_;
    atb::SVector<AclNnTensor> aclOutTensors_;
};

}  // namespace dicp
