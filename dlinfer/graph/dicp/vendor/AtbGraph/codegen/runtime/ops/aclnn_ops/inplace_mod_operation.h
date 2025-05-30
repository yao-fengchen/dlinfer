#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"
#include <bits/stdint-intn.h>

namespace dicp {
class AclNnInplaceModOperation : public AclNnOperation {
public:
    explicit AclNnInplaceModOperation(const std::string& name, int64_t other);
    ~AclNnInplaceModOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int64_t other_;
    aclScalar* aclOther_ = nullptr;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
