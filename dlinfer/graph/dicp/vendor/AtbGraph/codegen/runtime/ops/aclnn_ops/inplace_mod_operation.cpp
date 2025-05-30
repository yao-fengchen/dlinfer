#include "inplace_mod_operation.h"

#include <algorithm>
#include <bits/stdint-intn.h>

#include "aclnnop/aclnn_fmod_scalar.h"
#include "utils/log.h"
#include "utils/misc.h"

namespace dicp {

const int NUM1 = 1;
const int NUM2 = 2;

AclNnInplaceModOperation::AclNnInplaceModOperation(const std::string& name, int64_t other) : AclNnOperation(name), other_(other) {
    aclOther_ = aclCreateScalar(&other, aclDataType::ACL_INT64);
}

AclNnInplaceModOperation::~AclNnInplaceModOperation() {
    if (aclOther_ != nullptr) {
        aclDestroyScalar(aclOther_);
    }
}

atb::Status AclNnInplaceModOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }
    return 0;
}

uint32_t AclNnInplaceModOperation::GetInputNum() const { return NUM1; }

uint32_t AclNnInplaceModOperation::GetOutputNum() const { return NUM1; }

int AclNnInplaceModOperation::SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) {
    int ret = aclnnInplaceFmodScalarGetWorkspaceSize(aclInTensors_.at(0).tensor, aclOther_, &workspaceSize, &aclExecutor_);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceFmodScalarGetWorkspaceSize end, ret:" << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << aclExecutor_;
    return ret;
}

int AclNnInplaceModOperation::CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) {
    int ret = aclnnInplaceFmodScalar(workspace, workspaceSize, aclExecutor, stream);
    DICP_LOG(INFO) << opName_ << " aclnnInplaceFmodScalar end, ret:" << ret;
    return ret;
}

atb::Operation* AclNnInplaceModOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    int64_t other;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    if (paramJson.contains("other")) {
        other = paramJson["other"].get<int64_t>();
    }
    DICP_LOG(INFO) << "AclNnInplaceModOperation: name: " << opName << ", other: " << other;
    atb::Operation* op = new AclNnInplaceModOperation(opName, other);
    return op;
}

REGISTER_OPERATION(AclNnInplaceModOperation, AclNnInplaceModOperationCreate);

}  // namespace dicp
