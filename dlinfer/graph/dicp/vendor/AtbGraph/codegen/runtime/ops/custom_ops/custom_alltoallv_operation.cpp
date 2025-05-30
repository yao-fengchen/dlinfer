// #include "custom_alltoallv_operation.h"
// #include "nlohmann/json.hpp"
// #include "ops/operation_creator.h"
// #include "utils/log.h"
// #include <bits/stdint-intn.h>
// #include <bits/stdint-uintn.h>
// #include <cstddef>
// #include <string>
// #include <vector>

// namespace dicp {

// const int NUM0 = 0;
// const int NUM1 = 1;
// const int NUM3 = 3;

// CustomAllToAllVOperation::CustomAllToAllVOperation(const std::string& name, atb::infer::AllToAllVParam param) : opName_(name), param_(param) {}

// CustomAllToAllVOperation::~CustomAllToAllVOperation() {}

// std::string CustomAllToAllVOperation::GetName() const {return opName_;}

// uint32_t CustomAllToAllVOperation::GetInputNum() const { return NUM3; }

// uint32_t CustomAllToAllVOperation::GetOutputNum() const { return NUM1; }

// int CustomAllToAllVOperation::GetDataFromAtbtensor(const atb::Tensor& tensor, std::vector<int64_t>& counts, std::vector<int64_t> displs) {
//     auto ret = aclrtMemcpy(counts.data(), counts.size() * sizeof(int64_t), tensor.deviceData, param_.rankSize, ACL_MEMCPY_DEVICE_TO_HOST);
//     for (size_t i = 0; i < counts.size(); ++i) {
//         counts[i] *= unit;
//         displs[i] = i == 0 ? 0 : counts[i] - counts[0];
//     }
//     return 0;
// }

// atb::Status CustomAllToAllVOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const {
//     DICP_LOG(INFO) << opName_ << " infer shape start";

//     for (size_t i = 1; i < inTensorDescs.at(0).shape.dimNum; ++i) {
//         unit *= inTensorDescs.at(0).shape.dims[i];
//     }
//     atb::SVector<atb::TensorDesc> newInTensorDescs(1, inTensorDescs.at(0));
//     atb::Operation::InferShape(newInTensorDescs, outTensorDescs);

//     DICP_LOG(INFO) << opName_ << " infer shape end";
//     return 0;
// }

// int CustomAllToAllVOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) {
//     DICP_LOG(INFO) << opName_ << " CustomAllToAllVOperation setup start";

//     const atb::Tensor scatterSizes = variantPack.inTensors.at(1);
//     const atb::Tensor gatherSizes = variantPack.inTensors.at(2);
//     std::vector<int64_t> sendCounts(param_.rankSize), sdispls(param_.rankSize), recvCounts(param_.rankSize), rdispls(param_.rankSize);
//     GetDataFromAtbtensor(scatterSizes, sendCounts, sdispls);
//     GetDataFromAtbtensor(gatherSizes, recvCounts, rdispls);
//     param_.sendCounts = sendCounts;
//     param_.sdispls = sdispls;
//     param_.recvCounts = recvCounts;
//     param_.rdispls = rdispls;

//     atb::SVector<struct atb::Tensor> inTensors(1, variantPack.inTensors.at(0));
//     atb::VariantPack newVariantPack;
//     newVariantPack.inTensors = inTensors;
//     newVariantPack.outTensors = variantPack.outTensors;
//     atb::Operation::Setup(newVariantPack, workspaceSize, context);

//     DICP_LOG(INFO) << opName_ << " CustomAllToAllVOperation setup end";
//     return 0;
// }

// int CustomAllToAllVOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context) {
//     DICP_LOG(INFO) << opName_ << " execute start";

//     atb::SVector<struct atb::Tensor> inTensors(1, variantPack.inTensors.at(0));
//     atb::VariantPack newVariantPack;
//     newVariantPack.inTensors = inTensors;
//     newVariantPack.outTensors = variantPack.outTensors;

//     atb::Operation::Execute(newVariantPack, workspace, workspaceSize, context);

//     DICP_LOG(INFO) << opName_ << " execute end";
//     return 0;
// }


// atb::Operation* CustomAllToAllVOperationCreate(const nlohmann::json& paramJson) {
//     atb::infer::AllToAllVParam param;
//     std::string opName;
//     if (paramJson.contains("name")) {
//         opName = paramJson["name"].get<std::string>();
//     }
//     if (paramJson.contains("rank")) {
//         param.rank = paramJson["rank"].get<int>();
//     }
//     if (paramJson.contains("rankSize")) {
//         param.rankSize = paramJson["rankSize"].get<int>();
//     }
//     if (paramJson.contains("rankRoot")) {
//         param.rankRoot = paramJson["rankRoot"].get<int>();
//     }
//     if (paramJson.contains("backend")) {
//         param.backend = paramJson["backend"].get<std::string>();
//     }
//     if (paramJson.contains("commMode")) {
//         auto tmp = paramJson["commMode"].get<int32_t>();
//         param.commMode = static_cast<atb::infer::CommMode>(tmp);
//     }
//     if (paramJson.contains("rankTableFile")) {
//         param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
//     }
//     if (paramJson.contains("commDomain")) {
//         param.commDomain = paramJson["commDomain"].get<std::string>();
//     }
//     atb::Operation* op = new CustomAllToAllVOperation(opName, param);
//     return op;
// }

// REGISTER_OPERATION(CustomPrepareMoeOperation, CustomAllToAllVOperationCreate);

// }