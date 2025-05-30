#include "atb_ops.h"
#include <string>

namespace dicp {

atb::Operation* AllToAllOperationCreate(const nlohmann::json& paramJson) {
    atb::infer::AllToAllParam param;
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("rankRoot")) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("backend")) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        auto tmp = paramJson["commMode"].get<int32_t>();
        param.commMode = static_cast<atb::infer::CommMode>(tmp);
    }
    if (paramJson.contains("rankTableFile")) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.contains("commDomain")) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    if (paramJson.contains("transpose")) {
        param.transpose = paramJson["transpose"].get<bool>();
    }
    DICP_LOG(INFO) << "AllToAllParam: rank: " << param.rank << ", rankSize: " << param.rankSize << ", rankRoot: " << param.rankRoot << ", backend: " << param.backend << ", rankTableFile: " << param.rankTableFile;
    atb::Operation* op = nullptr;
    CREATE_OPERATION_NO_RETURN(param, &op);
    return op;
}

REGISTER_ATB_OPERATION("AllToAllOperation", AllToAllOperationCreate);

}  // namespace dicp
