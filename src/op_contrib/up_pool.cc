/**
 * \file bottom_pool.cc
 * \brief Register the UpBottom Pooling (cpu) Op
 *  * Register Parameter
 *  * Input Lists
 *  * Shape Infer
 *  * Type infer
 * 
 * \author Triloon
 * \copyright
 */

#include "./up_pool-inl.h"
#include "../elemwise_op_common.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(UpBottomParam);

static inline std::vector<std::string> ListArguments(const UpBottomParam& param_) {
    return {"data"};
}

static bool UpBottomShape(const nnvm::NodeAttrs& attrs, 
                            mxnet::ShapeVector* in_shape,
                            mxnet::ShapeVector* out_shape) {
    using namespace mshadow;
    out_shape->resize(1, mxnet::TShape());
    const mxnet::TShape& dshape = (*in_shape)[ubp::kData];
    if (!mxnet::ndim_is_known(dshape))
        return false;
    CHECK_EQ(dshape.ndim(), 4U) << "UpBottom Pooling input data not in NCHW layout!";

    Shape<4> dshp = dshape.get<4>();
    Shape<4> oshape;
    oshape[0] = dshp[0];
    oshape[1] = dshp[1];
    oshape[2] = dshp[2];
    oshape[3] = dshp[3];

    SHAPE_ASSIGN_CHECK(*out_shape, 0, oshape);
    return true;
}

static bool UpBottomType(const nnvm::NodeAttrs& attrs, 
                            std::vector<int> *in_type, std::vector<int> *out_type) {
    CHECK_GE(in_type->size(), 1U);
    // Tblob use type_flag_ to store the data type, use MSADOW_REAL_TYPE_SWITCH to restore the real CPP type
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    const UpBottomParam& param_ = nnvm::get<UpBottomParam>(attrs.parsed);
    for (size_t i = 0; i < in_type->size(); ++i) {
        if ((*in_type)[i] == -1)
        {
            (*in_type)[i] = dtype;
        }
        else
        {
            UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
        }
    }

    out_type->clear();
    out_type->push_back(dtype);
    return true;
}

// parse the op params from attrs.dict (str->str)
// the result is stored into the attrs.parsed (any) for later use
// or use operator_common.h::ParamParser() for simplicify
void UpBottomParamParser(nnvm::NodeAttrs* attrs) {
    using namespace mshadow;
    UpBottomParam param_;
    try {
        // see ParamManager::RunInit() for more details of Init() func
        param_.Init(attrs->dict);
    } catch (const dmlc::ParamError& e) {
        std::ostringstream os;
        os << e.what();
        os << ", in operator " << attrs->op->name << "("
           << "name=\"" << attrs->name << "\"";

        throw dmlc::ParamError(os.str());
    }
    attrs->parsed = std::move(param_);
}

struct UpBottomGrad {
    const char *op_name;
    std::vector<nnvm::NodeEntry> operator() (const nnvm::ObjectPtr& n,
                                             const std::vector<nnvm::NodeEntry>& ograds) const {
        const UpBottomParam& param = nnvm::get<UpBottomParam>(n->attrs.parsed);
        std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
        heads.push_back(n->inputs[ubp::kData]);

        return MakeGradNode(op_name, n, heads, n->attrs.dict);
    }
};

NNVM_REGISTER_OP(UpBottomPooling)
.add_alias("_npx_ubpooling")
.describe(R"code(Compute NCHW input's bottom up pooling.

The usage can be found in the CornerNet & CenterNet etc)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
    return 1;
})
.set_num_outputs(1)
.set_attr_parser(UpBottomParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", UpBottomShape)
.set_attr<nnvm::FInferType>("FInferType", UpBottomType)
.set_attr<FCompute>("FCompute<cpu>", UpBottomCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", UpBottomGrad{"_backward_UpBottomPooling"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the UpBottomPooling.")
.add_arguments(UpBottomParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_UpBottomPooling)
.set_num_inputs([](const NodeAttrs& attrs) {
    // const UpBottomParam & params = nnvm::get<UpBottomParam>(attrs.parsed);
    return 2;
})
.set_num_outputs([](const NodeAttrs& attrs) {
    return 1;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(UpBottomParamParser)
.set_attr<FCompute>("FCompute<cpu>", UpBottomGradCompute<cpu>);

} // namespace op
} // namespace mxnet
