/**
 * \file left_pool-inl.h
 * \author smh
 * \date 2020.06
 * \brief
 * \copyright
 */

#ifndef LEFT_POOL_INL_H_
#define LEFT_POOL_INL_H_
#include <mxnet/io.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/c_api.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/imperative.h>
#include "../mxnet_op.h"
#include "../operator_common.h"

#include <algorithm>
#include <vector>
#include <iostream>

namespace mxnet {
namespace op {

/// left->right pooling
namespace lrp {
enum LeftRightInputs {kData};
enum LeftRightOutputs {kOut};
enum LeftRightGradInputs {kGrad, kIn};
}

struct LeftRightParam : public dmlc::Parameter<LeftRightParam> {
    DMLC_DECLARE_PARAMETER(LeftRightParam) {
    }
};

struct lr_maxout_thread {
    template<typename DType>
    static void Map(size_t i, DType* out_data, DType* in_data, int height, int width, int stride)
    {
        // i: i-th channel
        DType *in_start  = in_data  + int(i * height * stride);
        DType *out_start = out_data + int(i * height * stride);

        for (int row = 0; row < height; ++row)
        {
            int row_offset = row * stride + width - 1;
            out_start[row_offset] = in_start[row_offset];
            for (int col = 1; col < width; ++col)
            {
                // col >= 1 && width >= 2
                int curr_offset = row_offset - col;
                out_start[curr_offset] = std::max<DType>(in_start[curr_offset], out_start[curr_offset+1]);
            }
        }
    }
};

struct lr_maxout_back_thread {
    template<typename DType>
    static void Map(size_t i, DType* out_grad, DType* in_grad, DType* in_data, int height, int width, int stride)
    {
        int offset = i * height * stride;
        DType* in_data_start  = in_data  + offset;
        DType* in_grad_start  = in_grad  + offset;
        DType* out_grad_start = out_grad + offset;

        for (int row = 0; row < height; ++row)
        {
            int row_offset = row * stride + width - 1;
            out_grad_start[row_offset] = in_grad_start[row_offset];

            int max_idx = row_offset;      
            DType max_val = in_data_start[max_idx];
            for (int col = 1; col < width; ++col)
            {
                int curr_idx = row_offset - col;
                if (max_val < in_data_start[curr_idx])
                {
                    max_idx = curr_idx;
                    max_val = in_data_start[curr_idx];
                    out_grad_start[curr_idx] = in_grad_start[curr_idx];
                }
                else
                {
                    out_grad_start[max_idx] += in_grad_start[curr_idx];
                    out_grad_start[curr_idx] = 0;
                }
            }
        }
    }
};

template<typename xpu, typename DType>
class LeftRightOp {
public:
    void Init (LeftRightParam p)
    {
        this->param_ = p;
    }

    /// forward func
    void Forward(const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data)
    {
        using namespace mshadow;
        using namespace mshadow::expr;

        CHECK_EQ(req[lrp::kOut], kWriteTo);
        size_t expected = 1;
        CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), 1U);
        Stream<xpu>* s = ctx.get_stream<xpu>();
        int ndims = in_data[lrp::kData].ndim();
        CHECK_EQ(ndims, 4U);
        LayerSetUp(in_data[lrp::kData].shape_);

        // get the I/O data, from TBlob -> Tensor, bottom pointer is shared
        Tensor<xpu, 4, DType> input_4d  = in_data[lrp::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> output_4d = out_data[lrp::kOut].get<xpu, 4, DType>(s);

        for (int t = 0; t < num_; ++t)
        {
            Tensor<xpu, 4, DType> curr_in = input_4d.Slice(t, t+1);
            Tensor<xpu, 4, DType> curr_out = output_4d.Slice(t, t+1);
            int stride = curr_in.stride_;
            mxnet_op::Kernel<lr_maxout_thread, xpu>::Launch(s, channels_, curr_out.dptr_, curr_in.dptr_, height_, width_, stride);
        }
    }

    /// backward func
    void Backward(const OpContext &ctx,
                  const std::vector<TBlob>& out_grad,
                  const std::vector<TBlob>& in_data,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& in_grad)
    {
        using namespace mshadow;
        using namespace mshadow::expr;

        CHECK_EQ(out_grad.size(), 1U);
        CHECK_EQ(in_data.size(), 1U);
        CHECK_EQ(in_grad.size(), 1U);
        Stream<xpu>* s = ctx.get_stream<xpu>();
        int ndims = in_data[lrp::kGrad].ndim();
        CHECK_EQ(ndims, 4U);
        LayerSetUp(in_data[lrp::kGrad].shape_);     // TODO: use 'lrp::kIn' instead

        Tensor<xpu, 4, DType> top_grad_in = out_grad[lrp::kGrad].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> bottom_in = in_data[lrp::kGrad].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> bottom_grad_out = in_grad[lrp::kGrad].get<xpu, 4, DType>(s);

        for (int t = 0; t < num_; ++t)
        {
            Tensor<xpu, 4, DType> curr_grad_in = top_grad_in.Slice(t, t+1);
            Tensor<xpu, 4, DType> curr_data_in = bottom_in.Slice(t, t+1);
            Tensor<xpu, 4, DType> curr_grad_out = bottom_grad_out.Slice(t, t+1);

            int stride = curr_data_in.stride_;
            mxnet_op::Kernel<lr_maxout_back_thread, xpu>::Launch(s, channels_, curr_grad_out.dptr_, curr_grad_in.dptr_, curr_data_in.dptr_, height_, width_, stride);
        }
    }

private:
    LeftRightParam param_;
    void LayerSetUp(const mxnet::TShape& ishape)
    {
        num_        = ishape[0];
        channels_   = ishape[1];
        height_     = ishape[2];
        width_      = ishape[3];
    }
    int num_;
    int channels_;
    int width_, height_;
};      // class LeftRightOp

template<typename xpu>
void LeftRightCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx, const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {

    MSHADOW_REAL_TYPE_SWITCH(inputs[lrp::kData].type_flag_, DType, {
        LeftRightOp<xpu, DType> op;
        op.Forward(ctx, inputs, req, outputs);
    });
}

template <typename xpu>
void LeftRightGradCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx, const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob>& outputs) {

    // OR: move belowed code to the Backward() func
    std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
    const TBlob& out_grad = inputs[0];
    const std::vector<TBlob>& in_grad = outputs;

    MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        LeftRightOp<xpu, DType> op;
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    });
}


}   // namespace op
}   // namespace mxnet

#endif
