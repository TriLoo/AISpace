/**
 * \file bottom_pool-inl.h
 * \brief
 * \author Triloon
 */

#ifndef BOTTOM_POOL_INL_H_
#define BOTTOM_POOL_INL_H_

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

/// bottom up pool namespace
namespace bup {
enum BottomUpInputs {kData};
enum BottomUpOutputs {kOut};
enum BottomUpGradInputs {kGrad, kIn};
}

struct BottomUpParam : public dmlc::Parameter<BottomUpParam> {
    /// instance `ParamManager` & `__DECLARE__()`
    /// do nothing here
    DMLC_DECLARE_PARAMETER(BottomUpParam) {
    }
};

struct maxout_thread {
    template<typename DType>
    static void Map(size_t i, DType* out_data, DType* in_data, int height, int width, int stride)
    {
        DType *in_start  = in_data + int(i * height * stride);
        DType *out_start = out_data + int(i * height * stride);
        // init the first row of output
        for (int col = 0; col < width; ++col)
            out_start[col] = in_start[col];
        // actually, this is a 2D spatial ergodic
        for (int row = 1; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                int in_offset = (row - 1) * stride + col;
                int out_offset = row * stride + col;
                out_start[out_offset] = std::max<DType>(in_start[out_offset], out_start[in_offset]);
            }
        }
    }
};

struct maxout_back_thread {
    template<typename DType>
    static void Map(size_t i, DType* out_grad, DType* in_grad, DType* in_data, int height, int width, int stride)
    {
        int offset = i * height * stride;
        DType* in_data_start  = in_data  + offset;
        DType* in_grad_start  = in_grad  + offset;
        DType* out_grad_start = out_grad + offset;

        for (int col = 0; col < width; ++col)
            out_grad_start[col] = in_grad_start[col];

        for (int col = 0; col < width; ++col)
        {
            int max_idx = col;         // only store the offset is OK, see MaxPooling implementation
            DType max_val = in_data_start[max_idx];
            for (int row = 1; row < height; ++row)
            {
                int curr_idx = row * stride + col;
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
class BottomUpOp {
public:
    void Init (BottomUpParam p)
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

        CHECK_EQ(req[bup::kOut], kWriteTo);
        size_t expected = 1;
        CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), 1U);
        Stream<xpu>* s = ctx.get_stream<xpu>();
        int ndims = in_data[bup::kData].ndim();
        CHECK_EQ(ndims, 4U);
        LayerSetUp(in_data[bup::kData].shape_);

        // get the I/O data, from TBlob -> Tensor, bottom pointer is shared
        Tensor<xpu, 4, DType> input_4d = in_data[bup::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> output_4d = out_data[bup::kOut].get<xpu, 4, DType>(s);

        /// do the forward, bottom-up pooling: calculate from top -> bottom
        // the impmlementation of gluon.nn.MaxPool2D is also based on a 6-folder loop, i.e. NCHW & Kernel WH
        // no need any temp space
        // Tensor <xpu, 1, DType> workspace = ctx.requested[bup::kTempSpace].get_space_typed<xpu, 1, DType>(Shape1(width), s);
        // no need to use MSHADOW_REAL_TYPE_SWITCH, the DType is already infered in the Compute Func, see below: BottomUpCompute()
        // MSHADOW_REAL_TYPE_SWITCH(in_data[bup::kData].type_flag_, DType,
        // {});
        for (int t = 0; t < num_; ++t)
        {
            Tensor<xpu, 4, DType> curr_in = input_4d.Slice(t, t+1);
            Tensor<xpu, 4, DType> curr_out = output_4d.Slice(t, t+1);
            int stride = curr_in.stride_;
            mxnet_op::Kernel<maxout_thread, xpu>::Launch(s, channels_, curr_out.dptr_, curr_in.dptr_, height_, width_, stride);
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
        int ndims = in_data[bup::kGrad].ndim();
        CHECK_EQ(ndims, 4U);
        LayerSetUp(in_data[bup::kGrad].shape_);     // TODO: use 'bup::kIn' instead

        Tensor<xpu, 4, DType> top_grad_in = out_grad[bup::kGrad].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> bottom_in = in_data[bup::kGrad].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> bottom_grad_out = in_grad[bup::kGrad].get<xpu, 4, DType>(s);

        for (int t = 0; t < num_; ++t)
        {
            Tensor<xpu, 4, DType> curr_grad_in = top_grad_in.Slice(t, t+1);
            Tensor<xpu, 4, DType> curr_data_in = bottom_in.Slice(t, t+1);
            Tensor<xpu, 4, DType> curr_grad_out = bottom_grad_out.Slice(t, t+1);

            int stride = curr_data_in.stride_;
            mxnet_op::Kernel<maxout_back_thread, xpu>::Launch(s, channels_, curr_grad_out.dptr_, curr_grad_in.dptr_, curr_data_in.dptr_, height_, width_, stride);
        }
    }

private:
    BottomUpParam param_;
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
};      // class BottomUpOp

template<typename xpu>
void BottomUpCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx, const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {

    MSHADOW_REAL_TYPE_SWITCH(inputs[bup::kData].type_flag_, DType, {
        BottomUpOp<xpu, DType> op;
        op.Forward(ctx, inputs, req, outputs);
    });
}

template <typename xpu>
void BottomUpGradCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx, const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob>& outputs) {

    // OR: move belowed code to the Backward() func
    std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
    const TBlob& out_grad = inputs[0];
    const std::vector<TBlob>& in_grad = outputs;

    MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        BottomUpOp<xpu, DType> op;
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    });
}

}       // namespace op
}       // namespace mxnet

#endif
