/**
* \file bottom_pool.cu
* \date 2020.06
* \author Triloon
* \brief
* \copyright
*/

#include "./bottom_pool-inl.h"
#include <vector>

namespace mxnet {
namespace op {

struct maxout_kernel {
    // Now, each loop in mxnet_generic_kernel process (kMaxGridNum * kBaseThreadNum) cols, or all cols processed in one iterator
    template <typename DType>
    __device__ __forceinline__ static void Map (int tid,
                                                DType *out_data,
                                                DType *in_data,
                                                int height,
                                                int width,
                                                int stride,
                                                int channels)
    {
        // not exceed the total col nums
        if (tid >= channels * width)
            return ;

        int c   = tid / width;
        int col = tid % width;
        int offset = c * stride * height + col;

        DType *in_start  = in_data  + offset;
        DType *out_start = out_data + offset;
        // ladkbjoidanbl
        out_start[0] = in_start[0];

        for (int row = 1; row < height; ++row)
        {
            int in_idx  = (row - 1) * stride;      // within each channel, the offset is just stride
            int out_idx = row * stride;
            out_start[out_idx] = in_start[out_idx] > out_start[in_idx] ? in_start[out_idx] : out_start[in_idx];
        }
    }
};

// template <typename DType>
// __global__ void maxout_kernel (DType *out_data, DType* in_data, int height, int width, int stride, int channels)
// {
// }
// 
/// Another scheme: blockPerGrid -> Channels, threadPerBlock -> width, then pass the corresponding col into each thread
///  in the kernel func, loop across channels, inner loop across width
// namespace bup {
// struct Kernel {
//     /*! \brief Launch GPU kernel */
//     template<typename ...Args>
//     inline static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
//         if (0 == N) return;
//         using namespace mshadow::cuda;
//         // if use the kBaseThreadNum, no need to use the Channels !
//         int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
// 
//         // int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
//         // mxnet_generic_kernel<OP, Args...>
//         // <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
//         //     N, args...);
//         // MSHADOW_CUDA_POST_KERNEL_CHECK(mxnet_generic_kernel);
//     }
// };      // struct bup::Kernel
// }       // namespace bup

struct maxout_back_kernel {
    template <typename DType>
    __device__ __forceinline__ static void Map (int tid,
                                                DType *out_grad,
                                                DType *in_grad,
                                                DType *in_data,
                                                int height,
                                                int width,
                                                int stride,
                                                int channels)
    {
        // not exceed the total col nums
        if (tid >= channels * width)
            return ;

        int c   = tid / width;
        int col = tid % width;
        int offset = c * stride * height + col;

        DType *in_data_start  = in_data + offset;
        DType *in_grad_start  = in_grad + offset;
        DType *out_grad_start = out_grad + offset;
        out_grad_start[0] = in_grad_start[0];

        int max_idx = 0;
        DType max_val = in_data_start[max_idx];
        for (int row = 1; row < height; ++row)
        {
            int curr_idx = row * stride;
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
};

template <typename DType>
class BottomUpOp<gpu, DType> {
public:
    void Init (BottomUpParam p)
    {
        this->param_ = p;
    }

    // forward function
    void Forward(const OpContext& ctx,
                 const std::vector<TBlob>& in_data,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& out_data)
    {
        using namespace mshadow;
        using namespace mshadow::expr;

        CHECK_EQ(req[bup::kOut], kWriteTo);
        size_t expected = 1;
        CHECK_EQ(in_data.size(), expected);
        CHECK_EQ(out_data.size(), 1U);
        Stream<gpu>* s = ctx.get_stream<gpu>();
        int ndims = in_data[bup::kData].ndim();
        CHECK_EQ(ndims, 4U);
        LayerSetUp(in_data[bup::kData].shape_);

        // get the I/O data, from TBlob -> Tensor, bottom pointer is shared
        Tensor<gpu, 4, DType> input_4d  = in_data[bup::kData].get<gpu, 4, DType>(s);
        Tensor<gpu, 4, DType> output_4d = out_data[bup::kOut].get<gpu, 4, DType>(s);

        for (int t = 0; t < num_; ++t)
        {
            Tensor<gpu, 4, DType> curr_in = input_4d.Slice(t, t+1);
            Tensor<gpu, 4, DType> curr_out = output_4d.Slice(t, t+1);
            int stride = curr_in.stride_;
            /// blockPerGrid -> each channel
            mxnet_op::Kernel<maxout_kernel, gpu>::Launch(s, channels_ * width_, curr_out.dptr_, curr_in.dptr_, height_, width_, stride, channels_);
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
        Stream<gpu>* s = ctx.get_stream<gpu>();
        int ndims = in_data[bup::kGrad].ndim();
        CHECK_EQ(ndims, 4U);
        LayerSetUp(in_data[bup::kGrad].shape_);     // TODO: use 'bup::kIn' instead

        Tensor<gpu, 4, DType> top_grad_in = out_grad[bup::kGrad].get<gpu, 4, DType>(s);
        Tensor<gpu, 4, DType> bottom_in = in_data[bup::kGrad].get<gpu, 4, DType>(s);
        Tensor<gpu, 4, DType> bottom_grad_out = in_grad[bup::kGrad].get<gpu, 4, DType>(s);

        for (int t = 0; t < num_; ++t)
        {
            Tensor<gpu, 4, DType> curr_grad_in = top_grad_in.Slice(t, t+1);
            Tensor<gpu, 4, DType> curr_data_in = bottom_in.Slice(t, t+1);
            Tensor<gpu, 4, DType> curr_grad_out = bottom_grad_out.Slice(t, t+1);

            int stride = curr_data_in.stride_;
            /// blockPerGrid -> each channel
            mxnet_op::Kernel<maxout_back_kernel, gpu>::Launch(s, channels_ * width_, curr_grad_out.dptr_, curr_grad_in.dptr_, curr_data_in.dptr_, height_, width_, stride, channels_);
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
};      // class partial specialized BottomUpOp class

/// specialize the implementation of BottomUpCompute & BottomUpGradCompute
template<>
void BottomUpCompute<gpu>(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx, const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
    // start here
    MSHADOW_REAL_TYPE_SWITCH(inputs[bup::kData].type_flag_, DType, {
        BottomUpOp<gpu, DType> op;
        op.Forward(ctx, inputs, req, outputs);
    });
}

template <>
void BottomUpGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx, const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob>& outputs) {

    // OR: move belowed code to the Backward() func
    std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
    const TBlob& out_grad = inputs[0];
    const std::vector<TBlob>& in_grad = outputs;

    MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        BottomUpOp<gpu, DType> op;
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    });
}

NNVM_REGISTER_OP(BottomUpPooling)
.set_attr<FCompute>("FCompute<gpu>", BottomUpCompute<gpu>);

NNVM_REGISTER_OP(_backward_BottomUpPooling)
.set_attr<FCompute>("FCompute<gpu>", BottomUpGradCompute<gpu>);

}   // namespace op
}   // namespace mxnet
