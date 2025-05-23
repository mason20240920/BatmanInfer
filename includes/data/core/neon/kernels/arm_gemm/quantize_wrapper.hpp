//
// Created by Mason on 2025/1/14.
//

#pragma once

#include <cpu/kernels/assembly/bi_arm_gemm.hpp>

#include "barrier.hpp"
#include "gemm_implementation.hpp"
#include "quantized.hpp"

namespace BatmanGemm {
    /* Quantized wrapper - do an integer GEMM and wrap around the quantization. */
    template<typename To, typename Tr, typename Tgemm>
    class QuantizeWrapper : public BIGemmCommon<To, To, Tr> {
    private:
        UniqueGemmCommon<To, To, Tgemm> _subgemm = nullptr;
        int32_t *_row_sums = nullptr;
        int32_t *_col_sums = nullptr;
        Requantize32 _params;
        GemmArgs _args;
        barrier _barrier;

        void *working_space = nullptr;
        bool arrays_set = false;

        /* We need a subgemm which outputs the 32-bit intermediates - how much space is needed for that? */
        size_t subgemm_output_size() const {
            return (_args._Msize * _args._Nsize * _args._nbatches * _args._nmulti * sizeof(int32_t));
        }

        size_t col_sum_size() const {
            return (_args._Nsize * _args._nmulti * sizeof(int32_t));
        }

        size_t row_sum_size() const {
            return (_args._Msize * _args._nbatches * _args._nmulti * sizeof(int32_t));
        }

        /* Local working space: We need space for the subgemm output (above) and
         * the row sums.  */
        size_t local_working_size() const {
            return subgemm_output_size() + row_sum_size();
        }

        void set_child_arrays() {
            if (working_space == nullptr || arrays_set == false)
                return;

            auto &g_array = this->_gemm_array;
            /* Use the first part of our working space for the subgemm result, pass the operand details straight through. */
            _subgemm->set_arrays(g_array._Aptr, g_array._lda, g_array._A_batch_stride, g_array._A_multi_stride,
                                 g_array._Bptr, g_array._ldb, g_array._B_multi_stride,
                                 reinterpret_cast<Tgemm *>(working_space), _args._Nsize, (_args._Nsize * _args._Msize),
                                 (_args._Nsize * _args._Msize * _args._nbatches),
                                 nullptr, 0);
        }

        void col_sums_pretransposed(const To *B, const int ldb, const int B_multi_stride) {
            for (unsigned int multi = 0; multi < _args._nmulti; multi++) {
                compute_col_sums(_params, _args._Nsize, _args._Ksize, B + (multi * B_multi_stride), ldb,
                                 _col_sums + (multi * _args._Nsize), _args._Ksize, multi, 0);
            }
        }

        void requantize_runtime(unsigned int threadid) {
            unsigned int first_row = (threadid * _args._Msize) / _args._maxthreads;
            unsigned int last_row = ((threadid + 1) * _args._Msize) / _args._maxthreads;
            auto &g_array = this->_gemm_array;

            for (unsigned int multi = 0; multi < _args._nmulti; multi++) {
                for (unsigned int batch = 0; batch < _args._nbatches; batch++) {
                    /* Compute row sums now */
                    compute_row_sums(_params, _args._Ksize, (last_row - first_row),
                                     g_array._Aptr + (multi * g_array._A_multi_stride) +
                                     (batch * g_array._A_batch_stride) + (first_row * g_array._lda), g_array._lda,
                                     _row_sums +
                                     (multi * _args._nbatches * _args._Msize) + (batch * _args._Msize) + first_row);
                    // If we don't care about negative values, call the version of this function that doesn't correct before shifting.
                    // 'c_offset' represents zero, so if the lowest possible quantized output value is the same or more than that we will not output negative numbers.
                    requantize_block_32(_params, _args._Nsize, (last_row - first_row),
                                        reinterpret_cast<Tgemm *>(working_space) +
                                        (multi * (_args._Msize * _args._Nsize * _args._nbatches)) +
                                        (batch * (_args._Msize * _args._Nsize)) +
                                        (first_row * _args._Nsize), _args._Nsize,
                                        g_array._Cptr + (multi * g_array._C_multi_stride) +
                                        (batch * g_array._C_batch_stride) + (first_row * g_array._ldc), g_array._ldc,
                                        _row_sums +
                                        (multi * _args._nbatches * _args._Msize) + (batch * _args._Msize) + first_row,
                                        _col_sums +
                                        (multi * _args._Nsize), 0);
                }
            }
        }

    public:
        QuantizeWrapper(const QuantizeWrapper &) = delete;

        QuantizeWrapper operator=(const QuantizeWrapper &) = delete;

        QuantizeWrapper(const GemmArgs &args, const Requantize32 &qp) : _params(qp), _args(args),
                                                                        _barrier(args._maxthreads) {
            GemmArgs newargs = GemmArgs(args._ci, args._Msize, args._Nsize, args._Ksize, args._Ksections,
                                        args._nbatches, args._nmulti, args._indirect_input, Activation(),
                                        args._maxthreads);
            _subgemm = gemm<To, To, Tgemm>(newargs);

            if (_subgemm == nullptr) {
                return;
            }
        }

        void set_arrays(const To *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                        const To *B, const int ldb, const int B_multi_stride,
                        Tr *C, const int ldc, const int C_batch_stride, const int C_multi_stride,
                        const Tr *bias, const int bias_multi_stride) override {
            BIGemmCommon<To, To, Tr>::set_arrays(A, lda, A_batch_stride, A_multi_stride, B, ldb, B_multi_stride, C, ldc,
                                                 C_batch_stride, C_multi_stride, bias, bias_multi_stride);

            arrays_set = true;
            set_child_arrays();
        }

        ndrange_t get_window_size() const override {
            return {_subgemm->get_window_size()};
        }

        void set_nthreads(int nthreads) override {
            _subgemm->set_nthreads(nthreads);
            _barrier.set_nthreads(nthreads);
            _args._maxthreads = nthreads;
        }

        bool set_dynamic_M_size(int M_size) override {
            // TODO: fixed for the future
            return false;
        }

        bool set_dynamic_batch_size(int batch_size) override {
            // TODO: fixed for the future
            return false;
        }

        void update_parameters() override {
            // TODO: For future dynamic
        }

        bool set_dynamic_nmulti_size(int nmulti) override {
            // TODO: fixed for the future
            return false;
        }

        bool set_dynamic_K_size(int K_size) override {
            return false;
        }

        bool set_dynamic_N_size(int N_size) override {
            return false;
        }

        // TODO: Make this actually stateless. This still uses the stateful
        // execution data because it requires a workspace which would also need to
        // be handled statelessly.
        void execute_stateless(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid,
                               BIGemmArrays<To, To, Tr> &) override {
            _subgemm->execute(work_range, thread_locator, threadid);

            _barrier.arrive_and_wait();

            requantize_runtime(threadid);
        }

        void execute(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid) override {
            execute_stateless(work_range, thread_locator, threadid, this->_gemm_array);
        }

        size_t get_working_size() const override {
            return _subgemm->get_working_size() + local_working_size();
        }

        // Space arrangement:

        // ptr
        // V
        // | subgemm output | row_sums | subgemm working space |
        void set_working_space(void *space) override {
            uintptr_t space_int = reinterpret_cast<uintptr_t>(space);

            working_space = space;
            _subgemm->set_working_space(reinterpret_cast<void *>(space_int + local_working_size()));

            _row_sums = reinterpret_cast<int32_t *>(space_int + subgemm_output_size());

            set_child_arrays();
        }

        bool B_is_pretransposed() const override {
            /* We clear this flag if the subgemm isn't pretransposed, so just return its value */
            return _subgemm->B_is_pretransposed();
        }

        bool B_pretranspose_required() const override {
            return _subgemm->B_pretranspose_required();
        }

        size_t get_B_pretransposed_array_size() const override {
            return _subgemm->get_B_pretransposed_array_size() + col_sum_size();
        }

        void requantize_bias(void *in_buffer, const To *B, const int ldb, const int B_multi_stride) override {
            _col_sums = reinterpret_cast<int32_t *>(in_buffer);
            col_sums_pretransposed(B, ldb, B_multi_stride);
        }

        void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride,
                                  bool transposed) override {
            assert(!transposed);

            uintptr_t buffer_int = reinterpret_cast<uintptr_t>(buffer);
            _subgemm->pretranspose_B_array(reinterpret_cast<void *>(buffer_int + col_sum_size()), B, ldb,
                                           B_multi_stride, transposed);

            requantize_bias(buffer, B, ldb, B_multi_stride);
        }

        void set_pretransposed_B_data(void *buffer) override {
            uintptr_t buffer_int = reinterpret_cast<uintptr_t>(buffer);
            _subgemm->set_pretransposed_B_data(reinterpret_cast<void *>(buffer_int + col_sum_size()));
            _col_sums = reinterpret_cast<int32_t *>(buffer);
        }

        void set_quantized_bias(const int32_t *bias, size_t bias_multi_stride) override {
            _params.bias = bias;
            _params.bias_multi_stride = bias_multi_stride;
        }

        GemmConfig get_config() override {
            GemmConfig c = _subgemm->get_config();

            std::string n = "quantize_wrapper[";
            n.append(c.filter);
            n.append("]");

            c.method = GemmMethod::QUANTIZE_WRAPPER;
            c.filter = n;

            return c;
        }

        void update_quantization_parameters(const Requantize32 &re) override {
            _params.bias = re.bias;
            _params.a_offset = re.a_offset;
            _params.b_offset = re.b_offset;
            _params.c_offset = re.c_offset;
            _params.per_layer_left_shift = re.per_layer_left_shift;
            _params.per_layer_right_shift = re.per_layer_right_shift;
            _params.per_layer_mul = re.per_layer_mul;
            _params.per_channel_requant = re.per_channel_requant;
            _params.per_channel_left_shifts = re.per_channel_left_shifts;
            _params.per_channel_right_shifts = re.per_channel_right_shifts;
            _params.per_channel_muls = re.per_channel_muls;
            _params.minval = re.minval;
            _params.maxval = re.maxval;
        }
    };
} // namespace BatmanGemm
