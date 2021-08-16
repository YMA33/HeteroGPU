#ifndef _GPU_GRADIENTDESCENT_H_
#define _GPU_GRADIENTDESCENT_H_

#include <math.h>
#include <random>
#include <stdio.h>
#include <iomanip>

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "Timer.h"

#include "SparseData.h"
#include "Para.h"
#include "FCModel.h"
#include "config.h"
#include "gd_cuda.h"

using std::ios;

class GPUGDImp : public EventProcessorImp {
private:
    idx_t rank;
	idx_t gpu_idx;

    SparseData* proc_data;
    idx_t testdata_max_row_nnz;
    idx_t testlabel_max_row_nnz;
    
	FCModel* h_model;
    vector<val_t*>* d_weight;
    
	Para* para;

    vector<val_t*> d_label;
    vector<val_t*> d_T;
    vector<val_t*> d_grad_T;

    /* CSR */
    vector<idx_t*> h_input_colIdx;
    vector<idx_t*> h_input_rowPtr;
    vector<idx_t*> h_label_colIdx;
    vector<idx_t> h_label_nnz;
    vector<idx_t*> h_label_rowPtr;
    vector<idx_t*> d_input_colIdx;
    vector<idx_t*> d_input_rowPtr;
    vector<idx_t> d_input_rowPtr_b;
    vector<idx_t*> d_label_colIdx;
    vector<idx_t> d_label_nnz;
    vector<idx_t*> d_label_rowPtr;

    /* CSC */
    vector<val_t*> d_csc_val;
    vector<idx_t*> d_csc_rowIdx;
    vector<idx_t*> d_csc_colPtr;

    /* COO */
    vector<idx_t*> h_dataX;
    vector<idx_t*> h_dataY;
    vector<idx_t*> d_dataX;
    vector<idx_t*> d_dataY;

    vector<vector<val_t*>> d_dlossy;
    vector<vector<val_t*>> d_dlossx;
    vector<val_t*> d_rowsum;
    vector<val_t*> d_exp;
    vector<int*> d_maxIdx;
    vector<idx_t*> d_maxAccuIdx;
    vector<idx_t*> h_maxAccuIdx;
    val_t* l2_sum;
    val_t* h_l2_sum;
    idx_t l2_flag;

    val_t accu;
    vector<val_t*> t_loss;
	EventProcessor scheduler;

    const val_t alpha = 1.f;
	const val_t beta = 0.;

    Timer transfer_timer;
    val_t transfer_time;

public:
    vector<vector<val_t*>> d_layers;
    val_t* h_output;
	vector<vector<val_t*>> d_gradient;
    // model averaging
    vector<vector<val_t*>*> d_glb_weight;
    vector<val_t*>* d_cb_weight;
    val_t* gpu_weight;
    val_t* cpu_weight;

    vector<cudaStream_t> gpu_streams;
	cublasHandle_t d_handle;

    GPUGDImp(EventProcessor& _scheduler);

    // destructor
    virtual ~GPUGDImp();

    void Init(SparseData* _proc_data, idx_t _testdata_max, idx_t _testlabel_max, FCModel* _model, vector<val_t*>* _d_weight, Para* _para, idx_t _gpu_idx, idx_t _rank);
    
	////////////////////////////////////////////////////////////////////////////
    MESSAGE_HANDLER_DECLARATION(RunDataLoad);
    MESSAGE_HANDLER_DECLARATION(RunAccuracy);
    MESSAGE_HANDLER_DECLARATION(RunTrain);
    MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class GPUGD : public EventProcessor {
public:

    GPUGD(EventProcessor& _scheduler) {
        evProc = new GPUGDImp(_scheduler);
    }

    virtual ~GPUGD() {}

    void Init(SparseData* _proc_data, idx_t _testdata_max, idx_t _testlabel_max, FCModel* _model, vector<val_t*>* _d_weight, Para* _para, idx_t _gpu_idx, idx_t _rank){
        GPUGDImp& obj = dynamic_cast<GPUGDImp&>(*evProc);
        obj.Init(_proc_data, _testdata_max, _testlabel_max, _model, _d_weight, _para, _gpu_idx, _rank);
    }
};
#endif /* _GPU_GRADIENTDESCENT_H_ */