#ifndef _GPUGD_SCHEDULER_H_
#define _GPUGD_SCHEDULER_H_

#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <iomanip>
#include <string>

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "GPUGD.h"

using std::ios;

class GPUGDSchedulerImp : public EventProcessorImp {
private:
    idx_t rank;
    idx_t n_ranks;

    SparseData* sparse_data;
    SparseData* sparse_testdata;
    FCModel* model;
    Para* para;

    idx_t n_gpus;
    vector<GPUGD*> gpuEV;
    vector<FCModel> gpu_model;
    vector<vector<val_t*>*> gpu_d_weight;
    val_t* gpu_weight;

    // 0: train_accuracy, 1: test_accuracy, 2: train
    idx_t task_idx;
    idx_t start_idx;

    vector<idx_t> accu_send; vector<idx_t> model_send;
    vector<idx_t> train_batches;
    vector<val_t> gpu_p;
    vector<idx_t> batch_size;
    idx_t max_gpu_bsize;
    idx_t s_merg_iter, merg_iter;
    idx_t train_tuples, train_idx;
    idx_t test_batches;
    idx_t l2_flag;

    vector<Timer> sche_data_timer;
    vector<val_t> sche_data_time; vector<val_t> pure_data_time;
    Timer task_testaccu_timer; val_t task_testaccu_time;
    vector<Timer> sche_testaccu_timer;
    vector<val_t> sche_testaccu_time; vector<val_t> pure_testaccu_time;
    Timer task_train_timer; val_t task_train_time;
    vector<Timer> sche_train_timer;
    vector<val_t> sche_train_time; vector<val_t> pure_train_time;
    Timer task_sampleaccu_timer; val_t task_sampleaccu_time;
    Timer mpi_timer;
    vector<val_t> mpi_time;
    Timer transfer_timer, localavg_timer;
    val_t transfer_time, localavg_time;

    val_t test_accuracy, sample_accuracy, test_loss, p_test_loss, sample_loss;
    idx_t incloss_cnt;
    idx_t train_iter, sample_tuples;
        cublasHandle_t d_handle;

public:

    GPUGDSchedulerImp();
    virtual ~GPUGDSchedulerImp();
    void RunGPUGD(SparseData* _sparse_data, SparseData* _sparse_testdata, FCModel* _model, Para* _para, idx_t _n_ranks, idx_t _rank, idx_t _n_gpus);

    ////////////////////////////////////////////////////////////////////////////
    MESSAGE_HANDLER_DECLARATION(GPUGD_Accuracy);
    MESSAGE_HANDLER_DECLARATION(GPUGD_Train);
    MESSAGE_HANDLER_DECLARATION(GPUGD_DataLoad);
    MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class GPUGDScheduler : public EventProcessor {
public:
    GPUGDScheduler() {
        evProc = new GPUGDSchedulerImp();
    }

    virtual ~GPUGDScheduler() {}

    void RunGPUGD(SparseData* _sparse_data, SparseData* _sparse_testdata, FCModel* _model, Para* _para, idx_t _n_ranks, idx_t _rank, idx_t _n_gpus){
        GPUGDSchedulerImp& obj = dynamic_cast<GPUGDSchedulerImp&>(*evProc);
        obj.RunGPUGD(_sparse_data, _sparse_testdata, _model, _para, _n_ranks, _rank, _n_gpus);
    }
};

#endif /* _GPUGD_SCHEDULER_H_ */
