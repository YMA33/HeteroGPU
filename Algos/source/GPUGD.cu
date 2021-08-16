#include <chrono>
#include <thread>
#include "GPUGD.h"

extern char* LOG_DIRECTORY;

GPUGDImp::GPUGDImp(EventProcessor& _scheduler){

    scheduler.CopyFrom(_scheduler);

    RegisterMessageProcessor(GPUGD_RunDataLoadMessage::type, &RunDataLoad, 100);
    RegisterMessageProcessor(GPUGD_RunAccuracyMessage::type, &RunAccuracy, 100);
    RegisterMessageProcessor(GPUGD_RunTrainMessage::type, &RunTrain, 100);
    RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


GPUGDImp::~GPUGDImp(){

    cudaSetDevice(gpu_idx);

    idx_t batches = para->num_mbatches;
    batches = 1;

    for(idx_t i = 0; i < batches; ++i){
        for(idx_t j = 0; j < h_model->num_layers-1; ++j){
            cudaFree(d_layers[i][j+1]);
            cudaFree(d_gradient[i][j]);
            cudaFree(d_dlossy[i][j]);
            cudaFree(d_dlossx[i][j]);

        }

        free(h_label_colIdx[i]);
        free(h_label_rowPtr[i]);
        free(h_dataX[i]);
        free(h_dataY[i]);

        cudaFree(d_layers[i][0]);
        cudaFree(d_input_rowPtr[i]);
        cudaFree(d_dataX[i]);
        cudaFree(d_dataY[i]);

        cudaFree(d_label[i]);
        cudaFree(d_label_colIdx[i]);
        cudaFree(d_label_rowPtr[i]);

        cudaFree(d_rowsum[i]);
        cudaFree(d_maxIdx[i]);
        cudaFree(t_loss[i]);
        cudaFree(d_maxAccuIdx[i]);
        free(h_maxAccuIdx[i]);
	}
    cudaFree(l2_sum);
    free(h_l2_sum);

    for(idx_t i = 0; i < 5; ++i){
        for(idx_t j = 0; j < h_model->num_layers-1; ++j)
            cudaFree((*(d_glb_weight[i]))[j]);
        delete d_glb_weight[i];
    }

    cudaFree(gpu_weight);
    cudaFree(cpu_weight);

    for(idx_t i = 0; i < para->num_blocks; ++i)   cudaStreamDestroy(gpu_streams[i]);
    cublasDestroy(d_handle);

        ofDebug.close();
}


void GPUGDImp::Init(SparseData* _proc_data, idx_t _testdata_max, idx_t _testlabel_max, FCModel* _model, vector<val_t*>* _d_weight, Para* _para, idx_t _gpu_idx, idx_t _rank){
    rank = _rank;
    gpu_idx = _gpu_idx;
    cudaSetDevice(gpu_idx);
    cublasCreate(&d_handle);

    ofDebug.open((string(LOG_DIRECTORY) + "GPU-GD-" + std::to_string(gpu_idx) + ".log.rank").c_str() + std::to_string(rank), ios::out);

    proc_data = _proc_data;
    testdata_max_row_nnz = _testdata_max;
    testlabel_max_row_nnz = _testlabel_max;
    h_model = _model;
    d_weight = _d_weight;
    para = _para;

    para->stepsize = para->init_stepsize;

    accu = 0.f;
    transfer_time = 0.;

    gpu_streams.resize(para->num_blocks);
    for(idx_t i = 0; i < para->num_blocks; ++i)   cudaStreamCreate(&gpu_streams[i]);
    std::cerr<<"create " << para->num_blocks << " gpu streams" << std::endl;

    idx_t batches = para->num_mbatches;
    batches = 1;
    idx_t max_tuples = para->max_batch_size;

    d_layers.resize(batches);
    d_label.resize(batches);
    d_gradient.resize(batches);

    h_label_colIdx.resize(batches);
    h_label_rowPtr.resize(batches);
    h_dataX.resize(batches);
    h_dataY.resize(batches);

    d_input_rowPtr.resize(batches);
    d_input_rowPtr_b.resize(batches);
    d_dataX.resize(batches);
    d_dataY.resize(batches);
    d_label_colIdx.resize(batches);
    d_label_rowPtr.resize(batches);

    d_dlossy.resize(batches);
    d_dlossx.resize(batches);
    d_rowsum.resize(batches);
    d_maxIdx.resize(batches);
    t_loss.resize(batches);
    d_maxAccuIdx.resize(batches);
    h_maxAccuIdx.resize(batches);

    idx_t t_data_max = (proc_data->data_max_row_nnz > testdata_max_row_nnz)?
        proc_data->data_max_row_nnz : testdata_max_row_nnz;
    idx_t t_label_max = (proc_data->label_max_row_nnz > testlabel_max_row_nnz)?
        proc_data->label_max_row_nnz : testlabel_max_row_nnz;

    long long mem_cnt = 0;
    for(idx_t i = 0; i < batches; ++i){

        d_layers[i].resize(h_model->num_layers);
        d_gradient[i].resize(h_model->num_layers-1);
        d_dlossy[i].resize(h_model->num_layers-1);
        d_dlossx[i].resize(h_model->num_layers-1);

        h_label_colIdx[i] = (idx_t*)malloc(sizeof(idx_t)*max_tuples*t_label_max);
        h_label_rowPtr[i] = (idx_t*)malloc(sizeof(idx_t)*(max_tuples+1));

        h_dataX[i] = (idx_t*)malloc(sizeof(idx_t)*max_tuples*t_data_max);
        h_dataY[i] = (idx_t*)malloc(sizeof(idx_t)*max_tuples*t_data_max);
        cudaMalloc(&d_dataX[i], sizeof(idx_t)*max_tuples*t_data_max);
        cudaMalloc(&d_dataY[i], sizeof(idx_t)*max_tuples*t_data_max);

        cudaMalloc(&d_layers[i][0], sizeof(val_t)*max_tuples*t_data_max);
        //mem_cnt += 3*max_tuples*t_data_max;
        cudaMalloc(&d_input_rowPtr[i], sizeof(idx_t)*(max_tuples+1));
        //mem_cnt += max_tuples+1;
        
        cudaMalloc(&d_label[i], sizeof(val_t)*max_tuples*t_label_max);
        cudaMalloc(&d_label_colIdx[i], sizeof(idx_t)*max_tuples*t_label_max);
        cudaMalloc(&d_label_rowPtr[i], sizeof(idx_t)*(max_tuples+1));
        //mem_cnt += 2*max_tuples*t_label_max;
        //mem_cnt += max_tuples+1;
        
        cudaMalloc(&d_rowsum[i], sizeof(val_t)*max_tuples);
        cudaMalloc(&d_maxIdx[i], sizeof(int)*max_tuples);
        cudaMalloc(&t_loss[i], sizeof(val_t)*max_tuples);
        cudaMalloc(&d_maxAccuIdx[i], sizeof(idx_t)*max_tuples);
        h_maxAccuIdx[i] = (idx_t*)mkl_calloc(max_tuples, sizeof(idx_t), 32);
        //mem_cnt += 2*max_tuples;
        
        for(idx_t j = 0; j < h_model->num_layers-1; ++j){
            cudaMalloc(&d_gradient[i][j], sizeof(val_t)*h_model->num_units[j]*h_model->num_units[j+1]);
            cudaMalloc(&d_layers[i][j+1], sizeof(val_t)*max_tuples*h_model->num_units[j+1]);
            cudaMalloc(&d_dlossy[i][j], sizeof(val_t)*max_tuples*h_model->num_units[j+1]);
            cudaMalloc(&d_dlossx[i][j], sizeof(val_t)*max_tuples*h_model->num_units[j+1]);
            //mem_cnt += 2*h_model->num_units[j]*h_model->num_units[j+1];
            //mem_cnt += 3*max_tuples*h_model->num_units[j+1];
        }
        
		std::cerr<<"cuda-malloc size:" << 4*mem_cnt << ", "<< 4*mem_cnt/1024/1024<<"(MB)"<<std::endl;
    }
    
	cudaMalloc(&l2_sum, sizeof(val_t)*(h_model->num_layers-1));
    h_l2_sum = (val_t*)malloc(sizeof(val_t)*(h_model->num_layers-1));
    l2_flag = 0;

    d_glb_weight.resize(5);
    for(idx_t i = 0; i < 5; ++i){
        d_glb_weight[i] = new vector<val_t*>();
        d_glb_weight[i]->resize(h_model->num_layers-1);
        for(idx_t j = 0; j < h_model->num_layers-1; ++j){
            cudaMalloc(&(*(d_glb_weight[i]))[j], sizeof(val_t)*h_model->num_units[j]*h_model->num_units[j+1]);
            if(i==gpu_idx || i==5){
                copy_weight<<<para->num_blocks, para->num_threads>>>(
                    (*(d_glb_weight[i]))[j], (*d_weight)[j],
                    h_model->num_units[j]*h_model->num_units[j+1],
                    para->num_blocks*para->num_threads);
                cudaDeviceSynchronize();
                std::cerr<<"init." <<std::endl;
                myprint_kernel<<<1,1>>>((*(d_glb_weight[i]))[j],1);
                cudaDeviceSynchronize();
                myprint_kernel<<<1,1>>>((*d_weight)[j],1);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaMalloc(&gpu_weight, sizeof(val_t)*1*5);
    cudaMalloc(&cpu_weight, sizeof(val_t)*1);


    ofDebug<<"inital stepsize:"<<para->stepsize<<endl;
    std::cerr<<"gpu_stepsize: "<<para->stepsize<<std::endl;
    std::cerr<<"gpu_batches: "<<batches<< std::endl;
    std::cerr<<"gpu_batch_size (per loading, single batch): "<<para->batch_size<< std::endl;
    std::cerr<<"gpu_blocks_threads: ("<<para->num_blocks <<", "
            <<para->num_threads << ")"<< std::endl;
}

MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDImp, RunDataLoad, GPUGD_RunDataLoadMessage) {

    // msg: gpu_idx, data, start_idx, batch_size, task_idx
    cudaSetDevice(evProc.gpu_idx);
    evProc.ofDebug  << "Task " << msg.task_idx
        << ": load " << msg.batch_size
        << " tuples starting from " << msg.start_idx << endl;
    
	evProc.proc_data = msg.data;
    evProc.l2_flag = msg.flag;

    Timer load_timer;   load_timer.Restart();
    idx_t local_batchIdx = 0;

    idx_t d_startIdx = evProc.proc_data->h_dataRowPtr[msg.start_idx];
    idx_t d_endIdx = evProc.proc_data->h_dataRowPtr[msg.start_idx + msg.batch_size];
    idx_t d_local_nnz = d_endIdx - d_startIdx;

    cudaMemcpy(evProc.d_layers[local_batchIdx][0], &evProc.proc_data->h_data[d_startIdx],
        sizeof(val_t)*d_local_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(evProc.d_input_rowPtr[local_batchIdx], &evProc.proc_data->h_dataRowPtr[msg.start_idx],
        sizeof(idx_t)*(msg.batch_size+1), cudaMemcpyHostToDevice);
    evProc.d_input_rowPtr_b[local_batchIdx] = evProc.proc_data->h_dataRowPtr[msg.start_idx];
    
    idx_t t_nnzIdx = 0;
    for(idx_t i = 0; i < msg.batch_size; ++i){
        idx_t l_nnz = evProc.proc_data->h_dataRowPtr[msg.start_idx+i+1]
            - evProc.proc_data->h_dataRowPtr[msg.start_idx+i];
        for(idx_t j = 0; j < l_nnz; ++j){
            evProc.h_dataX[local_batchIdx][t_nnzIdx] = i;
            evProc.h_dataY[local_batchIdx][t_nnzIdx] = evProc.proc_data->h_dataColIdx[d_startIdx+t_nnzIdx];
            ++t_nnzIdx;
        }
    }
    cudaMemcpy(evProc.d_dataX[local_batchIdx], evProc.h_dataX[local_batchIdx],
        sizeof(idx_t)*d_local_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(evProc.d_dataY[local_batchIdx], evProc.h_dataY[local_batchIdx],
        sizeof(idx_t)*d_local_nnz, cudaMemcpyHostToDevice);

    idx_t l_startIdx = evProc.proc_data->h_labelRowPtr[msg.start_idx];
    idx_t l_endIdx = evProc.proc_data->h_labelRowPtr[msg.start_idx + msg.batch_size];
    idx_t l_local_nnz = l_endIdx - l_startIdx;

    memcpy(evProc.h_label_colIdx[local_batchIdx], &evProc.proc_data->h_labelColIdx[l_startIdx], sizeof(idx_t)*l_local_nnz);
    memcpy(evProc.h_label_rowPtr[local_batchIdx], &evProc.proc_data->h_labelRowPtr[msg.start_idx], sizeof(idx_t)*(msg.batch_size+1));

    idx_t rowPtr_b = evProc.h_label_rowPtr[local_batchIdx][0];
    for(idx_t i = 0; i < msg.batch_size+1; ++i){
        evProc.h_label_rowPtr[local_batchIdx][i] -= rowPtr_b;
    }

    cudaMemcpy(evProc.d_label[local_batchIdx], &evProc.proc_data->h_label[l_startIdx],
        sizeof(val_t)*l_local_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(evProc.d_label_colIdx[local_batchIdx], &evProc.proc_data->h_labelColIdx[l_startIdx],
        sizeof(idx_t)*l_local_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(evProc.d_label_rowPtr[local_batchIdx], &evProc.h_label_rowPtr[local_batchIdx][0],
        sizeof(idx_t)*(msg.batch_size+1), cudaMemcpyHostToDevice);

    val_t load_time = load_timer.GetTime();
    evProc.ofDebug << "RunDataLoad time = " << load_time << endl;
    evProc.ofDebug.flush();

    if(msg.task_idx == 0 || msg.task_idx == 1)
        GPUGD_AccuracyMessage_Factory(evProc.scheduler, evProc.gpu_idx, msg.start_idx, msg.task_idx, load_time);
    else if(msg.task_idx == 2)  GPUGD_TrainMessage_Factory(evProc.scheduler, evProc.gpu_idx, msg.start_idx, msg.task_idx, load_time);

}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDImp, RunAccuracy, GPUGD_RunAccuracyMessage) {

    // msg: gpu_idx, start_idx, msg.batch_size, task_idx
    cudaSetDevice(evProc.gpu_idx);
    evProc.ofDebug << "Compute accuracy on gpu_idx:" << evProc.gpu_idx << endl;

    idx_t local_tuples = msg.batch_size;

    cublasHandle_t d_handle;
    cublasCreate(&d_handle);
    cusparseHandle_t s_handle;
    cusparseCreate(&s_handle);

    idx_t local_batchIdx = 0;
    idx_t data_nnz = evProc.proc_data->h_dataRowPtr[msg.start_idx + local_tuples]
        - evProc.proc_data->h_dataRowPtr[msg.start_idx];
 
    Timer accu_timer;   accu_timer.Restart();
    evProc.accu = local_tuples*get_accuracy_cuda_coo_stable(evProc.gpu_streams, s_handle, d_handle,
        evProc.h_label_colIdx[local_batchIdx], evProc.h_label_rowPtr[local_batchIdx],
        evProc.d_layers[local_batchIdx], evProc.d_dataX[local_batchIdx], evProc.d_dataY[local_batchIdx],
        *(evProc.d_weight),
        evProc.h_model->num_units, local_tuples,
        evProc.d_rowsum[local_batchIdx], evProc.d_maxAccuIdx[local_batchIdx], evProc.h_maxAccuIdx[local_batchIdx],
        evProc.alpha, evProc.beta, evProc.gpu_idx, data_nnz,
        evProc.para->num_blocks, 1024);
    val_t accu_time = accu_timer.GetTime();
    idx_t t_flag = evProc.l2_flag;
    val_t b_loss = local_tuples*get_loss_cuda_coo_stable(d_handle, evProc.d_layers[local_batchIdx],
        evProc.d_label[local_batchIdx], evProc.d_label_colIdx[local_batchIdx],
        evProc.d_label_rowPtr[local_batchIdx], evProc.t_loss[local_batchIdx],
        *(evProc.d_weight), evProc.l2_sum, evProc.h_l2_sum, evProc.l2_flag,
        evProc.h_model->num_units, local_tuples, evProc.gpu_idx, evProc.para->num_blocks, 1024);
    if(t_flag==-1 && evProc.l2_flag==1) evProc.l2_flag = -1;
    
    evProc.ofDebug  << "RunAccuracy time: " << accu_time
        << " accu = " << std::setprecision(6) << evProc.accu
        << " loss = " << std::setprecision(6) << b_loss << endl;
    evProc.ofDebug.flush();

    ++evProc.para->accu_batches;
    cublasDestroy(d_handle);
    cusparseDestroy(s_handle);
    std::cerr<<"l2_flag(GPUGD):" << evProc.l2_flag << std::endl;
    GPUGD_DataLoadMessage_Factory(evProc.scheduler, evProc.gpu_idx, msg.start_idx, msg.task_idx, evProc.accu, b_loss, evProc.l2_flag, accu_time);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDImp, RunTrain, GPUGD_RunTrainMessage) {

    // msg: gpu_idx, start_idx, batch_size, task_idx, stepsize, model
    cudaSetDevice(evProc.gpu_idx);

    evProc.ofDebug << "Train" << endl;
    
    idx_t local_tuples = msg.batch_size;

    cublasHandle_t d_handle;
    cublasCreate(&d_handle);
    cusparseHandle_t s_handle;
    cusparseCreate(&s_handle);

    idx_t local_batchIdx = 0;
    idx_t data_nnz = evProc.proc_data->h_dataRowPtr[msg.start_idx + local_tuples]
        - evProc.proc_data->h_dataRowPtr[msg.start_idx];
    idx_t label_nnz = evProc.proc_data->h_labelRowPtr[msg.start_idx + local_tuples]
        - evProc.proc_data->h_labelRowPtr[msg.start_idx];

    Timer train_timer;  train_timer.Restart();
    forward_cuda_coo(evProc.gpu_streams, s_handle, d_handle,
        evProc.d_layers[local_batchIdx],
        evProc.d_dataX[local_batchIdx], evProc.d_dataY[local_batchIdx],
        *(evProc.d_weight),
        evProc.h_model->num_units, local_tuples,
        evProc.d_rowsum[local_batchIdx],
        evProc.d_maxIdx[local_batchIdx],
        evProc.alpha, evProc.beta, evProc.gpu_idx, data_nnz,
        evProc.para->num_blocks, 1024);
    backprop_cuda_coo(s_handle, d_handle,
        evProc.d_label[local_batchIdx], evProc.d_label_colIdx[local_batchIdx], evProc.d_label_rowPtr[local_batchIdx],
        evProc.d_layers[local_batchIdx], evProc.d_dataX[local_batchIdx], evProc.d_dataY[local_batchIdx],
        *(evProc.d_weight),
        evProc.h_model->num_units, local_tuples,
        evProc.d_gradient[local_batchIdx], 
        evProc.d_dlossy[local_batchIdx], evProc.d_dlossx[local_batchIdx],
        evProc.alpha, evProc.beta, evProc.gpu_idx, data_nnz, label_nnz,
        evProc.para->num_blocks, 1024);
    
    update_model_cuda(*(evProc.d_weight), evProc.d_gradient[local_batchIdx], evProc.h_model->num_units,
        msg.stepsize, evProc.para->num_blocks, 1024);
    
    val_t train_time = train_timer.GetTime();
    evProc.ofDebug << "RunTrain time = " << train_time << endl;
    evProc.ofDebug.flush();

    cublasDestroy(d_handle);
    cusparseDestroy(s_handle);

    GPUGD_DataLoadMessage_Factory(evProc.scheduler, evProc.gpu_idx, msg.start_idx, msg.task_idx, 0.f, 0.f, evProc.l2_flag, train_time);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDImp, newDieHandler, DieMessage)
        return true;
}
