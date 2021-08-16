#include "GPUGDScheduler.h"
#include <fstream>
#include <ctime>

extern char* LOG_DIRECTORY;
extern idx_t SERVER_RANK;

GPUGDSchedulerImp::GPUGDSchedulerImp() : sparse_data(NULL), sparse_testdata(NULL), model(NULL), para(NULL), gpuEV(NULL){

    //idx_t n_gpus = 4;
    idx_t n_gpus = 2;
    gpuEV.resize(n_gpus);
    for(idx_t i = 0; i < n_gpus; ++i){
        gpuEV[i] = new GPUGD(myInterface);
    }

    for(idx_t i = 0; i < n_gpus; ++i){
        gpuEV[i]->ForkAndSpin();
    }

    RegisterMessageProcessor(GPUGD_AccuracyMessage::type, &GPUGD_Accuracy, 100);
    RegisterMessageProcessor(GPUGD_TrainMessage::type, &GPUGD_Train, 100);
    RegisterMessageProcessor(GPUGD_DataLoadMessage::type, &GPUGD_DataLoad, 100);
    RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


GPUGDSchedulerImp::~GPUGDSchedulerImp() {
        ofDebug << "Release memory" << endl;
        ofDebug.close();

    cublasDestroy(d_handle);

    for(idx_t i = 0; i < n_gpus; ++i){
        delete gpuEV[i];
        for(idx_t j = 0; j < model->num_layers-1; ++j)
            cudaFree((*gpu_d_weight[i])[j]);
    }
}


void GPUGDSchedulerImp::RunGPUGD(SparseData* _sparse_data, SparseData* _sparse_testdata, FCModel* _model, Para* _para, idx_t _n_ranks, idx_t _rank, idx_t _n_gpus) {
    rank = _rank;
    n_ranks = _n_ranks;
    n_gpus = _n_gpus;

    ofDebug.open((string(LOG_DIRECTORY) + "GPUGDScheduler.log.rank").c_str() + std::to_string(rank), ios::out);

    sparse_data = _sparse_data;
    sparse_testdata = _sparse_testdata;
    model = _model;
    para = _para;
    gpu_model.resize(n_gpus);
    gpu_d_weight.resize(n_gpus);
    for(idx_t i = 0; i < n_gpus; ++i){
        cudaSetDevice(i);
        gpu_model[i] = gpu_model[i].deep_copy(*model);

        gpu_d_weight[i] = new vector<val_t*>();
        gpu_d_weight[i]->resize(model->num_layers-1);
        for(idx_t j = 0; j < model->num_layers-1; ++j){
            cudaMalloc(&(*gpu_d_weight[i])[j], sizeof(val_t)*model->num_units[j]*model->num_units[j+1]);
            cudaMemcpy((*gpu_d_weight[i])[j], model->weight[j], sizeof(val_t)
                *model->num_units[j]*model->num_units[j+1], cudaMemcpyHostToDevice);
        }
    }
    
	// Enable Peer-to-Peer
    for(idx_t i = 0; i < n_gpus; ++i){
        cudaSetDevice(i);
        for(idx_t k = 0; k < n_gpus; ++k){
            if(k==i)    continue;
            cudaDeviceEnablePeerAccess(k,0);
        }
    }
    cublasCreate(&d_handle);

    ofDebug << "Start GPUGD" << endl;
    ofDebug << "num_blocks: " << para->num_blocks
            << "num_threads: " << para->num_threads
            << ", num_tuples: " << sparse_data->num_tuples
            << ", num_testtuples: " << sparse_testdata->num_tuples
            << ", feature_size: " << sparse_data->feature_size
            << ", num_classes: " << sparse_data->num_classes
            << ", batch_size: " << para->batch_size
            << ", num_batches: " << para->num_batches
            << ", num_testbatches: " << para->num_testbatches
            << ", num_microbatches: 1" //<< para->num_mbatches
            << ", tuples_in_last_batch: " << para->tuples_last_batch
            << ", init_stepsize: " << para->init_stepsize
            << ", decay: " << para->decay<< endl;

    for(idx_t i = 1; i < model->num_layers-1; i++)
        ofDebug << "hidden layer " << i << ": " << model->num_units[i] << endl;
    ofDebug.flush();

    // init gpuEV.proc_data as sparse_data
    for(idx_t i = 0; i < n_gpus; ++i)
        gpuEV[i]->Init(sparse_data, sparse_testdata->data_max_row_nnz,
            sparse_testdata->label_max_row_nnz, &gpu_model[i], gpu_d_weight[i], para, i, rank);

    task_idx = 0;
    start_idx = 0;

    for(idx_t i = 0; i < n_gpus; ++i){
        accu_send.push_back(0);
        model_send.push_back(0);
        train_batches.push_back(0);
        gpu_p.push_back(0.);
        batch_size.push_back(para->batch_size);
    }
    
	max_gpu_bsize = para->batch_size;
    s_merg_iter = 0, merg_iter = 0;
    train_tuples = 0, train_idx = 0;
    test_batches = 0;

    sche_data_timer.resize(n_gpus);
    sche_testaccu_timer.resize(n_gpus);
    sche_train_timer.resize(n_gpus);
    for(idx_t i = 0; i < n_gpus; ++i){
        sche_data_time.push_back(0.); pure_data_time.push_back(0.);
        sche_testaccu_time.push_back(0.); pure_testaccu_time.push_back(0.);
        sche_train_time.push_back(0.); pure_train_time.push_back(0.);
    }
    task_testaccu_time = 0.; task_train_time = 0.; task_sampleaccu_time = 0.;
    for(idx_t i = 0; i < 3; ++i)    mpi_time.push_back(0.);
    transfer_time = 0.; localavg_time = 0.;

    test_accuracy = 0.; //sample_accuracy = 0.;
    test_loss = 0.; //sample_loss = 0.;
    p_test_loss = 0.;
    incloss_cnt = 0;
    train_iter = 0;
    sample_tuples = 0;

    task_idx = 1;
    l2_flag = 1;
    task_testaccu_timer.Restart();

    for(idx_t i = 0; i < n_gpus; ++i){
            GPUGD_RunDataLoadMessage_Factory(*gpuEV[i], i, sparse_testdata, start_idx, para->batch_size, l2_flag, task_idx);
            start_idx += para->batch_size;
            sche_data_timer[i].Restart();
            ++accu_send[i];
    }
}

MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDSchedulerImp, GPUGD_Accuracy, GPUGD_AccuracyMessage) {
    evProc.sche_data_time[msg.gpu_idx] += evProc.sche_data_timer[msg.gpu_idx].GetTime();
    evProc.pure_data_time[msg.gpu_idx] += msg.time;
    GPUGD_RunAccuracyMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.gpu_idx, msg.start_idx, evProc.para->batch_size, evProc.task_idx);
    ++evProc.test_batches;
    evProc.sche_testaccu_timer[msg.gpu_idx].Restart();
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDSchedulerImp, GPUGD_Train, GPUGD_TrainMessage) {
   if(evProc.task_idx == 2 || evProc.task_idx == 3){
        evProc.sche_data_time[msg.gpu_idx] += evProc.sche_data_timer[msg.gpu_idx].GetTime();
        evProc.pure_data_time[msg.gpu_idx] += msg.time;
        ++evProc.model_send[msg.gpu_idx];

        // warmup+adaptive+time-based decay
        val_t t_decay = 0.f;
        if(evProc.train_tuples/evProc.sparse_data->num_tuples <= 10)
            t_decay = 1.f;
        else
            t_decay = 1./(1+evProc.para->decay*(
                evProc.train_tuples/evProc.sparse_data->num_tuples-10));
        val_t lr = evProc.para->init_stepsize*t_decay
                /evProc.para->batch_size*evProc.batch_size[msg.gpu_idx];
        if(evProc.train_tuples/evProc.sparse_data->num_tuples <= 10){
            // gradual warmup
            lr = evProc.para->init_stepsize/1024*32/100 +
                ((evProc.para->init_stepsize/1024*32 -
                evProc.para->init_stepsize/1024*32/100)*evProc.train_tuples
                /(10*evProc.sparse_data->num_tuples));
        }/**/
        GPUGD_RunTrainMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.gpu_idx, msg.start_idx, evProc.batch_size[msg.gpu_idx], evProc.task_idx, lr, evProc.model);
        ++evProc.model_send[msg.gpu_idx];
        evProc.sche_train_timer[msg.gpu_idx].Restart();
    }
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDSchedulerImp, GPUGD_DataLoad, GPUGD_DataLoadMessage) {
    // msg: gpu_idx, start_idx, (task_idx,) val, loss, time

    evProc.l2_flag = msg.flag;
    if(evProc.task_idx == 1){
        evProc.test_accuracy += msg.val;
        evProc.test_loss += msg.loss;
        evProc.sche_testaccu_time[msg.gpu_idx] += evProc.sche_testaccu_timer[msg.gpu_idx].GetTime();
        evProc.pure_testaccu_time[msg.gpu_idx] += msg.time;
        --evProc.accu_send[msg.gpu_idx];
        bool isDone = true;
        for(idx_t i = 0; i < evProc.n_gpus; ++i){
            if(evProc.accu_send[i] > 0)    isDone = false;
        }

        if(evProc.start_idx + evProc.para->batch_size >=
            evProc.sparse_testdata->num_tuples && isDone){

            std::cerr<<"start_idx: "<<evProc.start_idx << std::endl;
            evProc.test_accuracy /= (evProc.start_idx+1.);
            evProc.test_loss /= (evProc.start_idx+1.);

            evProc.test_batches = 0;
            std::cerr<<"testing_accu: " << evProc.test_accuracy
                     << ", testing_loss: " << evProc.test_loss
                     << ", train_tuples: " << evProc.train_tuples << std::endl;
            evProc.task_testaccu_time += evProc.task_testaccu_timer.GetTime();
            evProc.ofDebug  << "testing_accuracy: " << std::setprecision(6) << evProc.test_accuracy
                            << " testing_loss: " << std::setprecision(6) << evProc.test_loss
                            << " train_tuples: " << evProc.train_tuples
                            << " task_testaccu_time: " << evProc.task_testaccu_time;
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                evProc.ofDebug  << " pure_data_time(GPU" << i
                                << "): " << evProc.pure_data_time[i]
                                << " pure_testaccu_time(GPU" << i
                                << "): " << evProc.pure_testaccu_time[i];
            }
            evProc.ofDebug << endl;
            evProc.ofDebug.flush();

            if (evProc.train_tuples/evProc.sparse_data->num_tuples > evProc.para->train_iter){
                evProc.ofDebug  << "Total task_testaccu_time: " << evProc.task_testaccu_time
                                << " total task_train_time: " << evProc.task_train_time
                                << " total task_sampleaccu_time: " << evProc.task_sampleaccu_time
                                << " total iterations: " << evProc.para->train_iter;
                evProc.ofDebug.flush();
                DieMessage_Factory(evProc.myInterface);
            }

            evProc.test_accuracy = 0.;
            evProc.p_test_loss = evProc.test_loss;
            evProc.test_loss = 0.;
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                evProc.accu_send[i] = 0;
            }

            evProc.task_idx = 2;
            evProc.start_idx = evProc.train_idx;
            evProc.task_train_timer.Restart();

            if(evProc.train_tuples/evProc.sparse_data->num_tuples == 11
                && evProc.s_merg_iter == 0){
                evProc.max_gpu_bsize = 1024;
				for(idx_t i = 0; i<evProc.n_gpus; ++i)	evProc.batch_size[i] = 1024;
                std::cerr<<"start adaptive bsize." <<std::endl;
            }

            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                if(evProc.start_idx + evProc.batch_size[i] >= evProc.sparse_data->num_tuples){
                    evProc.start_idx = 0;
                }
                GPUGD_RunDataLoadMessage_Factory(*evProc.gpuEV[i], i, evProc.sparse_data, evProc.start_idx, evProc.batch_size[i], evProc.l2_flag, evProc.task_idx);
                evProc.start_idx += evProc.batch_size[i];
                evProc.train_tuples += evProc.batch_size[i];
                ++evProc.train_batches[i];
                evProc.sche_data_timer[i].Restart();
                ++evProc.model_send[i];
            }
        } else if(evProc.start_idx + evProc.para->batch_size < evProc.sparse_testdata->num_tuples){
            GPUGD_RunDataLoadMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.gpu_idx, evProc.sparse_testdata, evProc.start_idx, evProc.para->batch_size, evProc.l2_flag, evProc.task_idx);
            evProc.start_idx += evProc.para->batch_size;
            evProc.sche_data_timer[msg.gpu_idx].Restart();
            ++evProc.accu_send[msg.gpu_idx];
        }

    } else if(evProc.task_idx == 2){
        evProc.sche_train_time[msg.gpu_idx] += evProc.sche_train_timer[msg.gpu_idx].GetTime();
        evProc.pure_train_time[msg.gpu_idx] += msg.time;

        evProc.sample_tuples += evProc.batch_size[msg.gpu_idx];

        evProc.model_send[msg.gpu_idx]-=3;
        bool isDone = true;
        for(idx_t i = 0; i < evProc.n_gpus; ++i){
            if(evProc.model_send[i] > 0)    isDone = false;
        }
        if(evProc.sample_tuples-evProc.max_gpu_bsize>=evProc.para->sample_tuples && isDone){
            evProc.task_train_time += evProc.task_train_timer.GetTime();
            evProc.ofDebug  << "iteration: " << evProc.train_tuples/evProc.sparse_data->num_tuples
                            << " merg_iter: " << evProc.merg_iter
                            << " task_train_time: " << evProc.task_train_time;
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                evProc.ofDebug  << " pure_data_time(GPU" << i
                                << "): " << evProc.pure_data_time[i]
                                << " pure_train_time(GPU" << i
                                << "): " << evProc.pure_train_time[i];
            }
            evProc.ofDebug << endl;
            evProc.ofDebug.flush();

            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                evProc.model_send[i] = 0;
            }

            val_t t_decay = 1./(1+evProc.para->decay*(
                evProc.train_tuples/evProc.sparse_data->num_tuples-10));
            std::cerr<<"decay: "<<t_decay<<std::endl;
            evProc.localavg_timer.Restart();
            /////////////////average_gpu_peer_b/////////////////
            std::cerr<<"gpu_idx:"<<msg.gpu_idx<<std::endl;
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                std::cerr<<"gpu"<<i<<"_";
                myprint_kernel<<<1,1>>>((*evProc.gpu_d_weight[i])[0],1);
                cudaDeviceSynchronize();
            }

            idx_t tot_batches = 0;
            for(idx_t i = 0; i < evProc.n_gpus; ++i)  tot_batches+=evProc.train_batches[i];
            for(idx_t i = 0; i < evProc.n_gpus; ++i)
                evProc.gpu_p[i] = 1.0f*evProc.train_batches[i]/tot_batches;
            std::cerr<<"GPU_Train: ";
            for(idx_t i = 0; i < evProc.n_gpus; ++i)    std::cerr<<","<<evProc.gpu_p[i];
            std::cerr<<std::endl;
            
            for(idx_t i = 0; i < evProc.n_gpus; ++i)    std::cerr<<","<<evProc.train_batches[i];
            std::cerr<<std::endl;

            // adaptive batch size
            idx_t t_iter = evProc.train_tuples/evProc.sparse_data->num_tuples;
        
            idx_t bsize_min = 32, bsize_max = 1024;
            val_t mu = 1.f/evProc.n_gpus;
            val_t p_tuples = 0.f;
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                p_tuples += evProc.gpu_p[i]*evProc.batch_size[i];
            }
            idx_t tmp_b = 4;

			if(evProc.train_tuples/evProc.sparse_data->num_tuples > 10){

				for(idx_t i = 0; i < evProc.n_gpus; ++i){
					if(evProc.gpu_p[i]<mu && evProc.batch_size[i]>bsize_min)
						//evProc.batch_size[i]/=tmp_b;
						evProc.batch_size[i]-=32;
					else if(evProc.gpu_p[i]>mu && evProc.batch_size[i]<bsize_max)
						//evProc.batch_size[i]*=tmp_b;
						evProc.batch_size[i]+=32;   
				}/**/

				val_t max_p = 0., min_p = 1.;
				idx_t max_gidx = -1, min_gidx = -1;
				for(idx_t i = 0; i < evProc.n_gpus; ++i){
					if(max_p < evProc.gpu_p[i]){
						max_p = evProc.gpu_p[i];
						max_gidx = i;
					}
					if(min_p > evProc.gpu_p[i]){
						min_p = evProc.gpu_p[i];
						min_gidx = i;
					}
				}
				max_p/=10, min_p/=10;
				
				std::cerr<<"l2_flag: " << evProc.l2_flag << std::endl;
				if(evProc.l2_flag == 1){
					evProc.gpu_p[max_gidx] += max_p;
					evProc.gpu_p[min_gidx] -= min_p;
				}/**/
			
				std::cerr<<"max_weight "<< evProc.gpu_p[max_gidx]<<" on GPU" << max_gidx<<std::endl;
				std::cerr<<"min_weight "<< evProc.gpu_p[min_gidx]<<" on GPU" << min_gidx<<std::endl;

			} /**/

			if(evProc.train_tuples/evProc.sparse_data->num_tuples <= 10){
				for(idx_t i = 0; i < evProc.n_gpus; ++i)
					evProc.batch_size[i] = 32;
			}/**/

            evProc.max_gpu_bsize = evProc.batch_size[0];
            for(idx_t i = 1; i < evProc.n_gpus; ++i){
                evProc.max_gpu_bsize = evProc.batch_size[i] > evProc.max_gpu_bsize?
                    evProc.batch_size[i] : evProc.max_gpu_bsize;
            }

            for(idx_t i = 0; i < evProc.n_gpus; ++i)    std::cerr<<","<<evProc.batch_size[i];
            std::cerr<<", max_gpu_bsize: "<< evProc.max_gpu_bsize << std::endl;

            evProc.localavg_timer.Restart();
            #pragma omp parallel num_threads(evProc.n_gpus)
            {
                idx_t i = omp_get_thread_num();
                cudaSetDevice(i);
                cudaMemcpy(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->gpu_weight+i, &evProc.gpu_p[i],
                    sizeof(val_t), cudaMemcpyHostToDevice);

                for(idx_t j = 0; j < evProc.model->num_layers-1; ++j){
                    mul_glb_weight<<<evProc.para->num_blocks,
                        evProc.para->num_threads>>>(
                        (*evProc.gpu_d_weight[i])[j],
                        ((GPUGDImp*)(evProc.gpuEV[i]->evProc))->gpu_weight+i,//evProc.gpu_weight,
                        evProc.model->num_units[j]*evProc.model->num_units[j+1],
                        evProc.para->num_blocks*evProc.para->num_threads);
                    cudaDeviceSynchronize();
                    #pragma omp barrier
                    /////////// b_2gpus ///////////
                    idx_t n_elements = evProc.model->num_units[j]*evProc.model->num_units[j+1];
                    idx_t chunks = 8; 
                    idx_t stream_size = n_elements/chunks;
                    idx_t m_streamsize = stream_size/2;
                    val_t t_gweight = (1.);
                    for(idx_t ts = 0; ts < chunks; ++ts){
                        idx_t dest_gpu = i, src_gpu= (i+1)%2;
                        idx_t data_idx= ts*stream_size+i*m_streamsize;
                        cudaMemcpyPeerAsync(
                            (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))
                            ->d_glb_weight[src_gpu]))[j]+data_idx, i,
                            (*evProc.gpu_d_weight[src_gpu])[j]+data_idx, src_gpu,
                            sizeof(val_t)*m_streamsize, ((GPUGDImp*)(evProc.gpuEV[i]->evProc))
                            ->gpu_streams[ts%4]);
                        cublasSetStream(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->d_handle,
                            ((GPUGDImp*)(evProc.gpuEV[i]->evProc))
                            ->gpu_streams[ts%4]);
                        cublasSaxpy(((GPUGDImp*)(evProc.gpuEV[i]->evProc))
                            ->d_handle, m_streamsize, &t_gweight,
                            (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))
                            ->d_glb_weight[src_gpu]))[j]+data_idx, 1,
                            (*evProc.gpu_d_weight[i])[j]+data_idx, 1);
                        cudaMemcpyPeerAsync(
                            (*evProc.gpu_d_weight[src_gpu])[j]+data_idx, src_gpu,
                            (*evProc.gpu_d_weight[i])[j]+data_idx, i,
                            sizeof(val_t)*m_streamsize, ((GPUGDImp*)(evProc.gpuEV[i]->evProc))
                            ->gpu_streams[ts%4]);
                    }
                    cudaDeviceSynchronize();
                    /////////// e_2gpus ///////////

                    /////////// GD momentum 
                    mu_weight<<<evProc.para->num_blocks,evProc.para->num_threads>>>(
                        (*evProc.gpu_d_weight[i])[j],
                        (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->d_glb_weight[i]))[j],
                        (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->d_glb_weight[evProc.n_gpus]))[j],
                        evProc.model->num_units[j]*evProc.model->num_units[j+1],
                        evProc.para->num_blocks*evProc.para->num_threads);
                    cudaDeviceSynchronize();
                    copy_weight<<<evProc.para->num_blocks,evProc.para->num_threads>>>(
                        (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->d_glb_weight[evProc.n_gpus]))[j],
                        (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->d_glb_weight[i]))[j],
                        evProc.model->num_units[j]*evProc.model->num_units[j+1],
                        evProc.para->num_blocks*evProc.para->num_threads);
                    cudaDeviceSynchronize();
                    copy_weight<<<evProc.para->num_blocks,evProc.para->num_threads>>>(
                        (*(((GPUGDImp*)(evProc.gpuEV[i]->evProc))->d_glb_weight[i]))[j],
                        (*evProc.gpu_d_weight[i])[j],
                        evProc.model->num_units[j]*evProc.model->num_units[j+1],
                        evProc.para->num_blocks*evProc.para->num_threads);
                    cudaDeviceSynchronize();
                    /////////// e_momentum

                } // end of #layers
                //cudaDeviceSynchronize();
            }
            evProc.localavg_time += evProc.localavg_timer.GetTime();
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                std::cerr<<"gpu"<<i<<"_";
                myprint_kernel<<<1,1>>>((*evProc.gpu_d_weight[i])[0],1);
                cudaDeviceSynchronize();
            }
            /////////////////average_gpu_peer_e/////////////////

            evProc.ofDebug << " localavg_time: " << evProc.localavg_time;
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                evProc.ofDebug  << " pure_data_time(GPU" << i
                                << "): " << evProc.pure_data_time[i]
                                << " pure_train_time(GPU" << i
                                << "): " << evProc.pure_train_time[i];
            }
            for(idx_t i = 0; i < evProc.n_gpus; ++i){
                evProc.ofDebug  << " new_bsize(GPU" << i
                                << "): " <<evProc.batch_size[i];
            }
            evProc.ofDebug << endl;

			if(evProc.train_tuples/evProc.sparse_data->num_tuples == 10){
				std::cerr<<"save model"<<std::endl;
				std::ofstream oFile("amazon-sgdm-warmup");
				for(idx_t k = 0; k < evProc.model->num_layers-1; ++k){
					cudaMemcpy(evProc.model->weight[k],
						(*evProc.gpu_d_weight[msg.gpu_idx])[k],
						sizeof(val_t)*evProc.model->num_units[k]*evProc.model->num_units[k+1],
						cudaMemcpyDeviceToHost);
					for(idx_t j = 0; j < evProc.model->num_units[k]*evProc.model->num_units[k+1]; ++j){
						oFile <<std::scientific << evProc.model->weight[k][j] << ",";
					}
					oFile << std::endl;
				}
				oFile.close();
			}/**/
            ////////////////////////////////
            
			evProc.sample_tuples = 0;
            for(idx_t i = 0; i < evProc.n_gpus; ++i)    evProc.train_batches[i] = 0;

            ++evProc.s_merg_iter;
            std::cerr<<"s_merg_iter: "<< evProc.s_merg_iter;
            if(evProc.s_merg_iter==evProc.para->sampletest_tuples){
                evProc.s_merg_iter = 0;
                ++evProc.merg_iter;
            }
            std::cerr<<", merg_iter: " << evProc.merg_iter << std::endl;


            if(evProc.s_merg_iter == 0){
                // compute testing accuracy
                evProc.train_idx = evProc.start_idx;
                evProc.start_idx = 0;
                evProc.task_idx = 1;
                evProc.l2_flag = 1;
                evProc.task_testaccu_timer.Restart();

                for(idx_t i = 0; i < evProc.n_gpus; ++i){
                    GPUGD_RunDataLoadMessage_Factory(*evProc.gpuEV[i], i, evProc.sparse_testdata, evProc.start_idx, evProc.para->batch_size, evProc.l2_flag, evProc.task_idx);
                    evProc.start_idx += evProc.para->batch_size;
                    evProc.sche_data_timer[i].Restart();
                    ++evProc.accu_send[i];
                }
            } else {
                evProc.task_train_timer.Restart();
                for(idx_t i = 0; i < evProc.n_gpus; ++i){
                    if(evProc.start_idx + evProc.batch_size[i] >= evProc.sparse_data->num_tuples){
                        evProc.start_idx = 0;
                    }
                    GPUGD_RunDataLoadMessage_Factory(*evProc.gpuEV[i], i, evProc.sparse_data, evProc.start_idx, evProc.batch_size[i], evProc.l2_flag, evProc.task_idx);
                    evProc.start_idx += evProc.batch_size[i];
                    evProc.train_tuples += evProc.batch_size[i];
                    ++evProc.train_batches[i];
                    evProc.sche_data_timer[i].Restart();
                    ++evProc.model_send[i];
                }
            }
        } else if(evProc.sample_tuples - evProc.max_gpu_bsize < evProc.para->sample_tuples){
            if(evProc.start_idx + evProc.batch_size[msg.gpu_idx] >= evProc.sparse_data->num_tuples){
                evProc.start_idx = 0;
            }
            GPUGD_RunDataLoadMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.gpu_idx, evProc.sparse_data, evProc.start_idx, evProc.batch_size[msg.gpu_idx], evProc.l2_flag, evProc.task_idx);
            evProc.start_idx += evProc.batch_size[msg.gpu_idx];
            evProc.train_tuples += evProc.batch_size[msg.gpu_idx];
            ++evProc.train_batches[msg.gpu_idx];
            evProc.sche_data_timer[msg.gpu_idx].Restart();
            ++evProc.model_send[msg.gpu_idx];
        }
    } else if (evProc.task_idx == 3){
        evProc.model_send[msg.gpu_idx]-=3;
    }
}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUGDSchedulerImp, newDieHandler, DieMessage)
        for(idx_t i = 0; i < evProc.n_gpus; ++i){
                DieMessage_Factory(*evProc.gpuEV[i]);
                evProc.gpuEV[i]->Join();
        }
        return true;
}
