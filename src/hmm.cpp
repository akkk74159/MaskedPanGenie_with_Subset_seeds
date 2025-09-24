#include <utility>
#include <math.h>
#include <cassert>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "hmm.hpp"
#include "emissionprobabilitycomputer.hpp"
#include <iostream>
#ifdef USE_CUDA
#include "hmm_gpu.h"   // wrapper 宣告
#endif
using namespace std;

#ifdef USE_CUDA
#pragma message("### USE_CUDA enabled in hmm.cpp ###")
#endif

void print_column(vector<long double>* column, ColumnIndexer* indexer) {
	for (size_t i = 0; i < column->size(); ++i) {
		pair<size_t,size_t> paths = indexer->get_path_ids_at(i);
		cout << setprecision(15) << column->at(i) << " paths: " << paths.first << " " <<  paths.second << endl;
	}
	cout << "" << endl;
}


HMM::HMM(vector<UniqueKmers*>* unique_kmers, ProbabilityTable* probabilities, bool run_genotyping, bool run_phasing, double recombrate, bool uniform, long double effective_N, vector<unsigned short>* only_paths, bool normalize)
	:unique_kmers(unique_kmers),
	 probabilities(probabilities),
	 genotyping_result(unique_kmers->size()),
	 recombrate(recombrate),
	 uniform(uniform),
	 effective_N(effective_N)
{
	// index all columns with at least one alternative allele
	index_columns(only_paths);

	size_t size = this->column_indexers.size();
	// initialize forward normalization sums
	this->forward_normalization_sums = vector<long double>(size, 0.0L);
	this->previous_backward_column = nullptr;
//	cerr << "Indexing the columns ..." << endl;

	if (run_genotyping) {
//		cerr << "Computing forward probabilities ..." << endl;
		compute_forward_prob();
//		cerr << "Computing backward probabilities ..." << endl;
		compute_backward_prob();

		if (normalize) {
			for (size_t i = 0; i < this->genotyping_result.size(); ++i) {
				genotyping_result[i].normalize();
			}
		}
	}

	if (run_phasing) {
//		cerr << "Computing Viterbi path ..." << endl;
		compute_viterbi_path();
	}
}

HMM::~HMM(){
	init(this->forward_columns,0);
	if (this->previous_backward_column != nullptr) delete this->previous_backward_column;
	init(this->viterbi_columns,0);
	init(this->viterbi_backtrace_columns,0);
	init(this->column_indexers, 0);
}

void HMM::index_columns(vector<unsigned short>* only_paths) {
	size_t column_count = this->unique_kmers->size();
	// do one forward pass to compute ColumnIndexers
	for (size_t column_index = 0; column_index < column_count; ++ column_index) {
		// get path ids of current column
		vector<unsigned short> current_paths;
		vector<unsigned char> current_alleles;
		this->unique_kmers->at(column_index)->get_path_ids(current_paths, current_alleles, only_paths);
		unsigned short nr_paths = current_paths.size();

		if (nr_paths == 0) {
			ostringstream oss;
			oss << "HMM::index_columns: column " << column_index << " is not covered by any paths.";
			throw runtime_error(oss.str());
		}

		// check whether there are any non-reference alleles in panel
		bool all_absent = true;
		for (unsigned short i = 0; i < nr_paths; ++i) {
			if ((current_alleles[i] != 0) && (!this->unique_kmers->at(column_index)->is_undefined_allele(current_alleles[i])) ) all_absent = false;
		}

		if (!all_absent) {
			// the ColumnIndexer to be filled
			ColumnIndexer* column_indexer = new ColumnIndexer(column_index);
			for (unsigned short i = 0; i < nr_paths; ++i) {
				column_indexer->insert_path(current_paths[i], current_alleles[i]);
			}
			this->column_indexers.push_back(column_indexer);
		}
	}
}

void HMM::compute_forward_prob() {
#ifdef USE_CUDA
	cerr << "Using compute_forward_prob_gpu" << endl;
    compute_forward_prob_gpu();
    return;
#else
cerr << "Using compute_forward_prob" << endl;
	size_t column_count = this->column_indexers.size();
	init(this->forward_columns, column_count);
	
	// forward pass
	size_t k = (size_t) sqrt(column_count);
	for (size_t column_index = 0; column_index < column_count; ++column_index) {;
		compute_forward_column(column_index);
		// sparse table: check whether to delete previous column
		if ( (k > 1) && (column_index > 0) && (((column_index - 1)%k != 0)) ) {
			delete this->forward_columns[column_index-1];
			this->forward_columns[column_index-1] = nullptr;
		}
	}
#endif
}

void HMM::compute_backward_prob() {
#ifdef USE_CUDA
	cerr << "Using compute_backward_prob_gpu" << endl;
    compute_backward_prob_gpu();
    return;
#else
	cerr << "Using compute_backward_prob" << endl;
	size_t column_count = this->column_indexers.size();
	if (column_count == 0) return;
	if (this->previous_backward_column != nullptr) {
		delete this->previous_backward_column;
		this->previous_backward_column = nullptr;
	}

	// backward pass
	for (int column_index = column_count-1; column_index >= 0; --column_index) {
		compute_backward_column(column_index);
	}
#endif
}

void HMM::compute_viterbi_path() {
#ifdef USE_CUDA
	cerr << "Using compute_viterbi_path_gpu" << endl;
    compute_viterbi_path_gpu();
    return;
#else
	cerr << "Using compute_viterbi_path_gpu" << endl;
	size_t column_count = this->column_indexers.size();
	if (column_count == 0) return;
	init(this->viterbi_columns, column_count);
	init(this->viterbi_backtrace_columns, column_count);

	// perform viterbi algorithm
	size_t k = (size_t) sqrt(column_count);
	for (size_t column_index = 0; column_index < column_count; ++column_index) {
		compute_viterbi_column(column_index);
		// sparse table: check whether to delete previous column
		if ((k > 1) && (column_index > 0) && (((column_index - 1)%k != 0)) ) {
			delete this->viterbi_columns[column_index-1];
			this->viterbi_columns[column_index-1] = nullptr;
			delete this->viterbi_backtrace_columns[column_index-1];
			this->viterbi_backtrace_columns[column_index-1] = nullptr;
		}
	}

	// find best value (+ index) in last column
	size_t best_index = 0;
	long double best_value = 0.0L;
	vector<long double>* last_column = this->viterbi_columns.at(column_count-1);
	assert (last_column != nullptr);
	for (size_t i = 0; i < last_column->size(); ++i) {
		long double entry = last_column->at(i);
		if (entry >= best_value) {
			best_value = entry;
			best_index = i;
		}
	}

	// backtracking
	size_t column_index = column_count - 1;
	while (true) {
		pair<unsigned short, unsigned short> path_ids = this->column_indexers.at(column_index)->get_path_ids_at(best_index);
		unsigned char allele1 = this->column_indexers.at(column_index)->get_allele (path_ids.first);
		unsigned char allele2 = this->column_indexers.at(column_index)->get_allele (path_ids.second);

		// columns might have to be re-computed
		if (this->viterbi_backtrace_columns[column_index] == nullptr) {
			size_t j = column_index / k*k;
			assert (this->viterbi_columns[j] != nullptr);
			for (j = j+1; j<=column_index; ++j) {
				compute_viterbi_column(j);
			}
		}

		// store resulting haplotypes
		size_t variant_id = this->column_indexers.at(column_index)->get_variant_id();
		this->genotyping_result.at(variant_id).add_first_haplotype_allele(allele1);
		this->genotyping_result.at(variant_id).add_second_haplotype_allele(allele2);

		if (column_index == 0) break;

		// update best index 
		best_index = this->viterbi_backtrace_columns.at(column_index)->at(best_index);
		column_index -= 1;
	}
#endif
}

#ifdef USE_CUDA
void HMM::compute_forward_prob_gpu() {
    size_t column_count = this->column_indexers.size();
    init(this->forward_columns, column_count);

    for (size_t col = 0; col < column_count; ++col) {
        ColumnIndexer* idx   = column_indexers[col];
        int P                = idx->nr_paths();
        std::vector<double> emis(P*P);
        /* 1. 準備 emission (使用 double) */
        EmissionProbabilityComputer epc(unique_kmers->at(idx->get_variant_id()), probabilities);
        for (int a=0; a<P; ++a) {
            for (int b=0; b<P; ++b) {
                unsigned char al1 = idx->get_allele(a);
                unsigned char al2 = idx->get_allele(b);
                emis[a*P + b] = static_cast<double>(epc.get_emission_probability(al1, al2));
            }
        }

        std::vector<double> prev, trans;
        int prevP = 0;
        if (col > 0) {
            ColumnIndexer* prevIdx = column_indexers[col-1];
            prevP = prevIdx->nr_paths();
            prev.resize(prevP*prevP);
            for (size_t i=0;i<prev.size();++i)
                prev[i] = static_cast<double>(forward_columns[col-1]->at(i));

			size_t prevIndex = prevIdx->get_variant_id();
			size_t curIndex  = idx->get_variant_id();
			size_t prev_pos  = unique_kmers->at(prevIndex)->get_variant_position();
			size_t cur_pos   = unique_kmers->at(curIndex)->get_variant_position();
            // 建立 transition table (prevP² × P²)
            trans.resize(P*P*prevP*prevP);
            TransitionProbabilityComputer tpc(prev_pos, cur_pos, this->recombrate, P, this->uniform, this->effective_N);
            size_t k=0;
            for(int cur1=0;cur1<P;++cur1) for(int cur2=0;cur2<P;++cur2)
              for(int pv1=0;pv1<prevP;++pv1) for(int pv2=0;pv2<prevP;++pv2)
                trans[k++] = static_cast<double>(tpc.compute_transition_prob(prevIdx->get_path(pv1), prevIdx->get_path(pv2),
                                                                            idx->get_path(cur1),  idx->get_path(cur2)));
        }

        // GPU compute
        std::vector<double> cur;
        compute_forward_column_gpu(prev, trans, emis, prevP, P, cur);
		cerr << "forward P = " << P << endl;
		// if (col < 3) {
		// 	std::cerr << "[GPU fwd col " << col << "] ";
		// 	std::cerr << std::setprecision(15);
		// 	for (double v : cur) std::cerr << v << ' ';
		// 	std::cerr << std::endl;
		// }

        // 轉回 long double 儲存
        auto* vec = new std::vector<long double>(cur.begin(), cur.end());
        forward_columns[col] = vec;
        forward_normalization_sums[col] = 1.0L; // 已正規化

        // 可視 k 值釋放 vec 與 prev (同 CPU path) ...
    }
}
void HMM::compute_backward_prob_gpu() {
    size_t n = column_indexers.size();
    if (this->previous_backward_column != nullptr) {
        delete this->previous_backward_column;
        this->previous_backward_column = nullptr;
    }
    if(n==0) return;
    // 預先確保 forward_columns 已存在 (GPU forward 已算好)
    for(size_t i=0;i<n;++i){ if(!forward_columns[i]) throw std::runtime_error("forward column missing"); }

    std::vector<double> nextB; // 下一 col backward
    size_t nextP=0;
    // 初始化最後一列 (全 1  => uniform)
    ColumnIndexer* lastIdx = column_indexers[n-1];
    size_t P_last = lastIdx->nr_paths();
    nextB.assign(P_last*P_last,1.0);
    std::vector<double> nextEmis;

    for(int col=n-1; col>=0; --col){
        ColumnIndexer* idx = column_indexers[col];
        int P = idx->nr_paths();

        // --- 準備 emission (來自 next column，若 col==n-1 則任意 1) ---
        if(col<n-1){ nextEmis.resize(nextP*nextP); ColumnIndexer* nidx = column_indexers[col+1]; EmissionProbabilityComputer epc(unique_kmers->at(nidx->get_variant_id()),probabilities);
            for(int a=0;a<nextP;++a) for(int b=0;b<nextP;++b) nextEmis[a*nextP+b]=static_cast<double>(epc.get_emission_probability(nidx->get_allele(a),nidx->get_allele(b))); }
        else { nextEmis.assign(P*P,1.0);} // not used

        // --- transition table (轉置，大小 nextP²×P²) ---
        std::vector<double> trans;
        if(col<n-1){ trans.resize(nextP*nextP*P*P); size_t k=0;
            ColumnIndexer* nidx=column_indexers[col+1];
            size_t prev_pos=unique_kmers->at(idx->get_variant_id())->get_variant_position();
            size_t cur_pos =unique_kmers->at(nidx->get_variant_id())->get_variant_position();
            TransitionProbabilityComputer tpc(prev_pos,cur_pos,recombrate,nextP,uniform,effective_N);
            for(int np1=0;np1<nextP;++np1) for(int np2=0;np2<nextP;++np2)
              for(int p1=0;p1<P;++p1) for(int p2=0;p2<P;++p2)
                trans[k++]=static_cast<double>(tpc.compute_transition_prob(idx->get_path(p1),idx->get_path(p2),nidx->get_path(np1),nidx->get_path(np2)));
        }
        // --- GPU backward --
        std::vector<double> curB;
        compute_backward_column_gpu(nextB, trans, nextEmis, nextP, P, curB);
		cerr << "backward P = " << P << endl;
        // --- 更新 genotype likelihood & store prev column ---
        size_t variant_id = idx->get_variant_id();
        EmissionProbabilityComputer epc(unique_kmers->at(variant_id),probabilities);
        size_t i=0; for(int a=0;a<P;++a){ for(int b=0;b<P;++b){
            long double fb = forward_columns[col]->at(i) * curB[i] * forward_normalization_sums[col];
            genotyping_result[variant_id].add_to_likelihood(idx->get_allele(a),idx->get_allele(b),fb);
            ++i; }}

        // 準備下一輪
        nextB.assign(curB.begin(),curB.end()); nextP=P; // cur -> next
    }
}

void HMM::compute_viterbi_path_gpu() {
    size_t n=column_indexers.size(); if(n==0) return;
    init(viterbi_columns,n); init(viterbi_backtrace_columns,n);

    // forward sweep
    for(size_t col=0; col<n; ++col){ ColumnIndexer* idx=column_indexers[col]; int P=idx->nr_paths();
        // emission cur
        std::vector<double> emis(P*P);
        EmissionProbabilityComputer epc(unique_kmers->at(idx->get_variant_id()),probabilities);
        for(int a=0;a<P;++a) for(int b=0;b<P;++b) emis[a*P+b]=static_cast<double>(epc.get_emission_probability(idx->get_allele(a),idx->get_allele(b)));
        std::vector<double> prevV; std::vector<double> trans; int prevP=0;
        if(col>0){ ColumnIndexer* pidx=column_indexers[col-1]; prevP=pidx->nr_paths(); prevV.assign(viterbi_columns[col-1]->begin(),viterbi_columns[col-1]->end());
            trans.resize(P*P*prevP*prevP);
            size_t prev_pos=unique_kmers->at(pidx->get_variant_id())->get_variant_position();
            size_t cur_pos =unique_kmers->at(idx ->get_variant_id())->get_variant_position();
            TransitionProbabilityComputer tpc(prev_pos,cur_pos,recombrate,P,uniform,effective_N);
            size_t k=0; for(int c1=0;c1<P;++c1) for(int c2=0;c2<P;++c2)
              for(int p1=0;p1<prevP;++p1) for(int p2=0;p2<prevP;++p2)
                trans[k++]=static_cast<double>(tpc.compute_transition_prob(pidx->get_path(p1),pidx->get_path(p2),idx->get_path(c1),idx->get_path(c2)));
        }
        std::vector<double> curV; std::vector<size_t> arg;
        compute_viterbi_column_gpu(prevV,trans,emis,prevP,P,curV,arg);
        viterbi_columns[col]=new std::vector<long double>(curV.begin(),curV.end());
        auto* bt=new std::vector<size_t>(arg.begin(),arg.end()); viterbi_backtrace_columns[col]=bt;
    }

    // 找最後 column 最大機率索引
    size_t col_last=n-1; auto* last=viterbi_columns[col_last]; size_t bestIdx=0; long double bestVal=0.0;
    for(size_t i=0;i<last->size();++i){ if(last->at(i)>=bestVal){bestVal=last->at(i); bestIdx=i;} }

    // backtrace
    for(int col=n-1; col>=0; --col){ ColumnIndexer* idx=column_indexers[col]; auto paths=idx->get_path_ids_at(bestIdx);
        unsigned char al1=idx->get_allele(paths.first); unsigned char al2=idx->get_allele(paths.second);
        size_t var_id=idx->get_variant_id(); genotyping_result[var_id].add_first_haplotype_allele(al1); genotyping_result[var_id].add_second_haplotype_allele(al2);
        if(col==0) break; bestIdx=viterbi_backtrace_columns[col]->at(bestIdx);
    }
}
#endif

void HMM::compute_forward_column(size_t column_index) {
	assert(column_index < this->column_indexers.size());
	size_t variant_id = this->column_indexers.at(column_index)->get_variant_id();

	// check whether column was computed already
	if (this->forward_columns[column_index] != nullptr) return;

	// get previous column and previous path ids (if existent)
	vector<long double>* previous_column = nullptr;
	ColumnIndexer* previous_indexer = nullptr;
	TransitionProbabilityComputer* transition_probability_computer = nullptr;
	
	// get ColumnIndexer
	ColumnIndexer* column_indexer = column_indexers.at(column_index);
	assert (column_indexer != nullptr);
	// nr of paths
	unsigned short nr_paths = column_indexer->nr_paths();
	
	if (column_index > 0) {
		previous_column = this->forward_columns[column_index-1];
		previous_indexer = this->column_indexers.at(column_index-1);
		size_t prev_index = this->column_indexers.at(column_index-1)->get_variant_id();
		size_t cur_index = this->column_indexers.at(column_index)->get_variant_id();
		size_t prev_pos = this->unique_kmers->at(prev_index)->get_variant_position();
		size_t cur_pos = this->unique_kmers->at(cur_index)->get_variant_position();
		transition_probability_computer = new TransitionProbabilityComputer(prev_pos, cur_pos, this->recombrate, nr_paths, this->uniform, this->effective_N);
		
	}

	// construct new column
	vector<long double>* current_column = new vector<long double>();

	// emission probability computer
	EmissionProbabilityComputer emission_probability_computer(this->unique_kmers->at(variant_id), this->probabilities);

	// normalization
	long double normalization_sum = 0.0L;

	// state index
	size_t i = 0;
	unsigned short nr_prev_paths = 0;
	if (column_index > 0) nr_prev_paths = previous_indexer->nr_paths();
	// iterate over all pairs of current paths
	for (unsigned short path_id1 = 0; path_id1 < nr_paths; ++path_id1) {
		for (unsigned short path_id2 = 0; path_id2 < nr_paths; ++path_id2) {
			// get paths corresponding to path indices
			unsigned short path1 = column_indexer->get_path(path_id1);
			unsigned short path2 = column_indexer->get_path(path_id2);
			long double previous_cell = 0.0L;
			if (column_index > 0) {
				// previous state index
				size_t j = 0;
				// iterate over all pairs of previous paths
				for (unsigned short prev_path_id1 = 0; prev_path_id1 < nr_prev_paths; ++prev_path_id1) {
					for (unsigned short prev_path_id2 = 0; prev_path_id2 < nr_prev_paths; ++prev_path_id2) {
						// forward probability of previous cell
						long double prev_forward = previous_column->at(j);
						// paths corresponding to path indices
						unsigned short prev_path1 = previous_indexer->get_path(prev_path_id1);
						unsigned short prev_path2 = previous_indexer->get_path(prev_path_id2);

						// determine transition probability
						long double transition_prob = transition_probability_computer->compute_transition_prob(prev_path1, prev_path2, path1, path2);
						previous_cell += prev_forward * transition_prob;
						j += 1;
					}
				}
			} else {
				previous_cell = 1.0L;
			}
			// determine alleles current paths (ids) correspond to
			unsigned char allele1 = column_indexer->get_allele(path_id1);
			unsigned char allele2 = column_indexer->get_allele(path_id2);
			// determine emission probability
			long double emission_prob = emission_probability_computer.get_emission_probability(allele1,allele2);

			// set entry of current column
			long double current_cell = previous_cell * emission_prob;
			current_column->push_back(current_cell);
			normalization_sum += current_cell;
			i += 1;
		}
	}

	if (normalization_sum > 0.0L) {
		// normalize the entries in current column to sum up to 1
		transform(current_column->begin(), current_column->end(), current_column->begin(), bind(divides<long double>(), placeholders::_1, normalization_sum));
	} else {
		long double uniform = 1.0L / (long double) current_column->size();
		transform(current_column->begin(), current_column->end(), current_column->begin(),  [uniform](long double c) -> long double {return uniform;});
//		cerr << "Underflow in Forward pass at position: " << this->unique_kmers->at(column_index)->get_variant_position() << ". Column set to uniform." << endl;
	}

	// store the column
	this->forward_columns.at(column_index) = current_column;
	// if (column_index < 3) {
    //     std::cerr << "[CPU fwd col " << column_index << "] ";
    //     std::cerr << std::setprecision(15);
    //     for (long double v : *current_column) std::cerr << v << ' ';
    //     std::cerr << std::endl;
    // }
	if (normalization_sum > 0.0L) {
		this->forward_normalization_sums.at(column_index) = normalization_sum;
	} else {
		this->forward_normalization_sums.at(column_index) = 1.0L;
	}

	if (transition_probability_computer != nullptr) {
		delete transition_probability_computer;
	}
}

void HMM::compute_backward_column(size_t column_index) {
	size_t column_count = this->column_indexers.size();
	assert(column_index < column_count);
	size_t variant_id = this->column_indexers.at(column_index)->get_variant_id();

	// get previous indexers and probabilitycomputers
	ColumnIndexer* previous_indexer = nullptr;
	TransitionProbabilityComputer* transition_probability_computer = nullptr;
	EmissionProbabilityComputer* emission_probability_computer = nullptr;
	vector<long double>* forward_column = this->forward_columns.at(column_index);
	
	// get ColumnIndexer
	ColumnIndexer* column_indexer = column_indexers.at(column_index);
	assert (column_indexer != nullptr);

	// nr of paths
	unsigned short nr_paths = column_indexer->nr_paths();

	if (column_index < column_count-1) {
		assert (this->previous_backward_column != nullptr);
		size_t prev_index = this->column_indexers.at(column_index)->get_variant_id();
		size_t cur_index = this->column_indexers.at(column_index+1)->get_variant_id();
		size_t prev_pos = this->unique_kmers->at(prev_index)->get_variant_position();
		size_t cur_pos = this->unique_kmers->at(cur_index)->get_variant_position();
		transition_probability_computer = new TransitionProbabilityComputer(prev_pos, cur_pos, this->recombrate, nr_paths, this->uniform, this->effective_N);	
		previous_indexer = this->column_indexers.at(column_index+1);
		emission_probability_computer = new EmissionProbabilityComputer(this->unique_kmers->at(this->column_indexers.at(column_index+1)->get_variant_id()), this->probabilities);

		// get forward probabilities (needed for computing posteriors
		if (forward_column == nullptr) {
			// compute index of last column stored
			size_t k = (size_t)sqrt(column_count);
			size_t next = min((size_t) ( (column_index / k) * k ), column_count-1);
			for (size_t j = next+1; j <= column_index; ++j) {
				compute_forward_column(j);
			}
		}

		forward_column = this->forward_columns.at(column_index);
		assert (forward_column != nullptr);
	}

	// construct new column
	vector<long double>* current_column = new vector<long double>();

	// normalization
	long double normalization_sum = 0.0L;

	// normalization of forward-backward
	long double normalization_f_b = 0.0L;

	// state index
	size_t i = 0;
	unsigned short nr_prev_paths = 0;
	if (column_index < column_count - 1) nr_prev_paths = previous_indexer->nr_paths();
	// iterate over all pairs of current paths
	for (unsigned short path_id1 = 0; path_id1 < nr_paths; ++path_id1) {
		for (unsigned short path_id2 = 0; path_id2 < nr_paths; ++path_id2) {
			// get paths corresponding to path indices
			unsigned short path1 = column_indexer->get_path(path_id1);
			unsigned short path2 = column_indexer->get_path(path_id2);
			// get alleles on current paths
			unsigned char allele1 = column_indexer->get_allele(path_id1);
			unsigned char allele2 = column_indexer->get_allele(path_id2);
			long double current_cell = 0.0L;
			if (column_index < column_count - 1) {
				// iterate over previous column (ahead of this)
				size_t j = 0;
				for (unsigned short prev_path_id1 = 0; prev_path_id1 < nr_prev_paths; ++prev_path_id1) {
					for (unsigned short prev_path_id2 = 0; prev_path_id2 < nr_prev_paths; ++prev_path_id2) {
						// paths corresponding to path indices
						unsigned short prev_path1 = previous_indexer->get_path(prev_path_id1);
						unsigned short prev_path2 = previous_indexer->get_path(prev_path_id2);
						// alleles on previous paths
						unsigned char prev_allele1 = previous_indexer->get_allele(prev_path_id1);
						unsigned char prev_allele2 = previous_indexer->get_allele(prev_path_id2);
						long double prev_backward = this->previous_backward_column->at(j);
						// determine transition probability
						long double transition_prob = transition_probability_computer->compute_transition_prob(path1, path2, prev_path1, prev_path2);
						current_cell += prev_backward * transition_prob * emission_probability_computer->get_emission_probability(prev_allele1, prev_allele2);
						j += 1;
					}
				}
			} else {
				current_cell = 1.0L;
			}
			// store computed backward prob in column
			current_column->push_back(current_cell);
			normalization_sum += current_cell;

			// compute forward_prob * backward_prob
			long double forward_backward_prob = forward_column->at(i) * current_cell;
			normalization_f_b += forward_backward_prob;

			// update genotype likelihood
			this->genotyping_result.at(variant_id).add_to_likelihood(allele1, allele2, forward_backward_prob * this->forward_normalization_sums.at(column_index));
			i += 1;
		}
	}

	if (normalization_sum > 0.0L) {
		transform(current_column->begin(), current_column->end(), current_column->begin(), bind(divides<long double>(), placeholders::_1, normalization_sum));
	} else {
		long double uniform = 1.0L / (long double) current_column->size();
		transform(current_column->begin(), current_column->end(), current_column->begin(), [uniform](long double c) -> long double {return uniform;});
//		cerr << "Underflow in Backward pass at position: " << this->unique_kmers->at(column_index)->get_variant_position() << ". Column set to uniform." << endl;
	}

//	cout << "FORWARD COLUMN: " << endl;
//	print_column(forward_column, column_indexer);

//	cout << "BACKWARD COLUMN: "  << endl;
//	print_column(current_column, column_indexer);

	// store computed column (needed for next step)
	if (this->previous_backward_column != nullptr) {
		delete this->previous_backward_column;
		this->previous_backward_column = nullptr;
	}
	this->previous_backward_column = current_column;
	if (emission_probability_computer != nullptr) delete emission_probability_computer;

	// delete forward column as it's not needed any more
	if (this->forward_columns.at(column_index) != nullptr) {
		delete this->forward_columns.at(column_index);
		this->forward_columns.at(column_index) = nullptr;
	}

	if (transition_probability_computer != nullptr) {
		delete transition_probability_computer;
	}
	
//	if (normalization_f_b > 0.0L) {
//		// normalize the GenotypingResults likelihoods 
//		this->genotyping_result.at(column_index).divide_likelihoods_by(normalization_f_b);
//	}
}

void HMM::compute_viterbi_column(size_t column_index) {
	assert(column_index < this->column_indexers.size());
	size_t variant_id = this->column_indexers.at(column_index)->get_variant_id();

	// check whether column was computed already
	if (this->viterbi_columns[column_index] != nullptr) return;

	// get previous column and previous path ids (if existent)
	vector<long double>* previous_column = nullptr;
	ColumnIndexer* previous_indexer = nullptr;
	
	// get ColumnIndexer
	ColumnIndexer* column_indexer = this->column_indexers.at(column_index);
	assert (column_indexer != nullptr);
	// nr of paths
	unsigned short nr_paths = column_indexer->nr_paths();
	
	TransitionProbabilityComputer* transition_probability_computer = nullptr;
	if (column_index > 0) {
		previous_column = this->viterbi_columns[column_index-1];
		previous_indexer = this->column_indexers.at(column_index-1);
		size_t prev_index = this->column_indexers.at(column_index-1)->get_variant_id();
		size_t cur_index = this->column_indexers.at(column_index)->get_variant_id();
		size_t prev_pos = this->unique_kmers->at(prev_index)->get_variant_position();
		size_t cur_pos = this->unique_kmers->at(cur_index)->get_variant_position();
		transition_probability_computer = new TransitionProbabilityComputer(prev_pos, cur_pos, this->recombrate, nr_paths, this->uniform, this->effective_N);
	}

	// construct new column
	vector<long double>* current_column = new vector<long double>();

	// emission probability computer
	EmissionProbabilityComputer emission_probability_computer(this->unique_kmers->at(variant_id), this->probabilities);

	// normalization 
	long double normalization_sum = 0.0L;

	// backtrace table
	vector<size_t>* backtrace_column = new vector<size_t>();

	// state index
	size_t i = 0;
	unsigned short nr_prev_paths = 0;
	if (column_index > 0) nr_prev_paths = previous_indexer->nr_paths();
	// iterate over all pairs of current paths
	for (unsigned short path_id1 = 0; path_id1 < nr_paths; ++path_id1) {
		for (unsigned short path_id2 = 0; path_id2 < nr_paths; ++path_id2) {
			// get paths corresponding to path indices
			unsigned short path1 = column_indexer->get_path(path_id1);
			unsigned short path2 = column_indexer->get_path(path_id2);
			long double previous_cell = 0.0L;
			if (column_index > 0) {
				// previous state index
				size_t j = 0;
				long double max_value = 0.0L;
				size_t max_index = 0;
				// iterate over all pairs of previous paths
				for (unsigned short prev_path_id1 = 0; prev_path_id1 < nr_prev_paths; ++prev_path_id1) {
					for (unsigned short prev_path_id2 = 0; prev_path_id2 < nr_prev_paths; ++prev_path_id2) {
						// paths corresponding to path indices
						unsigned short prev_path1 = previous_indexer->get_path(prev_path_id1);
						unsigned short prev_path2 = previous_indexer->get_path(prev_path_id2);
						// probability of previous cell
						long double prev_prob = previous_column->at(j);
						// determine transition probability
						long double transition_prob = transition_probability_computer->compute_transition_prob(prev_path1, prev_path2, path1, path2);
						prev_prob *= transition_prob;
						if (prev_prob >= max_value) {
							max_value = prev_prob;
							max_index = j;
						}
						j += 1;
					}
				}
				previous_cell = max_value;
				backtrace_column->push_back(max_index);
			} else {
				previous_cell = 1.0L;
			}

			// determine alleles current paths (ids) correspond to
			unsigned char allele1 = column_indexer->get_allele(path_id1);
			unsigned char allele2 = column_indexer->get_allele(path_id2);
			// determine emission probability
			long double emission_prob = emission_probability_computer.get_emission_probability(allele1,allele2);
			// set entry of current column
			long double current_cell = previous_cell * emission_prob;
			current_column->push_back(current_cell);
			normalization_sum += current_cell;
			i += 1;
		}
	}

	if (normalization_sum > 0.0L) {
		// normalize the entries in current column to sum up to 1 
		transform(current_column->begin(), current_column->end(), current_column->begin(), bind(divides<long double>(), placeholders::_1, normalization_sum));
	} else {
		long double uniform = 1.0L / (long double) current_column->size();
		transform(current_column->begin(), current_column->end(), current_column->begin(),  [uniform](long double c) -> long double {return uniform;});
//		cerr << "Underflow in Viterbi pass at position: " << this->unique_kmers->at(column_index)->get_variant_position() << ". Column set to uniform." << endl;
	}

	// store the column
	this->viterbi_columns.at(column_index) = current_column;
	if (column_index > 0) assert(backtrace_column->size() == column_indexer->nr_paths()*column_indexer->nr_paths());
	this->viterbi_backtrace_columns.at(column_index) = backtrace_column;
	
	if (transition_probability_computer != nullptr) {
		delete transition_probability_computer;
	}
}

vector<GenotypingResult> HMM::get_genotyping_result() const {
	return this->genotyping_result;
}

vector<GenotypingResult> HMM::move_genotyping_result() {
	return move(this->genotyping_result);
}
