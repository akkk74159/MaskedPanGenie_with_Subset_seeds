#pragma once
#include <vector>
#include <cstddef>

// Forward
void compute_forward_column_gpu(const std::vector<double>& prevF,
                                const std::vector<double>& trans,
                                const std::vector<double>& emis,
                                int prevP, int P,
                                std::vector<double>& outCur);
// Backward
void compute_backward_column_gpu(const std::vector<double>& nextB,
                                 const std::vector<double>& trans,
                                 const std::vector<double>& emisNext,
                                 int nextP, int P,
                                 std::vector<double>& outCur);
// Viterbi
void compute_viterbi_column_gpu(const std::vector<double>& prevV,
                                const std::vector<double>& trans,
                                const std::vector<double>& emisCur,
                                int prevP, int P,
                                std::vector<double>& outCur,
                                std::vector<size_t>& argmax);