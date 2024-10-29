#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <errno.h>
#include <vector>
#include <list>
#include <cstring>

#define ABS(A) (((A) >= 0) ? (A) : -(A))
#define HARD(A) (((A) < 0)?1:0)
#define INFTY 1000000

using namespace std;
namespace py = pybind11;

double safe_exp(double x) {
    if (x > 700) {
        return 1.014232054735005e+304;
    }
    if (x < -700) {
        return 9.859676543759770e-305;
    }
    return exp(x);
}

double safe_log(double x) {
    if (x < 9.859676543759770e-305) {
        return -700;
    }
    return log(x);
}

double phi(double x) {
    static const double lim_tanh = 31.0;
    static const double min_tanh = 6.883382752676208e-14; //log( (exp((double)lim) + 1)/(exp((double)lim) - 1));

    if (x > lim_tanh) {
        return 2 * safe_exp(-x);
    }

    return -safe_log(tanh(x / 2));
}

// Adjusted Min-Sum Decoder
py::array_t<double> AdjMinSum(py::array_t<int> H, py::array_t<double> in_llr_array, int max_iter) {
    // Auxiliary matrices
    int m = H.shape(0);
    int n = H.shape(1);
    vector<int> col_weight(n);
    vector<int> row_weight(m);
    auto h = H.unchecked<2>();
    for (py::ssize_t i = 0; i < m; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            col_weight[j] += h(i, j);
            row_weight[i] += h(i, j);
        }
    }

    // int cmax = *max_element(col_weight.begin(), col_weight.end());
    // int rmax = *max_element(row_weight.begin(), row_weight.end());

    vector<vector<double>>  R_msgs(m, vector<double>(n)); // messages from check to variable nodes
    vector<vector<double>>  Q_abs(m, vector<double>(n)); // messages from variable to check nodes
    vector<vector<int>>     Q_signs(m, vector<int>(n));
    vector<double> out_llr(n, 0);
    double sum_tanhs = 0;
    int sum_sign = 0;
    int sign = 0;
    double temp = 0;
    double Q_min = INFTY;
    double Q_max = 0;
    int min_index = 0;
    int max_index = 0;
    auto in_llr = in_llr_array.unchecked<1>();




    vector<vector<pair<int, int>>> msgs_col(n, vector<pair<int, int>>());
    vector<vector<int>> col_row(n, vector<int>(m));
    vector<vector<int>> col_N(n, vector<int>(m));
    vector<int> count(n, 0);


    for (int i = 0; i < m; ++i) {
        int row_count = 0;
        for (int v = 0; v < n; ++v) {
            if (h(i, v) == 1) {
                col_row[v][count[v]] = i;
                col_N[v][count[v]] = row_count++;
                count[v]++;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < col_weight[i]; ++j) {
            msgs_col[i].push_back(make_pair(col_row[i][j], col_N[i][j]));
        }
    }

    // Initialization
    for (int i = 0; i < n; ++i) {
        out_llr[i] = in_llr(i);
        // y[i] = HARD(out_llr[i]);

        for (int j = 0; j < col_weight[i]; ++j) {
            Q_signs[msgs_col[i][j].first][msgs_col[i][j].second] = 0;
            if (in_llr(i) < 0) {
                Q_signs[msgs_col[i][j].first][msgs_col[i][j].second] = 1;
            }
            Q_abs[msgs_col[i][j].first][msgs_col[i][j].second] = fabs(in_llr(i));
        }
    }

    for (int loop = 0; loop < max_iter; ++loop) {
        // Update R
        for (int j = 0; j < m; j++) {
            sum_tanhs = 0;
            sum_sign = 0;
            Q_min = INFTY;
            Q_max = 0;
            min_index = 0;
            max_index = 0;

            for (int k = 0; k < row_weight[j]; k++) {
                sum_tanhs += phi(Q_abs[j][k]);
                sum_sign ^= Q_signs[j][k];
                if (Q_abs[j][k] < Q_min) {
                    Q_min = Q_abs[j][k];
                    min_index = k;
                }
                if (Q_abs[j][k] >= Q_max) {
                    Q_max = Q_abs[j][k];
                    max_index = k;
                }
            }
            for (int k = 0; k < row_weight[j]; k++) {
                sign = sum_sign ^ Q_signs[j][k];

                /* Enhanced Adjusted Min-Sum. Better performance is
                 * obtained when for the second stored message, the SP
                 * outgoing message along the maximum incoming
                 * reliability (or in other words the outgoing SP
                 * message with minimum reliability) is used instead.
                 * This decoder is a slight modification of [5] and is
                 * observed to yield better performance than the one
                 * provided in [5].
                 */
                if (k == min_index) {
                    R_msgs[j][k] = (1 - 2 * sign) * phi(sum_tanhs - phi(Q_min));
                }
                else {
                    R_msgs[j][k] = (1 - 2 * sign) * phi(sum_tanhs - phi(Q_max));
                }
            }
        }
        // Update Q
        for (int i = 0; i < n; i++) {
            out_llr[i] = in_llr(i);

            for (int k = 0; k < col_weight[i]; k++) {
                out_llr[i] += R_msgs[msgs_col[i][k].first][msgs_col[i][k].second];
            }

            // y[i] = HARD(out_llr[i]);

            for (int k = 0; k < col_weight[i]; k++) {
                temp = out_llr[i] - R_msgs[msgs_col[i][k].first][msgs_col[i][k].second];
                Q_signs[msgs_col[i][k].first][msgs_col[i][k].second] = 0;
                if (temp < 0) {
                    Q_signs[msgs_col[i][k].first][msgs_col[i][k].second] = 1;
                }
                Q_abs[msgs_col[i][k].first][msgs_col[i][k].second] = fabs(temp);
            }
        }
    }
    auto out_llr_array = py::array_t<double>(out_llr.size(), out_llr.data());
    return out_llr_array;
}


PYBIND11_MODULE(minsum, m) {
    m.def("adjusted_min_sum", &AdjMinSum);
}