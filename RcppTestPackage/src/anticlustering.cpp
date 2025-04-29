#include <gecode/int.hh>
#include <gecode/search.hh>
#include <iostream>
#include <map>
#include <Rcpp.h>
#include <iostream>
#include <sstream>
#include <string>



using namespace Gecode;
using namespace Rcpp;



// variant 2 from Lars Torben Schwabe
class CorrectColoring : public Space {
public:
    IntVarArray vertex_anticluster_assignment;
    int selection_index;
    int anticluster_count;
    int edge_count;
    std::vector<int> edge_table_u;
    std::vector<int> edge_table_v;
    int allowed_sum_lower;
    int allowed_sum_upper;

    CorrectColoring(int selection_index,
                    int anticluster_count,
                    int edge_count,
                    const std::vector<int>& edge_table_u,
                    const std::vector<int>& edge_table_v,
                    int allowed_sum_lower,
                    int allowed_sum_upper)
            : vertex_anticluster_assignment(*this, selection_index, 1, anticluster_count),
              selection_index(selection_index),
              anticluster_count(anticluster_count),
              edge_count(edge_count),
              edge_table_u(edge_table_u),
              edge_table_v(edge_table_v),
              allowed_sum_lower(allowed_sum_lower),
              allowed_sum_upper(allowed_sum_upper) {
        // neighbors don't have the same color
        for (int i = 0; i < edge_count; ++i) {
            rel(*this, vertex_anticluster_assignment[edge_table_u[i]], IRT_NQ, vertex_anticluster_assignment[edge_table_v[i]]);
        }

        // ~ same amount of points per Anticluster
        for (int i = 1; i <= anticluster_count; ++i) {
            count(*this, vertex_anticluster_assignment, i, IRT_GQ, allowed_sum_lower);
            count(*this, vertex_anticluster_assignment, i, IRT_LQ, allowed_sum_upper);
        }

        branch(*this, vertex_anticluster_assignment, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
    }


    // search support
    CorrectColoring(CorrectColoring& s)
            : Space(s),
              selection_index(s.selection_index),
              anticluster_count(s.anticluster_count),
              edge_count(s.edge_count),
              edge_table_u(s.edge_table_u),
              edge_table_v(s.edge_table_v),
              allowed_sum_lower(s.allowed_sum_lower),
              allowed_sum_upper(s.allowed_sum_upper)
    {
        vertex_anticluster_assignment.update(*this, s.vertex_anticluster_assignment);
    }


    virtual Space* copy(void) {
        return new CorrectColoring(*this);
    }


    void print(void) const {
        std::cout << vertex_anticluster_assignment << std::endl;
    }


    std::vector<int> getSolution(void) const {
        std::vector<int> sol(vertex_anticluster_assignment.size());
        for (int i = 0; i < vertex_anticluster_assignment.size(); ++i) {
            sol[i] = vertex_anticluster_assignment[i].val();
        }
        return sol;
    }
};



// [[Rcpp::export]]
Rcpp::NumericVector anticlustering(int n, int k, int d_count, Rcpp::NumericVector d_sorted, Rcpp::NumericMatrix d_matrix) {
    // TODO: check arguments etc.
    int data_count = n;
    int anticluster_count = k;
    int distance_count_unique = d_count;
    std::vector<double> sorted_distances_unique = Rcpp::as<std::vector<double>>(d_sorted);
    std::vector<std::vector<double>> distance_matrix;
    for (int i = 0; i < d_matrix.nrow(); ++i) {
        NumericVector zwischen = d_matrix.row(i);
        std::vector<double> zwischen_zwei = Rcpp::as<std::vector<double>>(zwischen);
        distance_matrix.push_back(zwischen_zwei);
    }
    std::vector<int> edge_table_u;
    std::vector<int> edge_table_v;
    int edge_count = 0;
    double current_distance;  // current distance investigated
    std::vector<int> vertex_index(data_count, -1);  // maps the true vertex-value (index in distance matrix) to the selection-index of the current selection
    int selection_index = 0;
    int allowed_sum_lower;
    int allowed_sum_upper;
    std::vector<int> solution;

    // algorithm
    for (int i = 0; i < distance_count_unique; ++i) {
        current_distance = sorted_distances_unique[i];

        // add edges of nodes with current distance to graph
        for (int u  = 0; u < data_count; ++u) {
            for (int v = u; v < data_count; ++v) {
                if (distance_matrix[u][v] == current_distance) {
                    if (vertex_index[u] == -1) {
                        vertex_index[u] = selection_index;
                        edge_table_u.push_back(selection_index);
                        ++selection_index;
                        if (vertex_index[v] == -1) {
                            vertex_index[v] = selection_index;
                            edge_table_v.push_back(selection_index);
                            ++selection_index;
                        } else {
                            edge_table_v.push_back(vertex_index[v]);
                        }
                    } else if (vertex_index[v] == -1) {
                        vertex_index[v] = selection_index;
                        edge_table_u.push_back(vertex_index[u]);
                        edge_table_v.push_back(selection_index);
                        ++selection_index;
                    } else {
                        edge_table_u.push_back(vertex_index[u]);
                        edge_table_v.push_back(vertex_index[v]);
                    }
                    edge_count++;
                }
            }
        }

        double portion = static_cast<double>(selection_index) / static_cast<double>(anticluster_count);
        allowed_sum_upper = ceil(portion);
        allowed_sum_lower = floor(portion);

        CorrectColoring* m = new CorrectColoring(selection_index, anticluster_count, edge_count, edge_table_u, edge_table_v, allowed_sum_lower, allowed_sum_upper);
        DFS<CorrectColoring> e(m);
        delete m;
        CorrectColoring* current_anticlustering = e.next();
        
        if (current_anticlustering == NULL) {
            delete current_anticlustering;
            break;
        }

        solution = current_anticlustering->getSolution();
        delete current_anticlustering;
    }


    int pseudo_random_anticluster = 0;
    Rcpp::NumericMatrix output(data_count, 3);
    for (int i = 0; i < data_count; ++i) {
        if(vertex_index[i] == -1 or vertex_index[i] >= solution.size()) {  // vertex tried for a solution or was tried but failed and is not included in solution
            pseudo_random_anticluster = (pseudo_random_anticluster % anticluster_count) + 1;
            output(i, 0) = pseudo_random_anticluster;
            output(i, 1) = 1;
        } else {
            output(i, 0) = solution[vertex_index[i]];
            output(i, 1) = 0;
        }
        output(i, 2) = current_distance;
    }
    return output;
}