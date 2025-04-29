#include <gecode/int.hh>
#include <gecode/search.hh>
#include <gecode/gist.hh>
#include <iostream>
#include <map>
#include <Rcpp.h>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <random>



using namespace Gecode;
using namespace Rcpp;



/*
##########################################################
#   ___  __  __ ______ __   ___ __    __ __  __  ______  #
#  // \\ ||\ || | || | ||  //   ||    || || (( \ | || |  #
#  ||=|| ||\\||   ||   || ((    ||    || ||  \\    ||    #
#  || || || \||   ||   ||  \\__ ||__| \\_// \_))   ||    #
#                                                        #
##########################################################

The model and code contained in the following section are
intended for the implementation in the anticlust package.
*/



// this model is used for anticlust
// uses the constraint model variant 2 from Lars Torben Schwabe
class CorrectColoringAnticlust : public Space {
public:
    IntVarArray vertex_anticluster_assignment;
    int selection_index;
    int anticluster_count;
    int edge_count;
    std::vector<int> edge_table_u;
    std::vector<int> edge_table_v;
    std::vector<int> target_groups_sizes;
	int symmetry_breaking;

    CorrectColoringAnticlust(int selection_index,
                    int anticluster_count,
                    int edge_count,
                    const std::vector<int>& edge_table_u,
                    const std::vector<int>& edge_table_v,
                    const std::vector<int>& target_groups_sizes,
					int symmetry_breaking)
            : vertex_anticluster_assignment(*this, selection_index, 1, anticluster_count),
              selection_index(selection_index),
              anticluster_count(anticluster_count),
              edge_count(edge_count),
              edge_table_u(edge_table_u),
              edge_table_v(edge_table_v),
              target_groups_sizes(target_groups_sizes),
			  symmetry_breaking(symmetry_breaking) {
        
		if(symmetry_breaking == 1 || symmetry_breaking == 3) {
			rel(*this, vertex_anticluster_assignment[0], IRT_EQ, 1);  // symmetry breaking
		}

		// neighbors don't have the same color
        for (int i = 0; i < edge_count; ++i) {
            rel(*this, vertex_anticluster_assignment[edge_table_u[i]-1], IRT_NQ, vertex_anticluster_assignment[edge_table_v[i]-1]);
        }

        // ~ same amount of points per Anticluster
        for (int i = 1; i <= anticluster_count; ++i) {
            count(*this, vertex_anticluster_assignment, i, IRT_LQ, target_groups_sizes[i-1]);
        }

		if(symmetry_breaking == 2 || symmetry_breaking == 3) {
			Symmetries syms;
			syms << ValueSymmetry(IntArgs::create(anticluster_count + 1, 1));
			Rnd r(std::rand());
			branch(*this, vertex_anticluster_assignment, INT_VAR_AFC_MAX(), INT_VAL_RND(r), syms);  // INT_VAR_AFC_MAX() is extremely important for the solver to not take an eternity!
		} else {
			Rnd r(std::rand());
			branch(*this, vertex_anticluster_assignment, INT_VAR_AFC_MAX(), INT_VAL_RND(r));  // INT_VAR_AFC_MAX() is extremely important for the solver to not take an eternity!
		}
    }


    // search support
    CorrectColoringAnticlust(CorrectColoringAnticlust& s)
            : Space(s),
              selection_index(s.selection_index),
              anticluster_count(s.anticluster_count),
              edge_count(s.edge_count),
              edge_table_u(s.edge_table_u),
              edge_table_v(s.edge_table_v),
              target_groups_sizes(s.target_groups_sizes),
			  symmetry_breaking(symmetry_breaking)
    {
        vertex_anticluster_assignment.update(*this, s.vertex_anticluster_assignment);
    }


    virtual Space* copy(void) {
        return new CorrectColoringAnticlust(*this);
    }


    void print(void) const {
        std::cout << vertex_anticluster_assignment << std::endl;
    }


    std::vector<int> getSolution(void) const {
        std::vector<int> solution(vertex_anticluster_assignment.size());
        for (int i = 0; i < vertex_anticluster_assignment.size(); ++i) {
            solution[i] = vertex_anticluster_assignment[i].val();
        }
        return solution;
    }
};



// this is the method integrated in the anticlust package
// only set number_of_solutions higher than one for the last second last iteration, otherwise it will be computed every iteration!
// multiple solutions through restarts
// [[Rcpp::export]]
Rcpp::NumericMatrix solve_k_graph_coloring_anticlust(int selection_index, int k, int edge_count, Rcpp::NumericVector edge_table_u, Rcpp::NumericVector edge_table_v, Rcpp::NumericVector target_groups_sizes, int number_of_threads, int number_of_solutions, int symmetry_breaking) {
    CorrectColoringAnticlust* m = new CorrectColoringAnticlust(
        selection_index, 
        k, 
        edge_count, 
        Rcpp::as<std::vector<int>>(edge_table_u), 
        Rcpp::as<std::vector<int>>(edge_table_v), 
        Rcpp::as<std::vector<int>>(target_groups_sizes),
		symmetry_breaking);
    Search::Options o;
	o.threads = number_of_threads;
    DFS<CorrectColoringAnticlust> e(m, o);
    delete m;
    CorrectColoringAnticlust* current_anticlustering = e.next();
    if (current_anticlustering == NULL) {
        Rcpp::NumericMatrix output(1, 1);
        output(0,0) = -1;
        delete current_anticlustering;
        return output;
    } else {
		// first solution
        Rcpp::NumericMatrix output(selection_index, number_of_solutions);
		std::vector<int> solution = current_anticlustering->getSolution();
		for (int j = 0; j < selection_index; ++j) {
			output(j, 0) = solution[j];
		}
		delete current_anticlustering;

		// further solutions, additional solutions may be similar to the first one (especially if there are only a few solutions) but it should be way faster this way
        for (int i = 1; i < number_of_solutions; ++i) {

			// rebuild problem instance
			CorrectColoringAnticlust* m = new CorrectColoringAnticlust(
				selection_index, 
				k, 
				edge_count, 
				Rcpp::as<std::vector<int>>(edge_table_u), 
				Rcpp::as<std::vector<int>>(edge_table_v), 
				Rcpp::as<std::vector<int>>(target_groups_sizes),
				symmetry_breaking);

			// start new search which will (hopefully) yield new values according to random value selection
			DFS<CorrectColoringAnticlust> e(m, o);
    		delete m;
			CorrectColoringAnticlust* current_anticlustering = e.next();

			if (current_anticlustering != NULL) {
				std::vector<int> solution = current_anticlustering->getSolution();
				for (int j = 0; j < selection_index; ++j) {
					output(j, i) = solution[j];
				}
			} else {
				output(0,i) = -1;
			}
			delete current_anticlustering;
        }

        return output;
    }
}



/*
########################################################################
#   ____ _   _ ____   ____ ____  __ ___  ___  ____ __  __ ______  __   #
#  ||    \\ // || \\ ||    || \\ || ||\\//|| ||    ||\ || | || | (( \  #
#  ||==   )X(  ||_// ||==  ||_// || || \/ || ||==  ||\\||   ||    \\   #
#  ||___ // \\ ||    ||___ || \\ || ||    || ||___ || \||   ||   \_))  #
#																	   #
########################################################################

The models and code contained in the following section are expreiments
and not intended for the implementation in the anticlust package.
*/



// this model is only used in my cpp implementation of the algorithm and not in anticlust
// differs only by a value offset (edge table entries in the R version start with 1 instead of 0)
// uses the constraint model variant 2 from Lars Torben Schwabe
class CorrectColoring : public Space {
public:
    IntVarArray vertex_anticluster_assignment;
    int selection_index;
    int anticluster_count;
    int edge_count;
    std::vector<int> edge_table_u;
    std::vector<int> edge_table_v;
    std::vector<int> target_groups_sizes;

    CorrectColoring(int selection_index,
                    int anticluster_count,
                    int edge_count,
                    const std::vector<int>& edge_table_u,
                    const std::vector<int>& edge_table_v,
                    const std::vector<int>& target_groups_sizes)
            : vertex_anticluster_assignment(*this, selection_index, 1, anticluster_count),
              selection_index(selection_index),
              anticluster_count(anticluster_count),
              edge_count(edge_count),
              edge_table_u(edge_table_u),
              edge_table_v(edge_table_v),
              target_groups_sizes(target_groups_sizes) {

        //rel(*this, vertex_anticluster_assignment[0], IRT_EQ, 1);

        // ~ same amount of points per Anticluster
        for (int i = 1; i <= anticluster_count; ++i) {
            count(*this, vertex_anticluster_assignment, i, IRT_LQ, target_groups_sizes[i-1]);
        }

        // neighbors don't have the same color
        for (int i = 0; i < edge_count; ++i) {
            rel(*this, vertex_anticluster_assignment[edge_table_u[i]], IRT_NQ, vertex_anticluster_assignment[edge_table_v[i]]);
        }

        //Symmetries syms;
        //syms << ValueSymmetry(IntArgs::create(anticluster_count + 1, 1));
		Rnd r(std::rand());
        branch(*this, vertex_anticluster_assignment, INT_VAR_AFC_MAX(), INT_VAL_RND(r));  // INT_VAR_AFC_MAX() is extremely important for the solver to not take an eternity!
    }


    // search support
    CorrectColoring(CorrectColoring& s)
            : Space(s),
              selection_index(s.selection_index),
              anticluster_count(s.anticluster_count),
              edge_count(s.edge_count),
              edge_table_u(s.edge_table_u),
              edge_table_v(s.edge_table_v),
              target_groups_sizes(s.target_groups_sizes)
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
        std::vector<int> solution(vertex_anticluster_assignment.size());
        for (int i = 0; i < vertex_anticluster_assignment.size(); ++i) {
            solution[i] = vertex_anticluster_assignment[i].val();
        }
        return solution;
    }
};



// experimental diversity model, work in progress, does not work yet
// constraint function missing, this function must compute the diversity on the variable-values of the current solution
// distance matrix must be parsed as well to look up the distance for the diversity formula
class Diversity : public Space {
public:
    IntVarArray vertex_anticluster_assignment;
    int selection_index;
    int anticluster_count;
    std::vector<int> target_groups_sizes;

    Diversity(int selection_index,
                    int anticluster_count,
                    int edge_count,
                    const std::vector<int>& target_groups_sizes)
            : vertex_anticluster_assignment(*this, selection_index, 1, anticluster_count),
              selection_index(selection_index),
              anticluster_count(anticluster_count),
              target_groups_sizes(target_groups_sizes) {

        // ~ same amount of points per Anticluster
        for (int i = 1; i <= anticluster_count; ++i) {
            count(*this, vertex_anticluster_assignment, i, IRT_LQ, target_groups_sizes[i-1]);
        }

        branch(*this, vertex_anticluster_assignment, INT_VAR_AFC_MAX(), INT_VAL_MIN());  // der ist es
    }


	virtual void constraint(const Space& _b) {
		const Diversity& b = static_cast<const Diversity&>(_b);


		
	}


    // search support
    Diversity(Diversity& s)
            : Space(s),
              selection_index(s.selection_index),
              anticluster_count(s.anticluster_count),
              target_groups_sizes(s.target_groups_sizes)
    {
        vertex_anticluster_assignment.update(*this, s.vertex_anticluster_assignment);
    }


    virtual Space* copy(void) {
        return new Diversity(*this);
    }


    void print(void) const {
        std::cout << vertex_anticluster_assignment << std::endl;
    }


    std::vector<int> getSolution(void) const {
        std::vector<int> solution(vertex_anticluster_assignment.size());
        for (int i = 0; i < vertex_anticluster_assignment.size(); ++i) {
            solution[i] = vertex_anticluster_assignment[i].val();
        }
        return solution;
    }
};



// for an experimental version of the own R implementation option to use different approaches to multiple solutions
// only set number_of_solutions higher than one for the last second last iteration, otherwise it will be computed every iteration!
// multiple solutions through simple next() calls
// [[Rcpp::export]]
Rcpp::NumericMatrix solve_k_graph_coloring_simple_next_call(int selection_index, int k, int edge_count, Rcpp::NumericVector edge_table_u, Rcpp::NumericVector edge_table_v, Rcpp::NumericVector target_groups_sizes, int number_of_threads, int number_of_solutions) {
    CorrectColoringAnticlust* m = new CorrectColoringAnticlust(
        selection_index, 
        k, 
        edge_count, 
        Rcpp::as<std::vector<int>>(edge_table_u), 
        Rcpp::as<std::vector<int>>(edge_table_v), 
        Rcpp::as<std::vector<int>>(target_groups_sizes),
		0);
    Search::Options o;
	o.threads = number_of_threads;
    DFS<CorrectColoringAnticlust> e(m, o);
    delete m;
    CorrectColoringAnticlust* current_anticlustering = e.next();
    if (current_anticlustering == NULL) {
        Rcpp::NumericMatrix output(1, 1);
        output(0,0) = -1;
        delete current_anticlustering;
        return output;
    } else {
		// first solution
        Rcpp::NumericMatrix output(selection_index, number_of_solutions);
		std::vector<int> solution = current_anticlustering->getSolution();
		for (int j = 0; j < selection_index; ++j) {
			output(j, 0) = solution[j];
		}
		delete current_anticlustering;

		// further solutions, additional solutions may be identical to the first one (especially if there are only a few solutions) but it should be way faster this way
        for (int i = 1; i < number_of_solutions; ++i) {

			// query search again
			CorrectColoringAnticlust* current_anticlustering = e.next();

			if (current_anticlustering != NULL) {
				std::vector<int> solution = current_anticlustering->getSolution();
				for (int j = 0; j < selection_index; ++j) {
					output(j, i) = solution[j];
				}
			} else {
				output(0,i) = -1;
			}
			delete current_anticlustering;
        }

        return output;
    }
}



// for an experimental version of the own R implementation with the option to use different approaches to multiple solutions
// only set number_of_solutions higher than one for the last second last iteration, otherwise it will be computed every iteration!
// multiple solutions through computing many different solutions by repeatedly calling next() and then picking a few randomly
// same solution may be picked twice, can be prevented by saving previously chosen indices, but is rather unlikely and I think it's ugly -> not considered for the thesis (I simply disregard duplicate solutions for the data and compute a new one)
// different heuristics on how to choose number of querries possible
// [[Rcpp::export]]
Rcpp::NumericMatrix solve_k_graph_coloring_repeated_next_calls_and_random_pick(int selection_index, int k, int edge_count, Rcpp::NumericVector edge_table_u, Rcpp::NumericVector edge_table_v, Rcpp::NumericVector target_groups_sizes, int number_of_threads, int number_of_solutions, int number_of_querries) {
	// higher n -> more solutions -> more ground needs to be covered for good spread; 
	// more solutions wanted -> cover more ground to have them spread further; 
	// higher k -> less solutions -> solutions may be further apart -> more ground needs to be covered
	// int number_of_querries = (selection_index * number_of_solutions * k);
	
	CorrectColoringAnticlust* m = new CorrectColoringAnticlust(
        selection_index, 
        k, 
        edge_count, 
        Rcpp::as<std::vector<int>>(edge_table_u), 
        Rcpp::as<std::vector<int>>(edge_table_v), 
        Rcpp::as<std::vector<int>>(target_groups_sizes),
		0);
    Search::Options o;
	o.threads = number_of_threads;
    DFS<CorrectColoringAnticlust> e(m, o);
    delete m;
    CorrectColoringAnticlust* current_anticlustering = e.next();
    if (current_anticlustering == NULL) {
        Rcpp::NumericMatrix output(1, 1);
        output(0,0) = -1;
        delete current_anticlustering;
        return output;
    } else {
		// first solution
        Rcpp::NumericMatrix output(selection_index, number_of_solutions);
		std::vector<int> solution = current_anticlustering->getSolution();
		for (int j = 0; j < selection_index; ++j) {
			output(j, 0) = solution[j];
		}
		delete current_anticlustering;

		if (number_of_solutions > 1) {
			Rcpp::NumericMatrix querried_solutions(selection_index, number_of_querries);
			int number_of_actually_querried_solutions = 0;

			for (int i = 0; i < number_of_querries; ++i) {
				// query search again
				CorrectColoringAnticlust* current_anticlustering = e.next();

				if (current_anticlustering != NULL) {
					++number_of_actually_querried_solutions;
					std::vector<int> solution = current_anticlustering->getSolution();
					for (int j = 0; j < selection_index; ++j) {
						querried_solutions(j, i) = solution[j];
					}
				} else {
					break;
				}
				delete current_anticlustering;
			}

			std::random_device rd; // obtain a random number from hardware
			std::mt19937 gen(rd()); // seed the generator
			std::uniform_int_distribution<> distr(0, number_of_actually_querried_solutions-1); // define the range

			// further solutions, additional solutions may be identical to the first one (especially if there are only a few solutions) but it should be way faster this way
			for (int i = 1; i < number_of_solutions && i < number_of_actually_querried_solutions; ++i) {
				int random_solution_index = distr(gen);
				for (int j = 0; j < selection_index; ++j) {
					output(j, i) = querried_solutions(j, random_solution_index);
				}
			}
		}

        return output;
    }
}



// method is used in own R implementation of the algorithm, do not use in anticlust
// only set number_of_solutions higher than one for the last second last iteration, otherwise it will be computed every iteration!
// multiple solutions through restarts
// [[Rcpp::export]]
Rcpp::NumericVector solve_k_graph_coloring(int selection_index, int k, int edge_count, Rcpp::NumericVector edge_table_u, Rcpp::NumericVector edge_table_v, Rcpp::NumericVector target_groups_sizes, int number_of_threads, int number_of_solutions) {
CorrectColoringAnticlust* m = new CorrectColoringAnticlust(
        selection_index, 
        k, 
        edge_count, 
        Rcpp::as<std::vector<int>>(edge_table_u), 
        Rcpp::as<std::vector<int>>(edge_table_v), 
        Rcpp::as<std::vector<int>>(target_groups_sizes),
		0);

    Search::Options o;
	//o.cutoff = Search::Cutoff::constant(1000);
    DFS<CorrectColoringAnticlust> e(m, o);
    delete m;
    CorrectColoringAnticlust* current_anticlustering = e.next();
    if (current_anticlustering == NULL) {
        Rcpp::NumericMatrix output(1, 1);
        output(0,0) = -1;
        delete current_anticlustering;
        return output;
    } else {
		// first solution
        Rcpp::NumericMatrix output(selection_index, number_of_solutions);
		std::vector<int> solution = current_anticlustering->getSolution();
		for (int j = 0; j < selection_index; ++j) {
			output(j, 0) = solution[j];
		}
		delete current_anticlustering;

		// further solutions, additional solutions may be identical to the first one (especially if there are only a few solutions) but it should be way faster this way
        for (int i = 1; i < number_of_solutions; ++i) {
			// rebuild problem instance
			CorrectColoringAnticlust* m = new CorrectColoringAnticlust(
				selection_index, 
				k, 
				edge_count, 
				Rcpp::as<std::vector<int>>(edge_table_u), 
				Rcpp::as<std::vector<int>>(edge_table_v), 
				Rcpp::as<std::vector<int>>(target_groups_sizes),
				0);

			// start new search which will (hopefully) yield new values according to random value selection
			DFS<CorrectColoringAnticlust> e(m, o);
    		delete m;
			CorrectColoringAnticlust* current_anticlustering = e.next();

			if (current_anticlustering != NULL) {
				std::vector<int> solution = current_anticlustering->getSolution();
				for (int j = 0; j < selection_index; ++j) {
					output(j, i) = solution[j];
				}
			} else {
				output(0,i) = -1;
			}
			delete current_anticlustering;
        }

        return output;
    }
}



// initial implementation of the dispersion maximization algorithm entirely in C++
// multiple solutions are similar to each other however computed very fast and guaranteed to not be the same solution
// [[Rcpp::export]]
Rcpp::NumericMatrix maximize_dispersion_gecode_cpp(int n, int k, int d_count, Rcpp::NumericVector d_sorted, Rcpp::NumericMatrix d_matrix, Rcpp::NumericVector target_groups_sizes, double number_of_threads, int maximum_number_of_solutions) {
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
    double current_distance;
    std::vector<int> vertex_index(data_count, -1);  // maps the true vertex-value (index in distance matrix) to the selection-index of the current selection
    int selection_index = 0;
    std::vector<std::vector<int>> solutions;
    CorrectColoring* previous_anticlustering;
    CorrectColoring* current_anticlustering;
    std::unique_ptr<DFS<CorrectColoring>> previous_search;
    int actual_number_of_solutions = 0;

    for (int i = 0; i < distance_count_unique; ++i) {
        current_distance = sorted_distances_unique[i];
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

        CorrectColoring* m = new CorrectColoring(
            selection_index, 
            anticluster_count, 
            edge_count, 
            edge_table_u, 
            edge_table_v, 
            Rcpp::as<std::vector<int>>(target_groups_sizes));
        Search::Options o;
        o.threads = number_of_threads;
        std::unique_ptr<DFS<CorrectColoring>> current_search = std::make_unique<DFS<CorrectColoring>>(m, o);
        delete m;
        current_anticlustering = current_search->next();

        if (current_anticlustering == NULL) {
            if (i != 0) {
                solutions.push_back(previous_anticlustering->getSolution());
                ++actual_number_of_solutions;
                for(int j = 1; j < maximum_number_of_solutions; ++j) {
                    if(CorrectColoring* another_solution = previous_search->next()) {
                        solutions.push_back(another_solution->getSolution());
                        ++actual_number_of_solutions;
                        delete another_solution;
                    } else {
                        break;
                    }
                }
            }
            break;
        } else if (i == distance_count_unique - 1 && current_anticlustering != NULL) {
            solutions.push_back(current_anticlustering->getSolution());
            ++actual_number_of_solutions;
            for(int j = 1; j < maximum_number_of_solutions; ++j) {
                if(CorrectColoring* another_solution = previous_search->next()) {
                    solutions.push_back(another_solution->getSolution());
                    ++actual_number_of_solutions;
                    delete another_solution;
                } else {
                    break;
                }
            }
            break;
        }
        previous_anticlustering = current_anticlustering;
        previous_search = std::move(current_search);
    }
    delete previous_anticlustering;
    delete current_anticlustering;

    Rcpp::NumericMatrix output(data_count, 3*actual_number_of_solutions);
    for (int i = 0; i < data_count; ++i) {
        for (int j = 0; j < actual_number_of_solutions; ++j) {
            if(vertex_index[i] == -1 or vertex_index[i] >= solutions[j].size()) {  // vertex tried for a solution or was tried but failed and is not included in solution
                output(i, (j*3)) = -1;
                output(i, (j*3)+1) = 1;
            } else {
                output(i, (j*3)) = solutions[j][vertex_index[i]];
                output(i, (j*3)+1) = 0;
            }
            output(i, (j*3)+2) = current_distance;
        }
    }
    return output;
}



// doesn't work well (often but not always segfaults lol)
// instead of trying to solve the problem every iteration, only certain iterations are considered in a manner similar to a binary search
// idea may be interesting though
Rcpp::NumericMatrix maximize_dispersion_binary_search(int n, int k, int d_count, Rcpp::NumericVector d_sorted, Rcpp::NumericMatrix d_matrix, Rcpp::NumericVector target_groups_sizes, double number_of_threads, int maximum_number_of_solutions) {
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
    double current_distance;
    int selection_index = 0;
    std::vector<std::vector<int>> solutions;
    CorrectColoring* previous_anticlustering;
    CorrectColoring* current_anticlustering;
    std::unique_ptr<DFS<CorrectColoring>> previous_search;
    int actual_number_of_solutions = 0;

    int distance_index = distance_count_unique/2;
    int distance_index_previous = -1;
    int lower_bound = 0;
    int upper_bound = distance_count_unique - 1;
    std::vector<int> successfull_at_distance_i(distance_count_unique, -1);
    std::vector<std::vector<int>> vertex_index(distance_count_unique, std::vector<int>(n, -1));
    std::vector<int> vertex_index_map_for_output(distance_count_unique, -1);
    

    while (lower_bound != upper_bound) {
        edge_count = 0;
        selection_index = 0;
        edge_table_u.clear();
        edge_table_v.clear();
        current_distance = sorted_distances_unique[distance_index];

        for (int u  = 0; u < data_count; ++u) {
            for (int v = u; v < data_count; ++v) {
                if (distance_matrix[u][v] <= current_distance && distance_matrix[u][v] > 0) {
                    if (vertex_index[distance_index][u] == -1) {
                        vertex_index[distance_index][u] = selection_index;
                        edge_table_u.push_back(selection_index);
                        ++selection_index;
                        if (vertex_index[distance_index][v] == -1) {
                            vertex_index[distance_index][v] = selection_index;
                            edge_table_v.push_back(selection_index);
                            ++selection_index;
                        } else {
                            edge_table_v.push_back(vertex_index[distance_index][v]);
                        }
                    } else if (vertex_index[distance_index][v] == -1) {
                        vertex_index[distance_index][v] = selection_index;
                        edge_table_u.push_back(vertex_index[distance_index][u]);
                        edge_table_v.push_back(selection_index);
                        ++selection_index;
                    } else {
                        edge_table_u.push_back(vertex_index[distance_index][u]);
                        edge_table_v.push_back(vertex_index[distance_index][v]);
                    }
                    edge_count++;
                }
            }
        }

        CorrectColoring* m = new CorrectColoring(
            selection_index, 
            anticluster_count, 
            edge_count, 
            edge_table_u, 
            edge_table_v, 
            Rcpp::as<std::vector<int>>(target_groups_sizes));
        Search::Options o;
        o.threads = number_of_threads;
        std::unique_ptr<DFS<CorrectColoring>> current_search = std::make_unique<DFS<CorrectColoring>>(m, o);
        delete m;
        current_anticlustering = current_search->next();

        if (current_anticlustering == NULL) {
            if(distance_index == 0) {
                break;
            }
            if(successfull_at_distance_i[distance_index - 1] == 1) {
                vertex_index_map_for_output = vertex_index[distance_index - 1];
                ++actual_number_of_solutions;
                for(int j = 1; j < maximum_number_of_solutions; ++j) {
                    if(CorrectColoring* another_solution = previous_search->next()) {
                        solutions.push_back(another_solution->getSolution());
                        ++actual_number_of_solutions;
                        delete another_solution;
                    } else {
                        break;
                    }
                }
                break;
            }
            successfull_at_distance_i[distance_index] = 0;
            distance_index_previous = distance_index;
            upper_bound = distance_index;
            distance_index = distance_index - ((distance_index - lower_bound + 1)/2);
        } else {
            if(successfull_at_distance_i[distance_index + 1] == 0 || distance_index == distance_count_unique - 1) {
                current_distance = sorted_distances_unique[distance_index + 1];
                vertex_index_map_for_output = vertex_index[distance_index];
                solutions.push_back(current_anticlustering->getSolution());
                ++actual_number_of_solutions;
                for(int j = 1; j < maximum_number_of_solutions; ++j) {
                    if(CorrectColoring* another_solution = current_search->next()) {
                        solutions.push_back(another_solution->getSolution());
                        ++actual_number_of_solutions;
                        delete another_solution;
                    } else {
                        break;
                    }
                }
                break;
            }
            successfull_at_distance_i[distance_index] = 1;
            distance_index_previous = distance_index;
            lower_bound = distance_index;
            distance_index = distance_index + ((upper_bound - distance_index + 1)/2);
            previous_anticlustering = current_anticlustering;
            previous_search = std::move(current_search);
        }
    }
    delete previous_anticlustering;
    delete current_anticlustering;


    Rcpp::NumericMatrix output(data_count, 3*actual_number_of_solutions);
    for (int i = 0; i < data_count; ++i) {
        for (int j = 0; j < actual_number_of_solutions; ++j) {
            if(vertex_index_map_for_output[i] == -1) {  // vertex tried for a solution or was tried but failed and is not included in solution
                output(i, (j*3)) = -1;
                output(i, (j*3)+1) = 1;
            } else {
                output(i, (j*3)) = solutions[j][vertex_index_map_for_output[i]];
                output(i, (j*3)+1) = 0;
            }
            output(i, (j*3)+2) = current_distance;
        }
    }
    return output;
}