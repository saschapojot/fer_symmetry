//
// Created by adada on 2/5/2025.
//

#ifndef MC_READ_LOAD_COMPUTE_HPP
#define MC_READ_LOAD_COMPUTE_HPP
// #include <armadillo>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cfenv> // for floating-point exceptions
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = boost::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;

constexpr double PI = M_PI;
class mc_computation
{
public: mc_computation(const std::string& cppInParamsFileName)
{


    std::ifstream file(cppInParamsFileName);
    if (!file.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        std::exit(20);
    }
    std::string line;
    int paramCounter = 0;
    while (std::getline(file, line))
    {
        // Check if the line is empty
        if (line.empty())
        {
            continue; // Skip empty lines
        }
        std::istringstream iss(line);
        //read T
        if (paramCounter == 0)
        {
            iss >> T;
            if (T <= 0)
            {
                std::cerr << "T must be >0" << std::endl;
                std::exit(1);
            } //end if
            std::cout << "T=" << T << std::endl;
            this->beta = 1.0 / T;
            std::cout << "beta=" << beta << std::endl;
            paramCounter++;
            continue;
        }//end T
        //read J
        if (paramCounter == 1)
        {
            iss >> J;
            std::cout << "J=" << J << std::endl;
            paramCounter++;
            continue;
        }//end J
        //read N
        if (paramCounter == 2)
        {
            iss >> N0;
            N1 = N0;
            if (N0 <= 0)
            {
                std::cerr << "N must be >0" << std::endl;
                std::exit(1);
            }
            std::cout << "N0=N1=" << N0 << std::endl;
            paramCounter++;
            continue;
        }//end N

        //read sweepToWrite
        if (paramCounter==3)
        {
            iss >> sweepToWrite;
            if (sweepToWrite <= 0)
            {
                std::cerr << "sweepToWrite must be >0" << std::endl;
                std::exit(1);
            }
            std::cout << "sweepToWrite=" << sweepToWrite << std::endl;
            paramCounter++;
            continue;
        }//end sweepToWrite
        //read newFlushNum
        if (paramCounter == 4)
        {
            iss >> newFlushNum;
            if (newFlushNum <= 0)
            {
                std::cerr << "newFlushNum must be >0" << std::endl;
                std::exit(1);
            }
            std::cout << "newFlushNum=" << newFlushNum << std::endl;
            paramCounter++;
            continue;
        }//end newFlushNum
        //read flushLastFile
        if (paramCounter == 5)
        {
            iss >> flushLastFile;
            std::cout << "flushLastFile=" << flushLastFile << std::endl;
            paramCounter++;
            continue;
        }//end flushLastFile

        //read TDirRoot
        if (paramCounter == 6)
        {
            iss >> TDirRoot;
            std::cout << "TDirRoot=" << TDirRoot << std::endl;
            paramCounter++;
            continue;

        } //end TDirRoot
        //read U_s_dataDir
        if (paramCounter == 7)
        { iss >> U_s_dataDir;
            std::cout << "U_s_dataDir=" << U_s_dataDir << std::endl;
            paramCounter++;
            continue;

        } //end U_s_dataDir
        //read sweep_multiple
        if (paramCounter == 8)
        {
            iss >> sweep_multiple;
            if (sweep_multiple <= 0)
            {
                std::cerr << "sweep_multiple must be >0" << std::endl;
                std::exit(1);
            }
            std::cout << "sweep_multiple=" << sweep_multiple << std::endl;
            paramCounter++;
            continue;
        }//end sweep_multiple
        //read init_path
        if (paramCounter == 9)
        {
            iss>>init_path;
            std::cout<<"init_path="<<init_path<<std::endl;
            paramCounter++;
            continue;
        }//end init_path

        //read num_parallel
        if (paramCounter == 10)
        {
            iss>>this->num_parallel;
            std::cout<<"num_parallel="<<num_parallel<<std::endl;
            paramCounter++;
            continue;
        }
    }//end while
    //allocate memory for data
    try
    {
        this->U_data_all_ptr = std::shared_ptr<double[]>(new double[sweepToWrite],
                                                            std::default_delete<double[]>());
        this->s_all_ptr = std::shared_ptr<double[]>(new double[sweepToWrite * N0 * N1],
                                                         std::default_delete<double[]>());

        this->s_init=std::shared_ptr<double[]>(new double[N0 * N1],
                                                      std::default_delete<double[]>());

    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
        std::exit(2);
    } catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        std::exit(2);
    }
    this->out_U_path = this->U_s_dataDir + "/U/";
    if (!fs::is_directory(out_U_path) || !fs::exists(out_U_path))
    {
        fs::create_directories(out_U_path);
    }
    this->out_s_path = this->U_s_dataDir + "/s/";
    if (!fs::is_directory(out_s_path) || !fs::exists(out_s_path))
    {
        fs::create_directories(out_s_path);
    }

    // this->unif_in_0_N0N1 = std::uniform_int_distribution<int>(0, N0 * N1-1);
    // std::cout<<"PI="<<PI<<std::endl;
    //
    // this->s_vals={-1.0,1.0};
    // std::cout<<"s_vals:"<<std::endl;
    // print_vector(s_vals);


}//end constructor

public:
    void init_and_run();
    void execute_mc(std::shared_ptr<const double[]> s_vec_init,const int& flushNum);


    void update_spins_parallel_1_sweep(double& U_base_value);

    // Thread-local RNGs
    static double get_thread_random(int thread_id) {
        static thread_local std::mt19937 generator(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(generator);
    }
    ///
    /// @param n0 index along direction 0
    /// @param n1 index along direction 1
    /// @return energy changed if spin [n0,n1] is flipped
    inline double delta_energy(int flattened_ind);
    ///
    /// @param s_vec flattened s array
    /// @return total interaction energy
    double H_tot(std::shared_ptr<const double[]> s_vec);

    ///
    /// @param flattened_ind_center (flattened) index of spin to be updated
    /// @param ind_neighbor index of spin around the center dipole (0..3)
    /// @param s_arma_vec flattened s array
    /// @return interaction energy of flattened_ind_center and ind_neighbor
    double H_interaction_local(const int& flattened_ind_center,const int& ind_neighbor, std::shared_ptr<const double[]> s_vec);
    void init_red_and_black_points();
    void init_flattened_ind_and_neighbors();//this function initializes each point's neigboring indices(flattened), neigboring vectors, distance^2, distance^4

    void construct_neighbors_1_point();
    void init_s();
    ///
    /// @param n0
    /// @param n1
    /// @return flatenned index
    int double_ind_to_flat_ind(const int& n0, const int& n1);
    int mod_direction0(const int&m0);
    int mod_direction1(const int&m1);
    void save_array_to_pickle(const std::shared_ptr<double[]>& ptr, int size, const std::string& filename);

    void load_pickle_data(const std::string& filename, std::shared_ptr<double[]>& data_ptr, std::size_t size);

    template <class T>
    void print_shared_ptr(const std::shared_ptr<T>& ptr, const int& size)
    {
        if (!ptr)
        {
            std::cout << "Pointer is null." << std::endl;
            return;
        }

        for (int i = 0; i < size; i++)
        {
            if (i < size - 1)
            {
                std::cout << ptr[i] << ",";
            }
            else
            {
                std::cout << ptr[i] << std::endl;
            }
        }
    } //end print_shared_ptr

    // Template function to print the contents of a std::vector<T>
    template <typename T>
    void print_vector(const std::vector<T>& vec)
    {
        // Check if the vector is empty
        if (vec.empty())
        {
            std::cout << "Vector is empty." << std::endl;
            return;
        }

        // Print each element with a comma between them
        for (size_t i = 0; i < vec.size(); ++i)
        {
            // Print a comma before all elements except the first one
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << vec[i];
        }
        std::cout << std::endl;
    }
public:


    double T; // temperature
    double beta;
    int init_path;
    double J;
    int N0;
    int N1;
    int num_parallel;
    int sweepToWrite;
    int newFlushNum;
    int flushLastFile;
    std::string TDirRoot;
    std::string U_s_dataDir;
   // thread_local  std::ranlux24_base e2;

   // thread_local  std::uniform_int_distribution<int> ind_int_0_1;
    int sweep_multiple;
    std::string out_U_path;
    std::string out_s_path;
    //data in 1 flush
    std::shared_ptr<double[]> U_data_all_ptr; //all U data
    std::shared_ptr<double[]> s_all_ptr; //all Px data
    // std::vector<double>s_vals;

    //initial value
    std::shared_ptr<double[]> s_init;
    // std::uniform_int_distribution<int> unif_in_0_N0N1;
    // std::uniform_real_distribution<> distUnif01;
    std::vector<int> flattened_red_points;// n0+n1 is even
    std::vector<int> flattened_black_points;// n0+n1 is odd
    std::vector<std::vector<int>> neigbors;//around (0,0)
    std::vector<std::vector<int>> flattened_ind_neighbors;// a point (flattened index) and its neighbors(also flattened ind)

};





#endif //MC_READ_LOAD_COMPUTE_HPP
