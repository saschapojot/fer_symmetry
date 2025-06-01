//
// Created by adada on 2/5/2025.
//

#include "mc_read_load_compute.hpp"

void mc_computation::execute_mc(std::shared_ptr<const double[]> s_vec_init,const int& flushNum)
{
    double U_base_value=-12345;
    int flushThisFileStart=this->flushLastFile+1;
    for (int fls = 0; fls < flushNum; fls++)
    {
        const auto tMCStart{std::chrono::steady_clock::now()};
        for (int swp = 0; swp < sweepToWrite*sweep_multiple; swp++)
        {
            this->update_spins_parallel_1_sweep(U_base_value);
            if(swp%sweep_multiple==0)
            {
                int swp_out=swp/sweep_multiple;
                this->U_data_all_ptr[swp_out]=U_base_value;
                std::memcpy(s_all_ptr.get()+swp_out*N0*N1,s_init.get(),N0*N1*sizeof(double));
            }//end save to array
        }//end sweep for
        int flushEnd=flushThisFileStart+fls;
        std::string fileNameMiddle =  "flushEnd" + std::to_string(flushEnd);
        std::string out_U_PickleFileName = out_U_path+"/" + fileNameMiddle + ".U.pkl";

        // std::string out_s_PickleFileName=out_s_path+"/"+fileNameMiddle+".s.pkl";
        //save U
        this->save_array_to_pickle(U_data_all_ptr,sweepToWrite,out_U_PickleFileName);
        //save s
        // this->save_array_to_pickle(s_all_ptr,sweepToWrite*N0*N1,out_s_PickleFileName);

        //compute M
        this->compute_all_magnetizations_parallel();
        std::string out_M_PickleFileName=this->out_M_path+"/" + fileNameMiddle + ".M.pkl";
        //save M
        this->save_array_to_pickle(M_all_ptr,sweepToWrite,out_M_PickleFileName);

        const auto tMCEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
        std::cout << "flush " + std::to_string(flushEnd)  + ": "
                  << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;

    }//end flush for loop

}
void mc_computation::update_spins_parallel_1_sweep(double& U_base_value)
{
    // const int num_threads = this->num_parallel;
    int actual_threads_red = std::min(this->num_parallel, static_cast<int>(flattened_red_points.size()));

    std::vector<std::thread> threads;

    // Update red points (parallel)
    int chunk_size = flattened_red_points.size() / actual_threads_red;
    for (int t = 0; t < actual_threads_red; ++t) {
        int start = t * chunk_size;
        int end = (t == actual_threads_red - 1) ? flattened_red_points.size() : start + chunk_size;

        threads.emplace_back([this, start, end, t] {
            // Set thread affinity at the beginning of each thread
            // cpu_set_t cpuset;
            // CPU_ZERO(&cpuset);
            // // Map thread to specific core, distributing across NUMA nodes
            // int core_id = t % 64;
            // CPU_SET(core_id, &cpuset);
            // pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            for (int i = start; i < end; ++i) {
                int idx = flattened_red_points[i];
                double dE = delta_energy(idx);
                if (dE <= 0 || get_thread_random(t) <= std::exp(-dE * beta)) {
                    s_init[idx] *= -1;  // Flip spin (no energy update)
                    // std::cout<<"s_init["<<idx<<"] updated."<<std::endl;
                }
            }
        });
    }//end for t
    for (auto& th : threads) th.join();
    threads.clear();
    //====================================================
    // Update black points (parallel)
    int actual_threads_black = std::min(this->num_parallel, static_cast<int>(flattened_black_points.size()));

    chunk_size = flattened_black_points.size() / actual_threads_black;
    for (int t = 0; t < actual_threads_black; ++t) {
        int start = t * chunk_size;
        int end = (t == actual_threads_black - 1) ? flattened_black_points.size() : start + chunk_size;

        threads.emplace_back([this, start, end, t] {
            // Set thread affinity at the beginning of each thread
                       // cpu_set_t cpuset;
                       // CPU_ZERO(&cpuset);
                       // // Map thread to specific core, distributing across NUMA nodes
                       // int core_id = t % 64;
                       // CPU_SET(core_id, &cpuset);
                       // pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            for (int i = start; i < end; ++i) {
                int idx = flattened_black_points[i];
                double dE = delta_energy(idx);
                if (dE <= 0 || get_thread_random(t) <= std::exp(-dE * beta)) {
                    s_init[idx] *= -1;  // Flip spin (no energy update)
                    // std::cout<<"s_init["<<idx<<"] updated."<<std::endl;
                }
            }
        });
    }//end for t
    for (auto& th : threads) th.join();



    U_base_value=this->H_tot(s_init);
    // std::cout<<"U_base_value="<<U_base_value<<std::endl;
}


///
/// @param n0 index along direction 0
/// @param n1 index along direction 1
/// @return energy changed if spin [n0,n1] is flipped
 double mc_computation::delta_energy(int flattened_ind)
{
    // int flattened_ind = double_ind_to_flat_ind(n0, n1);
    const auto& vec_ind_neighbor_tmp = flattened_ind_neighbors[flattened_ind];
    // std::cout<<"flattened_ind="<<flattened_ind<<", vec_ind_neighbor_tmp:"<<std::endl;
    // print_vector(vec_ind_neighbor_tmp);
    // std::cout<<"vec_ind_neighbor_tmp.size()="<<vec_ind_neighbor_tmp.size()<<std::endl;
    double sum_neigbors=0;
    //j=0,1,2,3
    for (int j = 0; j < vec_ind_neighbor_tmp.size(); j++)
    {
        int ind_tmp = vec_ind_neighbor_tmp[j];
        sum_neigbors += s_init[ind_tmp];
    } //end for j

    return -2.0*J*sum_neigbors*s_init[flattened_ind];
}

///
/// @param s_arma_vec flattened s array
/// @return total interaction energy
double mc_computation::H_tot(std::shared_ptr<const double[]> s_vec)
{
    double val = 0;
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            int flattened_ind = double_ind_to_flat_ind(n0, n1);
            const auto& vec_ind_neighbor_tmp = flattened_ind_neighbors[flattened_ind];

            // std::cout<<"flattened_ind="<<flattened_ind<<", vec_ind_neighbor_tmp:"<<std::endl;
            // print_vector(vec_ind_neighbor_tmp);
            // std::cout<<"vec_ind_neighbor_tmp.size()="<<vec_ind_neighbor_tmp.size()<<std::endl;


            for (int j = 0; j < vec_ind_neighbor_tmp.size(); j++)
            {
                val += H_interaction_local(flattened_ind, j, s_vec);
            } //end j
        } //end n0
    } //end n0
    // std::cout<<"val="<<val<<std::endl;
    return val * 0.5;
}


///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center dipole (0..3)
/// @param s_arma_vec flattened s array
/// @return interaction energy of flattened_ind_center and ind_neighbor
double mc_computation::H_interaction_local(const int& flattened_ind_center, const int& ind_neighbor,
                                           std::shared_ptr<const double[]> s_vec)
{
    // std::cout<<"flattened_ind_center="<<flattened_ind_center<<std::endl;
    // std::cout<<s_vec.n_elem<<std::endl;
    // s_vec.print("s_vec");


    // double s_center_val_tmp=s_arma_vec(flattened_ind_center);
    double s_center_val_tmp = s_vec[flattened_ind_center];
    // std::cout<<"s_center_val_tmp="<<s_center_val_tmp<<std::endl;


    int flattened_ind_one_neighbor = this->flattened_ind_neighbors[flattened_ind_center][ind_neighbor];


    // std::cout<<"flattened_ind_center="<<flattened_ind_center
    // <<", ind_neighbor="<<ind_neighbor
    // <<", flattened_ind_one_neighbor="<<flattened_ind_one_neighbor<<std::endl;

    double s_neighbor_val_tmp = s_vec[flattened_ind_one_neighbor]; //s_arma_vec(flattened_ind_one_neighbor);

    // std::cout<<"s_neighbor_val_tmp="<<s_neighbor_val_tmp<<std::endl;

    double val = J * s_center_val_tmp * s_neighbor_val_tmp;

    return val;
}

///
/// @param n0
/// @param n1
/// @return flatenned index
int mc_computation::double_ind_to_flat_ind(const int& n0, const int& n1)
{
    return n0 * N1 + n1;
}

void mc_computation::init_and_run()
{
    this->init_s();
    this->construct_neighbors_1_point();
    this->init_flattened_ind_and_neighbors();
    this->init_red_and_black_points();
    this->execute_mc(s_init,newFlushNum);


    // for (int j=0;j<sweepToWrite*N0*N1;j++)
    // {
    //     this->s_all_ptr[j]=j;
    // }
    // this->compute_all_magnetizations_parallel();
    // std::string outM_file=this->out_M_path+"/M_init.pkl";
    // this->save_array_to_pickle(this->M_all_ptr,sweepToWrite,outM_file);
    // std::cout<<"M_all_ptr[0]="<<M_all_ptr[0]<<std::endl;
}

void mc_computation::init_flattened_ind_and_neighbors()
{
    //this function initializes each point's neigboring indices(flattened),

    this->flattened_ind_neighbors = std::vector<std::vector<int>>(N0 * N1, std::vector<int>());
    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            // std::cout << "=======================" << std::endl;
            int point_curr_flattened = this->double_ind_to_flat_ind(n0, n1);
            for (const auto& vec_nghbrs : this->neigbors)
            {
                int diff_direc0 = vec_nghbrs[0];
                int diff_direc1 = vec_nghbrs[1];
                int m0 = n0 + diff_direc0;
                int m1 = n1 + diff_direc1;
                int m0_mod = mod_direction0(m0);
                int m1_mod = mod_direction1(m1);
                int flattened_ngb = double_ind_to_flat_ind(m0_mod, m1_mod);
                flattened_ind_neighbors[point_curr_flattened].push_back(flattened_ngb);


                // std::cout << "point_curr_flattened=" << point_curr_flattened
                    // << ", flattened_ngb=" << flattened_ngb << std::endl;
                // std::cout << "..." << std::endl;
            } //end neighbors
            // std::cout << "point_curr_flattened=" << point_curr_flattened << std::endl;
            // print_vector(flattened_ind_neighbors[point_curr_flattened]);
        } //end for n1
    } //end for n0
}

void mc_computation::init_red_and_black_points()
{
    this->flattened_red_points.reserve(N0 * N1 / 2);
    this->flattened_black_points.reserve(N0 * N1 / 2);

    for (int n0 = 0; n0 < N0; n0++)
    {
        for (int n1 = 0; n1 < N1; n1++)
        {
            if ((n0 + n1) % 2 == 0)
            {
                int flat_ind = this->double_ind_to_flat_ind(n0, n1);
                flattened_red_points.push_back(flat_ind);
            } //end if red
            else
            {
                int flat_ind = this->double_ind_to_flat_ind(n0, n1);
                flattened_black_points.push_back(flat_ind);
            }
        } //end for n1
    } //end for n0
    //
    // std::cout << "red points:\n";
    // this->print_vector(flattened_red_points);
    //
    // std::cout << "---------------\n";
    // std::cout << "black points:\n";
    // this->print_vector(flattened_black_points);
}

// neighbors of (0,0)
void mc_computation::construct_neighbors_1_point()
{
    this->neigbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    //print neighbors
    std::cout << "print neighbors:" << std::endl;
    for (const auto& vec : neigbors)
    {
        print_vector(vec);
    } //end for
}

void mc_computation::init_s()
{
    std::string name;
    std::string s_inFileName;
    if (this->flushLastFile == -1)
    {
        name = "init";
        s_inFileName = this->out_s_path + "/s_" + name + ".pkl";
        this->load_pickle_data(s_inFileName, s_init, N0 * N1);
    } //end flushLastFile==-1
    else
    {
        name = "flushEnd" + std::to_string(this->flushLastFile);
        s_inFileName = this->out_s_path + "/" + name + ".s.pkl";
        //load s
        this->load_pickle_data(s_inFileName, s_all_ptr, sweepToWrite * N0 * N1);
        //copy last N0*N1 elements of to s_init
        std::memcpy(this->s_init.get(), s_all_ptr.get() + N0 * N1 * (sweepToWrite - 1),
                    N0 * N1 * sizeof(double));
    } //end else
}


int mc_computation::mod_direction0(const int& m0)
{
    return ((m0 % N0) + N0) % N0;
}

int mc_computation::mod_direction1(const int& m1)
{
    return ((m1 % N1) + N1) % N1;
}


void mc_computation::load_pickle_data(const std::string& filename, std::shared_ptr<double[]> data_ptr,
                                      std::size_t size)
{
    // Initialize Python and NumPy
    Py_Initialize();
    np::initialize();


    try
    {
        // Use Python's 'io' module to open the file directly in binary mode
        py::object io_module = py::import("io");
        py::object file = io_module.attr("open")(filename, "rb"); // Open file in binary mode

        // Import the 'pickle' module
        py::object pickle_module = py::import("pickle");

        // Use pickle.load to deserialize from the Python file object
        py::object loaded_data = pickle_module.attr("load")(file);

        // Close the file
        file.attr("close")();

        // Check if the loaded object is a NumPy array
        if (py::extract<np::ndarray>(loaded_data).check())
        {
            np::ndarray np_array = py::extract<np::ndarray>(loaded_data);

            // Convert the NumPy array to a Python list using tolist()
            py::object py_list = np_array.attr("tolist")();

            // Ensure the list size matches the expected size
            ssize_t list_size = py::len(py_list);
            if (static_cast<std::size_t>(list_size) > size)
            {
                throw std::runtime_error("The provided shared_ptr array size is smaller than the list size.");
            }

            // Copy the data from the Python list to the shared_ptr array
            for (ssize_t i = 0; i < list_size; ++i)
            {
                data_ptr[i] = py::extract<double>(py_list[i]);
            }
        }
        else
        {
            throw std::runtime_error("Loaded data is not a NumPy array.");
        }
    }
    catch (py::error_already_set&)
    {
        PyErr_Print();
        throw std::runtime_error("Python error occurred.");
    }
}

void mc_computation::save_array_to_pickle(std::shared_ptr<const double[]> ptr, int size, const std::string& filename)
{
    using namespace boost::python;
    namespace np = boost::python::numpy;

    // Initialize Python interpreter if not already initialized
    if (!Py_IsInitialized())
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }
        np::initialize(); // Initialize NumPy
    }

    try
    {
        // Import the pickle module
        object pickle = import("pickle");
        object pickle_dumps = pickle.attr("dumps");

        // Convert C++ array to NumPy array using shared_ptr
        np::ndarray numpy_array = np::from_data(
            ptr.get(), // Use shared_ptr's raw pointer
            np::dtype::get_builtin<double>(), // NumPy data type (double)
            boost::python::make_tuple(size), // Shape of the array (1D array)
            boost::python::make_tuple(sizeof(double)), // Strides
            object() // Optional base object
        );

        // Serialize the NumPy array using pickle.dumps
        object serialized_array = pickle_dumps(numpy_array);

        // Extract the serialized data as a string
        std::string serialized_str = extract<std::string>(serialized_array);

        // Write the serialized data to a file
        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        file.write(serialized_str.data(), serialized_str.size());
        file.close();

        // Debug output (optional)
        // std::cout << "Array serialized and written to file successfully." << std::endl;
    }
    catch (const error_already_set&)
    {
        PyErr_Print();
        std::cerr << "Boost.Python error occurred." << std::endl;
    } catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}



///
/// @param startInd starting index of 1 configuration
/// @param length N*N
/// @param s_all_ptr containing each s_{ij} for each configuration
/// @return average of s_{ij} from index startInd to index startInd+length-1
double mc_computation::compute_M_avg_over_sites(const int &startInd, const int & length)
{
double avg=0.0;
    for (int j=startInd;j<startInd+length;j++)
    {
        avg+=this->s_all_ptr[j];
    }
    return avg/static_cast<double>(length);
}

///
///compute M in parallel for all configurations
void mc_computation::compute_all_magnetizations_parallel()
{
     int num_threads = num_parallel;
    int config_size=N0*N1;
   int  num_configs=sweepToWrite;
    // Ensure we don't create more threads than configurations
    num_threads = std::min(num_threads, static_cast< int>(num_configs));

    // Calculate how many configurations each thread will process
    int configs_per_thread = num_configs / num_threads;
    int remaining_configs = num_configs % num_threads;
    // Vector to store threads
    std::vector<std::thread> threads;

    // Create and launch threads
    int start_config = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        // Calculate range for this thread
        int configs_for_this_thread = configs_per_thread + (t < remaining_configs ? 1 : 0);
        int end_config = start_config + configs_for_this_thread;

        // Launch thread
        threads.emplace_back([this, start_config, end_config, config_size]() {
            for (int i = start_config; i < end_config; ++i) {
                int startInd = i * config_size;
                // Calculate magnetization and store directly in M_all_ptr
                this->M_all_ptr[i] = this->compute_M_avg_over_sites(startInd, config_size);
            }
        });

        start_config = end_config;
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

}