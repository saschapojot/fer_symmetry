//
// Created by adada on 2/5/2025.
//

#include "mc_read_load_compute.hpp"


double mc_computation::S_uni()
{
    return 1.0;
}

double mc_computation::acceptanceRatio_uni(const arma::dvec& arma_vec_curr,
                                           const arma::dvec& arma_vec_next, const int& flattened_ind,
                                           const double& UCurr, const double& UNext)
{
    double numerator = -this->beta * UNext;
    double denominator = -this->beta * UCurr;
    double R = std::exp(numerator - denominator);
    double ratio = 1.0;
    if (std::fetestexcept(FE_DIVBYZERO))
    {
        std::cout << "Division by zero exception caught." << std::endl;
        std::exit(15);
    }
    if (std::isnan(ratio))
    {
        std::cout << "The result is NaN." << std::endl;
        std::exit(15);
    }
    R *= ratio;

    return std::min(1.0, R);
}

/// flip 1 spin
double mc_computation::generate_uni_one_point()
{
int ind_tmp=ind_int_0_1(e2);
    return this->s_vals[ind_tmp];

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
    // print_shared_ptr(s_init,N0*N1);
    this->execute_mc(s_init,newFlushNum);
    // arma::dvec s_arma_vec_tmp(s_init.get(),N0*N1);
    // double UCurr,UNext;
    // this->H_update_local(7,s_arma_vec_tmp,s_arma_vec_tmp,UCurr,UNext);
}

// neighbors of (0,0)
void mc_computation::construct_neighbors_1_point()
{
    this->neigbors={{-1,0},{1,0},{0,-1},{0,1}};
    //print neighbors
    std::cout<<"print neighbors:"<<std::endl;
    for (const auto & vec:neigbors)
    {
        print_vector(vec);
    }//end for
}

void mc_computation::load_pickle_data(const std::string& filename, std::shared_ptr<double[]>& data_ptr,
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

void mc_computation::save_array_to_pickle(const std::shared_ptr<double[]>& ptr, int size, const std::string& filename)
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

void mc_computation::init_s()
{
    std::string name;
    std::string s_inFileName;
    if (this->flushLastFile == -1)
    {
        name = "init";
        s_inFileName=this->out_s_path+"/s_"+name+".pkl";
        this->load_pickle_data(s_inFileName,s_init,N0*N1);
    }//end flushLastFile==-1
    else
    {
        name="flushEnd"+std::to_string(this->flushLastFile);
        s_inFileName=this->out_s_path+"/"+name+".s.pkl";
        //load s
        this->load_pickle_data(s_inFileName,s_all_ptr,sweepToWrite * N0 * N1);
        //copy last N0*N1 elements of to s_init
        std::memcpy(this->s_init.get(),s_all_ptr.get()+N0*N1*(sweepToWrite-1),
            N0*N1*sizeof(double));
    }//end else

}

int mc_computation::mod_direction0(const int&m0)
{

    return ((m0 % N0) + N0) % N0;

}

int mc_computation::mod_direction1(const int&m1)
{return ((m1 % N1) + N1) % N1;
}

void mc_computation::init_flattened_ind_and_neighbors()
{
    //this function initializes each point's neigboring indices(flattened),

    this->flattened_ind_neighbors=std::vector<std::vector<int>>(N0*N1,std::vector<int>());
    for (int n0=0;n0<N0;n0++)
    {
        for (int n1=0;n1<N1;n1++)
        {
            // std::cout<<"======================="<<std::endl;
            int point_curr_flattened=this->double_ind_to_flat_ind(n0,n1);
            for (const auto&vec_nghbrs:this->neigbors)
            {
                int diff_direc0=vec_nghbrs[0];
                int diff_direc1=vec_nghbrs[1];
                int m0=n0+diff_direc0;
                int m1=n1+diff_direc1;
                int m0_mod=mod_direction0(m0);
                int m1_mod=mod_direction1(m1);
                int flattened_ngb=double_ind_to_flat_ind(m0_mod,m1_mod);
                flattened_ind_neighbors[point_curr_flattened].push_back(flattened_ngb);


                // std::cout<<"point_curr_flattened="<<point_curr_flattened
                // <<", flattened_ngb="<<flattened_ngb<<std::endl;
                // std::cout<<"..."<<std::endl;
            }//end neighbors
            // std::cout<<"point_curr_flattened="<<point_curr_flattened<<std::endl;
            // print_vector(flattened_ind_neighbors[point_curr_flattened]);
        }//end for n1
    }//end for n0
}



///
/// @param flattened_ind_center (flattened) index of spin to be updated
/// @param ind_neighbor index of spin around the center dipole (0..3)
/// @param s_arma_vec flattened s array
/// @return interaction energy of flattened_ind_center and ind_neighbor
double mc_computation::H_interaction_local(const int& flattened_ind_center,
    const int& ind_neighbor, const arma::dvec& s_arma_vec)
{
    // std::cout<<"flattened_ind_center="<<flattened_ind_center<<std::endl;
    // std::cout<<s_vec.n_elem<<std::endl;
    // s_vec.print("s_vec");


    double s_center_val_tmp=s_arma_vec(flattened_ind_center);

    // std::cout<<"s_center_val_tmp="<<s_center_val_tmp<<std::endl;


    int flattened_ind_one_neighbor=this->flattened_ind_neighbors[flattened_ind_center][ind_neighbor];


    // std::cout<<"flattened_ind_center="<<flattened_ind_center
    // <<", ind_neighbor="<<ind_neighbor
    // <<", flattened_ind_one_neighbor="<<flattened_ind_one_neighbor<<std::endl;

    double s_neighbor_val_tmp=s_arma_vec(flattened_ind_one_neighbor);

    // std::cout<<"s_neighbor_val_tmp="<<s_neighbor_val_tmp<<std::endl;

    double val=J*s_center_val_tmp*s_neighbor_val_tmp;

    return val;
}

///
/// @param s_arma_vec flattened s array
/// @return total interaction energy
double mc_computation::H_tot(const arma::dvec& s_arma_vec)
{
    double val=0;
    for (int n0=0;n0<N0;n0++)
    {
        for (int n1=0;n1<N1;n1++)
        {
            int flattened_ind=double_ind_to_flat_ind(n0,n1);
            const auto& vec_ind_neighbor_tmp=flattened_ind_neighbors[flattened_ind];

            // std::cout<<"flattened_ind="<<flattened_ind<<", vec_ind_neighbor_tmp:"<<std::endl;
            // print_vector(vec_ind_neighbor_tmp);
            // std::cout<<"vec_ind_neighbor_tmp.size()="<<vec_ind_neighbor_tmp.size()<<std::endl;


            for (int j=0;j<vec_ind_neighbor_tmp.size();j++)
            {
                val+=H_interaction_local(flattened_ind,j,s_arma_vec);
            }//end j

        }//end n0
    }//end n0
    // std::cout<<"val="<<val<<std::endl;
    return val*0.5;
}

///
/// @param flattened_ind flattened index of the element of s array to update
/// @param s_arma_vec_curr
/// @param s_arma_vec_next
/// @param UCurr
/// @param UNext
void  mc_computation::H_update_local(const int &flattened_ind,
                    const arma::dvec & s_arma_vec_curr,
                    const arma::dvec & s_arma_vec_next,
                    double& UCurr, double& UNext)
{
  double E_int_curr=0;

    double E_int_next=0;


    int neighbor_num=neigbors.size();

    // std::cout<<"neighbor_num="<<neighbor_num<<std::endl;

    //E_int_curr
    for (int j=0;j<neighbor_num;j++)
    {
        E_int_curr+=H_interaction_local(flattened_ind,j,s_arma_vec_curr);
    }//end j

    UCurr=E_int_curr;

    //E_int_next
    for (int j=0;j<neighbor_num;j++)
    {
        E_int_next+=H_interaction_local(flattened_ind,j,s_arma_vec_next);
    }//end j


    UNext=E_int_next;

    // std::cout<<"UCurr="<<UCurr<<", UNext="<<UNext<<std::endl;
}
void mc_computation::proposal_uni(const arma::dvec& arma_vec_curr, arma::dvec& arma_vec_next,
                                  const int& flattened_ind)
{

    double s_new=this->generate_uni_one_point();
    // std::cout<<"s_new="<<s_new<<std::endl;
    arma_vec_next = arma_vec_curr;
    arma_vec_next(flattened_ind) =s_new;
}

void mc_computation::execute_mc_one_sweep(arma::dvec& s_arma_vec_curr,
       arma::dvec& s_arma_vec_next,double& U_base_value)
{
    double UCurr=0;
    double UNext = 0;
    U_base_value=this->H_tot(s_arma_vec_curr);

    //update s
    for (int i=0;i<N0*N1;i++)
    {
        int flattened_ind = unif_in_0_N0N1(e2);
        this->proposal_uni(s_arma_vec_curr,s_arma_vec_next,flattened_ind);
        this->H_update_local(flattened_ind,s_arma_vec_curr,s_arma_vec_next,UCurr,UNext);

        double r=this->acceptanceRatio_uni(s_arma_vec_curr,s_arma_vec_next,flattened_ind,
           UCurr,UNext );
        double u = distUnif01(e2);

        // std::cout<<"r="<<r<<", u="<<u<<std::endl;
        // std::cout<<"UCurr="<<UCurr<<", UNext="<<UNext<<std::endl;
        if (u <= r)
        {
            s_arma_vec_curr=s_arma_vec_next;
            U_base_value+=UNext-UCurr;

        }//end of accept-reject

    }//end updating s

}

void mc_computation::execute_mc(const std::shared_ptr<double[]>& s_vec_init,const int& flushNum)
{
    arma::dvec s_arma_vec_curr(s_vec_init.get(),N0*N1);
    arma::dvec s_arma_vec_next(N0*N1,arma::fill::zeros);
    double U_base_value=-12345;

    int flushThisFileStart=this->flushLastFile+1;

    for (int fls = 0; fls < flushNum; fls++)
    {
        const auto tMCStart{std::chrono::steady_clock::now()};
        for (int swp = 0; swp < sweepToWrite*sweep_multiple; swp++)
        {
            this->execute_mc_one_sweep(s_arma_vec_curr,s_arma_vec_next,U_base_value);
            if(swp%sweep_multiple==0)
            {
                int swp_out=swp/sweep_multiple;
                this->U_data_all_ptr[swp_out]=U_base_value;
                std::memcpy(s_all_ptr.get()+swp_out*N0*N1,s_arma_vec_curr.memptr(),N0*N1*sizeof(double));
            }//end save to array
        }//end sweep for
        int flushEnd=flushThisFileStart+fls;
        std::string fileNameMiddle =  "flushEnd" + std::to_string(flushEnd);
        std::string out_U_PickleFileName = out_U_path+"/" + fileNameMiddle + ".U.pkl";

        std::string out_s_PickleFileName=out_s_path+"/"+fileNameMiddle+".s.pkl";
        //save U
        this->save_array_to_pickle(U_data_all_ptr,sweepToWrite,out_U_PickleFileName);

        //save s
        this->save_array_to_pickle(s_all_ptr,sweepToWrite*N0*N1,out_s_PickleFileName);
        const auto tMCEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
        std::cout << "flush " + std::to_string(flushEnd)  + ": "
                  << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    }//end flush for loop
    std::cout << "mc executed for " << flushNum << " flushes." << std::endl;
}