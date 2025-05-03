# 2d Ising, to compute  Critical exponents

python mk_dir.py, to set coefficients, T, and directories

##########################################
To manually perform each step of computations for s
1. python launch_one_run_dipole.py ./path/to/mc.conf
2. make run_mc
3. ./run_mc ./path/to/cppIn.txt


#########################################
To run 1 pass of mc without checking statistics of dipole
1. cmake .
2. make run_mc
3. python exec_noChecking_s.py T N startingFileIndSuggest init_path row
4. After completing computing s, generate s values by
   python  data2csv/pkl_s_data2csv.py N T init_path row
5. After completing computing s, generate U by:
   python data2csv/pkl_U_data2csv.py N T init_path row


##############################
plot U and C
in plt/
the plots iterate different T for the same N, the same init_path
1. convert csv file of U to average value, for all T
   python compute_U_avg.py N init_path row
2. plot U for all T
   python load_csv_plt_U.py  N init_path  row
3. compute C for all T
   python compute_C.py N init_path row
4. plot C for all T
   python load_csv_plt_C.py N init_path row

##############################

plot s
in plt/
1. compute average value of s (magnetization) for all T
   python compute_s_avg.py N init_path row
2. plot magnitude of magnetization
   python load_csv_plt_abs_magnetization.py N init_path row


##############################
plot auto-correlation for abs magnetization
1. compute auto-correlation of abs M for all T
   python compute_corr_abs_magnetization.py N init_path row
2. plot auto-correlation of abs M for all T
   python load_csv_plt_abs_magnetization_corr.py N init_path row