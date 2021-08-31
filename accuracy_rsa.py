import numpy as np
from pprint import pprint
import os
import pickle
from scipy.stats import pearsonr, spearmanr

sim_matrix_save_path = 'npy/rsa_sim_mat.npy'
if os.path.exists(sim_matrix_save_path):
    sim_matrix = np.load(sim_matrix_save_path)
else:
    sim_matrix = np.zeros((50,50))
    for i in range(50): # i is target
        print(i)
        for j in range(50):
            proposed_mat = np.load('rsa/target_{}/source_{}/rdm.npy'.format(i,j))
            base_mat = np.load('rsa/target_{}/source_{}/rdm.npy'.format(i,i))
            proposed_vec = []
            base_vec = []
            mat_length = base_mat.shape[0]
            for mat_i in range(mat_length):
                for mat_j in range(mat_length):
                    if mat_i <= mat_j: continue
                    proposed_vec.append(proposed_mat[mat_i][mat_j])
                    base_vec.append(base_mat[mat_i][mat_j])
            sim_matrix[i][j] = spearmanr(proposed_vec, base_vec)[0]
    np.save(sim_matrix_save_path, sim_matrix)


np.fill_diagonal(sim_matrix, -1e6)
names = pickle.load(open('pickles/names.pickle', 'rb'))
errors = pickle.load(open('pickles/cub_inat2018_performance_dict.pickle', 'rb'))

rel_errors =  []
absolute_errors = []
selections = []
pearson_p_values = []
pearson_r_values = []
spearman_p_values = []
spearman_r_values = []
for name_i1, (name, row) in enumerate(zip(names, sim_matrix)):
    max_sim_i = np.argmax(row)
    match_name = names[max_sim_i]
    absolute_errors.append(errors[name][match_name])
    selections.append((name, match_name))
    rel_errors.append( (errors[name][match_name]- errors[name]['optimal'])/errors[name]['optimal'])

    sub_sim_arr = []
    sub_err_arr = []
    for name_i2, (sim, name2) in enumerate(zip(row, names)):
        if name_i1 == name_i2: continue
        sub_sim_arr.append(sim)
        sub_err_arr.append(errors[name][name2])
    pearson_rval, pearson_pval = pearsonr(sub_sim_arr, sub_err_arr)
    spearman_rval, spearman_pval = spearmanr(sub_sim_arr, sub_err_arr)
    pearson_p_values.append(pearson_pval)
    spearman_p_values.append(spearman_pval)
    pearson_r_values.append(pearson_rval)
    spearman_r_values.append(spearman_rval)

pprint(list(zip(selections, absolute_errors)))

print("Relative Errors:", np.average(rel_errors))
print("Average Pearson r: {}\nAverage Spearman r: {}".format(np.average(pearson_r_values), np.average(spearman_r_values)))
print("Average Pearson p: {}\nAverage Spearman p: {}".format(np.average(pearson_p_values), np.average(spearman_p_values)))



