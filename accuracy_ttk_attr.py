import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr

sim_matrix = np.load('npy/ttk_inat_cub_attr.npy')
np.fill_diagonal(sim_matrix, -1e6)
print(sim_matrix, sim_matrix.shape)
names = pickle.load(open('pickles/names.pickle', 'rb'))
error_mat = np.load('npy/breast_pattern_errs.npy')
errors = { target_name:{source_name:error_mat[j][i] for j,source_name in enumerate(names)}
            for i,target_name in enumerate(names[:25])}
for k,v in errors.items():
    optimal_err = np.inf
    for source_name, source_err in v.items():
        if source_name==k: continue
        else:
            optimal_err = min(optimal_err, source_err)

    errors[k]['optimal'] = optimal_err

rel_errors =  []
pearson_p_values = []
pearson_r_values = []
spearman_p_values = []
spearman_r_values = []
for name_i1, (name, row) in enumerate(zip(names[:25], sim_matrix)):
    max_sim_i = np.argmax(row)
    match_name = names[max_sim_i]
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



print("Relative Errors:", np.average(rel_errors))
print("Average Pearson r: {}\nAverage Spearman r: {}".format(np.average(pearson_r_values), np.average(spearman_r_values)))
print("Average Pearson p: {}\nAverage Spearman p: {}".format(np.average(pearson_p_values), np.average(spearman_p_values)))



