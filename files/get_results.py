#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import os
import mysql.connector
import sklearn.metrics



confusion_matrices = []

migration_id_arr = [1,2,3,4,5,7,8,10,18]
models = ["CO","DS","MS","ALL"]
algorithms = ["ga","random","HC"]



avg_columns = []
for i in range(0, len(algorithms)):
    for j in range(0, len(models)):
        temp_1 = algorithms[i]+"_"+models[j]+" Accuracy"
        temp_2 = algorithms[i]+"_"+models[j]+" Precision"
        temp_3 = algorithms[i]+"_"+models[j]+" Recall"
        
        avg_columns.append(temp_1)
        avg_columns.append(temp_2)
        avg_columns.append(temp_3)
        avg_columns.append(temp_4)
        avg_columns.append(temp_5)
        avg_columns.append(temp_6)
        avg_columns.append(temp_7)



root_dir_name = "evoApps/dataset/"

hyperparam_results_path = root_dir_name + "/alt_master_averaged_results"
if not os.path.exists(hyperparam_results_path):
    os.system("mkdir "+hyperparam_results_path)

for migration_ru in migration_id_arr:
    migration_id = str(migration_ru)
    folder_path = os.path.join(root_dir_name, "migrationRule"+str(migration_id))
    hyper_params_folders = os.listdir(folder_path)
    averaged_results_df = pd.DataFrame(np.zeros(shape=(len(hyper_params_folders), len(avg_columns))),columns=avg_columns)
    hyperparam_idx = -1
    hyper_params_folders = ["pop_size:250"]
    for hyper_param in hyper_params_folders:
        input_folder_path = os.path.join(folder_path, hyper_param+"/input")
        output_folder_path = os.path.join(folder_path, hyper_param+"/output")
        
        hyperparam_idx = hyperparam_idx + 1
        
        for input_variations in os.listdir(input_folder_path):
            run_folder_paths = os.path.join(input_folder_path, input_variations)
            
            
            input_filename = run_folder_paths+"/knapsack_file"
            var_extension = input_variations.split("_")[1]
            
            if var_extension=="ALL":
                for algo in algorithms:
                    non_optimistic_precision = []
                    non_optimistic_recall = []
                    non_optimistic_accuracy = []
    
                    algo_folder = str(algo)+"_"+str(var_extension)
                    run_folder_paths_arr = os.listdir(output_folder_path+"/"+algo_folder)
                    count = -1
                    
                    column_arr = ["Run #","Non Optimist Accuracy","Error", "Non Optimist Precision","Non Optimist Recall"]
                    metric_dataframe = pd.DataFrame(data=np.zeros((30, len(column_arr))), columns = column_arr)
                    for run_folder in run_folder_paths_arr:
                        run_path = os.path.join(run_folder_paths, run_folder)
                        
                    
                        count = count + 1
                        
                        run_extension = run_folder.split("_")[1]
                        trial_num = int(run_extension)
                    
        
    
                        ga_result_num_arr = []
                        random_result_num_arr = []
                        ground_truth_result_num_arr = []
                        
                        total_count_ground = 0
                        corr_count_ga = 0
                        corr_count_random = 0
                        
                        # migration_id = str(migration_id)#"1"
        
                
                        num_ground_truth_zeros=0
                        
                        
                        num_errors = 0
                  
                        if algo=="HC":
                            ga_output_file = open(output_folder_path+"/"+algo_folder+"/"+run_folder+"/results.txt", "r")
                        else:
                            ga_output_file = open(output_folder_path+"/"+algo_folder+"/"+run_folder+"/results", "r")
                
                
                        input_file = open(input_filename, "r")
                        inputs = input_file.read()
                        match = re.findall("capacity: \+[0-9]+",inputs)
                        num_src_methods = int(re.sub("capacity: \+","",match[0]))
                        
                        items = re.findall("knapsack problem specification \(1 knapsacks, [0-9]+",inputs)
                        num_total_items = int(re.sub("knapsack problem specification \(1 knapsacks, ","",items[0]))
                        
                        num_target_items_per_src = int(num_total_items/num_src_methods)
                
                        similarity_dataset = pd.read_pickle(folder_path+"/"+hyper_param+"/datasets/similarity_master_migrationID_"+str(migration_id)+"_source"+".pkl")
                
                        
                        f = open(folder_path+"/"+hyper_param+"/groundtruth/knapsack_file_grndtrth.txt","r")
                        ground_truth_str = f.read()
                
                        ground_truth_str=re.sub("\n","",ground_truth_str)
                        ground_truth_str=re.sub(" ","",ground_truth_str).strip('[]')
                        ground_truth_str = [int(num_grd) for num_grd in ground_truth_str ]
                        ground_truth_str = list(ground_truth_str)
                        f.close()
                        
                        ground_truth_file = np.asarray(ground_truth_str).astype(int)
                        
                                  
                        ga_temp_arr = [re.findall('[0-9]+',line) for line in ga_output_file]
                        ga_temp_arr = ga_temp_arr[0][len(ga_temp_arr[0])-1]#[3]
                        ga_result = [item for item in ga_temp_arr if item!='']
                        ga_result_str = ga_result[len(ga_result)-1]
                        
                        
                        ga_result_str=re.sub("\n","",ga_result_str)
                        ga_result_str = list(ga_result_str)
                        
                        ga_result = np.asarray(ga_result).astype(float).astype(int)
                
                        ga_simple = ga_result.tolist()
                        ground_truth_simple = list(ground_truth_file)
                
                        src_arr = similarity_dataset["FromCode"]
                        src_arr_w_counts = np.unique(src_arr, return_counts=True)
                        curr_start_idx = 0
                        curr_end_index = 0
                        
                        for i in range(0, len(src_arr_w_counts[0])):
                            total_count_ground = 0
                            corr_count_ga = 0
                            
                            fn_count_ga = 0
                            fp_count_ga = 0
                            
                            curr_start_idx = curr_end_index #i*num_target_items_per_src
                            curr_end_index = curr_start_idx + src_arr_w_counts[1][i]#(i+1)*num_target_items_per_src
                            
                            curr_ground_truth = np.asarray(ground_truth_simple[curr_start_idx:curr_end_index])
                            curr_ga_result = np.asarray(ga_simple[curr_start_idx:curr_end_index])
                            
                            ground_truth_ones_len = len(np.where(curr_ground_truth==1)[0])
                            ga_truth_ones_len = len(np.where(curr_ga_result==1)[0])
                    
                            
                
                
                        ga_result_num_arr.extend(ga_result)
                        ground_truth_result_num_arr.extend(list(ground_truth_file))
                        curr_conf = sklearn.metrics.confusion_matrix(ground_truth_result_num_arr, ga_result_num_arr)
                        #TODO:true then preds
                        non_optimistic_accuracy.append(sklearn.metrics.accuracy_score(ground_truth_result_num_arr, ga_result_num_arr))
                        non_optimistic_precision.append(sklearn.metrics.precision_score(ground_truth_result_num_arr, ga_result_num_arr))
                        non_optimistic_recall.append(sklearn.metrics.recall_score(ground_truth_result_num_arr, ga_result_num_arr))
            
                        metric_dataframe.iloc[trial_num]["Run #"] = trial_num
                        
                        
                        
                        metric_dataframe.iloc[trial_num]["Non Optimist Accuracy"] = sklearn.metrics.accuracy_score(ground_truth_result_num_arr, ga_result_num_arr)
                        metric_dataframe.iloc[trial_num]["Error"] = (num_false_positives+num_false_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives)
                        
                        metric_dataframe.iloc[trial_num]["Non Optimist Precision"] = sklearn.metrics.precision_score(ground_truth_result_num_arr, ga_result_num_arr)
                        metric_dataframe.iloc[trial_num]["Non Optimist Recall"] = sklearn.metrics.recall_score(ground_truth_result_num_arr, ga_result_num_arr)
                    metrics_df_path = folder_path+"/"+hyper_param+"non_optimistic_metrics_dataframes"
                    if not os.path.exists(metrics_df_path):
                        os.mkdir(metrics_df_path)
                    metric_dataframe.to_pickle(metrics_df_path+"non_optimistic_metrics_dataframe_migration_ID_"+str(migration_id)+"_algo_model_"+algo_folder+".pkl")
                
    
                
                    print("**********************************************************")
                        
                    print("Migration Rule:",migration_id)
                    print("Averages")
                    #TODO
                    print("Algorithm model:", algo_folder)
                    print("Hyperparameters:", hyper_param)
                    print("Migration Rule:",migration_id)
                    print("Total number of source methods:", num_src_methods)
                    
                    
                    print("Non-Optimistic Average Accuracy:", np.mean(non_optimistic_accuracy))
                    print("Non-Optimistic Average Precision:",np.mean(non_optimistic_precision))
                    print("Non-Optimistic Average Recall:",  np.mean(non_optimistic_recall))
                    print("Similarity dataset shape:",similarity_dataset.shape)
                    
                
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Accuracy"] = metric_dataframe["Non Optimist Accuracy"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Error"] = metric_dataframe["Error"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Precision"] = metric_dataframe["Non Optimist Precision"].mean()
                    
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Recall"] = metric_dataframe["Non Optimist Recall"].mean()
    ############################################`
    # averaged_results_df.index = hyper_params_folders          
    averaged_results_df.to_pickle(hyperparam_results_path+"/alt_non_optimistic_migrationRule_"+migration_id+".pkl")
        
##############################################################################################################################################################################################################################################################################
