#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import os
import mysql.connector
import sklearn.metrics


def getArr(a):
    result = np.zeros(len(a), dtype=np.int64)
    
    for i in range(0, len(a)):
        result[i] = int(a[i])
        
    return result

cnx = mysql.connector.connect(user='apimig', password='password',
                              host='127.0.0.1',
                              database='FunctionMapping')
cursor = cnx.cursor(buffered=True)
temp_cursor = cnx.cursor(buffered=True)
query = ("select ID from MigrationRules;")

temp_cursor.execute(query)
migration_id_arr = []
for ids in temp_cursor:
    migration_id_arr.append(ids[0])

temp_cursor.close()
confusion_matrices = []

migration_id_arr = [1,2,3,4,5,7,8,10,18]
#TODO: HERE
models = ["ALL"]#["CO","DS","MS","ALL"]
#TODO: HERE
algorithms = ["HC"]#["ga","random"] #["ga"]



avg_columns = []
for i in range(0, len(algorithms)):
    for j in range(0, len(models)):
        temp_1 = algorithms[i]+"_"+models[j]+" Accuracy"
        temp_2 = algorithms[i]+"_"+models[j]+" Precision"
        temp_3 = algorithms[i]+"_"+models[j]+" Recall"
        
        temp_4 = algorithms[i] +"_"+models[j]+ " TP"
        temp_5 = algorithms[i] +"_"+models[j]+ " FP"
        temp_6 = algorithms[i] +"_"+models[j]+" TN"
        temp_7 = algorithms[i] +"_"+models[j]+" FN"
        
        avg_columns.append(temp_1)
        avg_columns.append(temp_2)
        avg_columns.append(temp_3)
        avg_columns.append(temp_4)
        avg_columns.append(temp_5)
        avg_columns.append(temp_6)
        avg_columns.append(temp_7)


# migration_id_arr = [3]
root_dir_name = "/home/nd7896/data/API_Migration/Small_Equal_Evaluations/Stochastic_Hill_Climbing/Input_Files_MANY"


hyperparam_results_path = root_dir_name + "/alt_master_averaged_results"
if not os.path.exists(hyperparam_results_path):
    os.system("mkdir "+hyperparam_results_path)

for migration_ru in migration_id_arr:
    migration_id = str(migration_ru)
    folder_path = os.path.join(root_dir_name, "migrationRule"+str(migration_id))
    hyper_params_folders = os.listdir(folder_path)
    averaged_results_df = pd.DataFrame(np.zeros(shape=(len(hyper_params_folders), len(avg_columns))),columns=avg_columns)
    hyperparam_idx = -1
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
                    
                    column_arr = ["Run #","Non Optimist Accuracy","Error", "Non Optimist Precision","Non Optimist Recall","TP","FP","TN","FN"]
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
        
                
                        num_false_negatives = 0
                        #num_successful_src_replacement = 0
                        num_unsuccessful_src_replacement = 0
                        num_false_positives = 0
                        num_ground_truth_zeros=0
                        
                        num_true_negatives = 0
                        num_true_positives = 0
                        
                        num_errors = 0
                        #while count<0:#num_knapsacks-1:
                  
                        #TODO:HERE
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
                        # ground_truth_str = ground_truth_str.split(',')
                        ground_truth_str = [int(num_grd) for num_grd in ground_truth_str ]
                        ground_truth_str = list(ground_truth_str)
                        f.close()
                        
                        ground_truth_file = np.asarray(ground_truth_str).astype(int)
                        
                                  
                        ga_temp_arr = [re.findall('[0-9]+.0',line) for line in ga_output_file]
                        #ga_temp_arr = [re.findall('[0-9]+',line) for line in ga_output_file]#[re.findall('0*1*0*',line) for line in ga_output_file]
                        ga_temp_arr = ga_temp_arr[0]#[3]
                        ga_result = [item for item in ga_temp_arr if item!='']
                        #[item for item in ga_temp_arr[len(ga_temp_arr)-1] if item!='']
                        ga_result_str = ga_result[len(ga_result)-1]
                        
                        
                        ga_result_str=re.sub("\n","",ga_result_str)
                        ga_result_str = list(ga_result_str)
                        
                        ga_result = np.asarray(ga_result).astype(float).astype(int)
                
                        ga_simple = ga_result.tolist()
                        #random_simple = list(getArr(random_result[0]))
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
                    
                            if ground_truth_ones_len>0 and ga_truth_ones_len>0:
                                #print("For SRC Method: #",i, " there exists a ground truth nd a ga result")
                                for g in range(0,len(curr_ga_result)):
                                    if curr_ga_result[g] == 1 and curr_ground_truth[g] == 1:
                                        corr_count_ga=corr_count_ga + 1
                                    #
                                    if curr_ground_truth[g]==1:
                                        total_count_ground = total_count_ground + 1
                                
                                if corr_count_ga == 0:
                                    #num_false_negatives += 1
                                    #num_errors += 1
                                    #I CHANGED HERE
                                    num_false_positives += 1
                                else:
                                    num_true_positives += 1
                
                                    
                            elif ground_truth_ones_len<=0 and ga_truth_ones_len<=0:
                                #print("Not found for SRC Method: #",i, " as ground truth has no selections available")
                                #num_successful_src_replacement += 1
                                num_true_negatives += 1
                            elif ground_truth_ones_len<=0 and ga_truth_ones_len>0:
                                #print("Not found for SRC Method: #",i, " as ground truth has no selections available")
                                num_false_positives += 1
                            elif ground_truth_ones_len >0 and ga_truth_ones_len<=0:
                                #print("Not found for SRC Method: #",i)
                                num_false_negatives  += 1 
                            if ground_truth_ones_len<=0:
                                num_ground_truth_zeros += 1
                            
                
                
                        ga_result_num_arr.extend(ga_result)#(list(getArr(ga_result[0])))
                        #random_result_num_arr.extend(list(getArr(random_result[0])))
                        #ground_truth_result_num_arr.append(list(getArr(ground_truth_result[0])))
                        ground_truth_result_num_arr.extend(list(ground_truth_file))
                        curr_conf = sklearn.metrics.confusion_matrix(ground_truth_result_num_arr, ga_result_num_arr)
                        #TODO:true then preds
                        non_optimistic_accuracy.append(sklearn.metrics.accuracy_score(ground_truth_result_num_arr, ga_result_num_arr))
                        non_optimistic_precision.append(sklearn.metrics.precision_score(ground_truth_result_num_arr, ga_result_num_arr))
                        non_optimistic_recall.append(sklearn.metrics.recall_score(ground_truth_result_num_arr, ga_result_num_arr))
            
                        metric_dataframe.iloc[trial_num]["Run #"] = trial_num
                        metric_dataframe.iloc[trial_num]["TP"] = num_true_positives
                        metric_dataframe.iloc[trial_num]["FP"] = num_false_positives
                        
                        metric_dataframe.iloc[trial_num]["TN"] = num_true_negatives
                        metric_dataframe.iloc[trial_num]["FN"] = num_false_negatives
                        
                        
                        # precision_denom = (num_true_positives + num_false_positives)
                        # recall_denom = num_true_positives + num_false_negatives
                        # if precision_denom == 0:
                        #     precision=0
                        # else:
                        #     precision = num_true_positives/precision_denom
                            
                        # if recall_denom == 0:
                        #     recall = 0
                        # else:
                        #     recall = num_true_positives/recall_denom
            
                        
                        metric_dataframe.iloc[trial_num]["Non Optimist Accuracy"] = sklearn.metrics.accuracy_score(ground_truth_result_num_arr, ga_result_num_arr)
                        #(num_true_positives+num_true_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives)#(num_true_positives+num_true_negatives)/(num_src_methods)
                        metric_dataframe.iloc[trial_num]["Error"] = (num_false_positives+num_false_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives)
                        
                        metric_dataframe.iloc[trial_num]["Non Optimist Precision"] = sklearn.metrics.precision_score(ground_truth_result_num_arr, ga_result_num_arr)
                        #precision
                        metric_dataframe.iloc[trial_num]["Non Optimist Recall"] = sklearn.metrics.recall_score(ground_truth_result_num_arr, ga_result_num_arr)
                        #recall
                    #TODO:HERE
                    metrics_df_path = folder_path+"/"+hyper_param+"non_optimistic_metrics_dataframes"
                    if not os.path.exists(metrics_df_path):
                        os.mkdir(metrics_df_path)
                    metric_dataframe.to_pickle(metrics_df_path+"non_optimistic_metrics_dataframe_migration_ID_"+str(migration_id)+"_algo_model_"+algo_folder+".pkl")
                
    
                
                    print("**********************************************************")
                        #print("Number of correct substitutions Random:", corr_count_random)
                        #confusion_matrices.append(curr_conf)
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
                    
                    # print("Optimistic Average Accuracy:", metric_dataframe["Non Optimist Accuracy"].mean())
                    # print("Optimistic Average Error:", metric_dataframe["Error"].mean())
                    # print("Optimistic Average Precision:", metric_dataframe["Non Optimist Precision"].mean())
                    # print("Optimistic Average Recall:",  metric_dataframe["Non Optimist Recall"].mean())
                    
                    # print("Optimistic Average TP:", metric_dataframe["TP"].mean())
                    # print("Optimistic Average FP:", metric_dataframe["FP"].mean())
                    # print("Optimistic Average TN:", metric_dataframe["TN"].mean())
                    # print("Optimistic Average FN:",  metric_dataframe["FN"].mean())
                
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Accuracy"] = metric_dataframe["Non Optimist Accuracy"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Error"] = metric_dataframe["Error"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Precision"] = metric_dataframe["Non Optimist Precision"].mean()
                    
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" Recall"] = metric_dataframe["Non Optimist Recall"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" TP"] = metric_dataframe["TP"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" FP"] = metric_dataframe["FP"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" TN"] = metric_dataframe["TN"].mean()
                    averaged_results_df.iloc[hyperparam_idx][algo+"_"+var_extension+" FN"] = metric_dataframe["FN"].mean()
    ############################################`
    averaged_results_df.index = hyper_params_folders          
    averaged_results_df.to_pickle(hyperparam_results_path+"/alt_non_optimistic_migrationRule_"+migration_id+".pkl")
        
##############################################################################################################################################################################################################################################################################
        
'''       
    for curr_model in models:
        ga_result_num_arr = []
        random_result_num_arr = []
        ground_truth_result_num_arr = []
        
        total_count_ground = 0
        corr_count_ga = 0
        corr_count_random = 0
        
        # migration_id = str(migration_id)#"1"
    
            
        count = -1
        num_false_negatives = 0
        #num_successful_src_replacement = 0
        num_unsuccessful_src_replacement = 0
        num_false_positives = 0
        num_ground_truth_zeros=0
        
        num_true_negatives = 0
        num_true_positives = 0
        
        num_errors = 0
        #while count<0:#num_knapsacks-1:
        column_arr = ["Run #","Accuracy","Error", "Precision","Recall","TP","FP","TN","FN"]
        metric_dataframe = pd.DataFrame(data=np.zeros((30, len(column_arr))), columns = column_arr)
            
        for trial_num in range(0,30):
            count+=1
        
            
            hyper_param_path = 
            
            input_file = open(dir_name+"/migrationRule"+str(migration_id)+"/input/run_"+str(trial_num)+"/knapsack_file", "r")
            #TODO:HERE
            ga_output_file = open(dir_name+"/migrationRule"+str(migration_id)+"/output/ga/run_"+str(trial_num)+"/results", "r")
            random_output_file = open(dir_name+"/migrationRule"+str(migration_id)+"/output/random/run_"+str(trial_num)+"/results","r")
            
            
            
            inputs = input_file.read()
            match = re.findall("capacity: \+[0-9]+",inputs)
            num_src_methods = int(re.sub("capacity: \+","",match[0]))
            
            items = re.findall("knapsack problem specification \(1 knapsacks, [0-9]+",inputs)
            num_total_items = int(re.sub("knapsack problem specification \(1 knapsacks, ","",items[0]))
            
            num_target_items_per_src = int(num_total_items/num_src_methods)
            
            #random_output_file = open(dir_name+"/migrationRule"+str(migration_id)+"/output/random/results_"+str(count), "r")
            #ground_truth_file = np.load("/home/nd7896/data/API Migration/Input_Files/migrationRule"+str(migration_id)+"/groundtruth/knapsack_file"+".npy").astype(int)
            similarity_dataset = pd.read_pickle("Input_Files_MANY/migrationRule"+str(migration_id)+"/datasets/similarity_master_migrationID_"+str(migration_id)+"_source"+".pkl")
            
            f = open("/home/nd7896/data/API Migration/Input_Files_MANY/migrationRule"+str(migration_id)+"/groundtruth/knapsack_file.txt","r")
            ground_truth_str = f.read()
            import re
            ground_truth_str=re.sub("\n","",ground_truth_str)
            
            ground_truth_str = list(ground_truth_str)
            f.close()
            
            ground_truth_file = np.asarray(ground_truth_str).astype(int)
    
            ga_temp_arr = [re.findall('[0-9]+',line) for line in ga_output_file]
            #ga_temp_arr = [re.findall('[0-9]+',line) for line in ga_output_file]#[re.findall('0*1*0*',line) for line in ga_output_file]
            ga_result = [item for item in ga_temp_arr[len(ga_temp_arr)-1] if item!='']
            ga_result_str = ga_result[len(ga_result)-1]
            
            
            ga_result_str=re.sub("\n","",ga_result_str)
            ga_result_str = list(ga_result_str)
            
            ga_result = np.asarray(ga_result_str).astype(int)
    
            ga_simple = ga_result.tolist()
            #random_simple = list(getArr(random_result[0]))
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
                
                if ground_truth_ones_len>0 and ga_truth_ones_len>0:
                    #print("For SRC Method: #",i, " there exists a ground truth nd a ga result")
                    for g in range(0,len(curr_ga_result)):
                        if curr_ga_result[g] == 1 and curr_ground_truth[g] == 1:
                            corr_count_ga=corr_count_ga + 1
                        # if curr_ga_result[g] == 0 and curr_ground_truth[g] == 1:
                        #     fn_count_ga = fn_count_ga + 1
                        # if curr_ga_result[g] == 1 and curr_ground_truth[g] == 0:
                        #     fp_count_ga = fp_count_ga + 1
                        # if curr_ga_result[g] == 0 and curr_ground_truth[g] == 0:
                        #     corr_count_ga=corr_count_ga + 1
                        if curr_ground_truth[g]==1:
                            total_count_ground = total_count_ground + 1
                    #print("Number of correct mappings found: ", corr_count_ga, " out of: ",total_count_ground)
                    
                    if corr_count_ga == 0:
                        #num_false_negatives += 1
                        #num_errors += 1
                        #I CHANGED HERE
                        num_false_positives += 1
                    else:
                        num_true_positives += 1
    
                        
                elif ground_truth_ones_len<=0 and ga_truth_ones_len<=0:
                    #print("Not found for SRC Method: #",i, " as ground truth has no selections available")
                    #num_successful_src_replacement += 1
                    num_true_negatives += 1
                elif ground_truth_ones_len<=0 and ga_truth_ones_len>0:
                    #print("Not found for SRC Method: #",i, " as ground truth has no selections available")
                    num_false_positives += 1
                elif ground_truth_ones_len >0 and ga_truth_ones_len<=0:
                    #print("Not found for SRC Method: #",i)
                    num_false_negatives  += 1 
                if ground_truth_ones_len<=0:
                    num_ground_truth_zeros += 1
                
            
            
            ga_result_num_arr.extend(ga_result)#(list(getArr(ga_result[0])))
            #random_result_num_arr.extend(list(getArr(random_result[0])))
            #ground_truth_result_num_arr.append(list(getArr(ground_truth_result[0])))
            ground_truth_result_num_arr.extend(list(ground_truth_file))
            curr_conf = metrics.confusion_matrix(ground_truth_result_num_arr, ga_result_num_arr)
            #print("EQUAL LENGTHS?",len(ground_truth_result_num_arr)==len(ga_result_num_arr))
            #curr_conf_random = metrics.confusion_matrix(ground_truth_result_num_arr, random_result_num_arr)
            # print("Migration Rule:",migration_id)
            # print("Total number of source methods:", num_src_methods)
            # print("Number of ground truth zeros:",num_ground_truth_zeros)
            # print("Number of errors:",num_errors)
            # print("Number of true negatives:",num_true_negatives)
            # print("Number of true positives:",num_true_positives)
            # print("Number of false negatives:",num_false_negatives)
            # print("Number of false positives:",num_false_positives)
            # print("Number of successful replacements:",num)
            #print("Number of unsuccessful replacements:",num_unsuccessful_src_replacement)
            
            metric_dataframe.iloc[trial_num]["Run #"] = trial_num
            metric_dataframe.iloc[trial_num]["TP"] = num_true_positives
            metric_dataframe.iloc[trial_num]["FP"] = num_false_positives
            
            metric_dataframe.iloc[trial_num]["TN"] = num_true_negatives
            metric_dataframe.iloc[trial_num]["FN"] = num_false_negatives
            
            
            precision_denom = (num_true_positives + num_false_positives)
            recall_denom = num_true_positives + num_false_negatives
            if precision_denom == 0:
                precision=0
            else:
                precision = num_true_positives/precision_denom
                
            if recall_denom == 0:
                recall = 0
            else:
                recall = num_true_positives/recall_denom
            # print("Precision: ", precision)
            # print("Recall: ", recall)
            # print("Accuracy: ", (num_true_positives+num_true_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives))#(num_true_positives+num_true_negatives)/(num_src_methods))
            # print("Error: ", (num_false_positives+num_false_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives))#(num_src_methods))
            
            metric_dataframe.iloc[trial_num]["Accuracy"] = (num_true_positives+num_true_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives)#(num_true_positives+num_true_negatives)/(num_src_methods)
            metric_dataframe.iloc[trial_num]["Error"] = (num_false_positives+num_false_negatives)/(num_true_positives+num_true_negatives+num_false_positives+num_false_negatives)
            
            metric_dataframe.iloc[trial_num]["Precision"] = precision
            metric_dataframe.iloc[trial_num]["Recall"] = recall
            #TODO:HERE
            metric_dataframe.to_pickle("/home/nd7896/data/API Migration/Input_Files_MANY/metrics_dataframes/ga_metrics_dataframe_migration_ID_"+str(trial_num)+".pkl")
            
            
            print("GA Confusion Matrix for Migration Rule",migration_id,":",curr_conf)
            #print("Random Confusion Matrix for Migration Rule",migration_id,":",curr_conf_random)
            #print("Number of correct substitutions GA:", corr_count_ga,total_count_ground)
            
            
            print("Precision GA:", metrics.precision_score(ground_truth_result_num_arr, ga_result_num_arr))
            print("Recall GA:", metrics.recall_score(ground_truth_result_num_arr, ga_result_num_arr))
            
            
        print("**********************************************************")
            #print("Number of correct substitutions Random:", corr_count_random)
            #confusion_matrices.append(curr_conf)
        print("Averages")
        print("Migration Rule:",migration_id)
        print("Total number of source methods:", num_src_methods)
        print("Average Accuracy:", metric_dataframe["Accuracy"].mean())
        print("Average Error:", metric_dataframe["Error"].mean())
        print("Average Precision:", metric_dataframe["Precision"].mean())
        print("Average Recall:",  metric_dataframe["Recall"].mean())
        
        print("Average TP:", metric_dataframe["TP"].mean())
        print("Average FP:", metric_dataframe["FP"].mean())
        print("Average TN:", metric_dataframe["TN"].mean())
        print("Average FN:",  metric_dataframe["FN"].mean())
'''