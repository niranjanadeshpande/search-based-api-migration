#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 19:16:33 2021

@author: nd7896
"""
import numpy as np
import pandas as pd
import re
import os
import sys

def read_file(filename):
    print("filename:", filename)
    num_methods, capacity, item_dataset = None, None, None
    specification_regex = re.compile("knapsack problem specification \(1 knapsacks, [0-9]* items\)")
    numbers = re.compile("\ *[0-9]* items")
    capacity_regex = re.compile("capacity: \+[0-9]*.[0-9]*")
    weight_regex = re.compile("weight: \+[0-9]+")
    profit_regex = re.compile("profit: \+[0-9]*.[0-9]*")
    item_regex = re.compile("item: [0-9]+")
    #item dataset must be a pandas datatype
    with open(filename) as file:
        lines = file.readlines()
        for i in range(0, len(lines)):
            # print(lines[i])
            line = lines[i].strip("\n")
            proc_line = line.rstrip()
            proc_line = proc_line.lstrip()
            if specification_regex.search(proc_line):
                num_methods = int(numbers.findall(proc_line)[0].strip(" items"))
                item_dataset = pd.DataFrame(data=np.zeros(shape=(num_methods,2)), columns=["Fitness", "Weight"])
            if capacity_regex.search(proc_line):
                capacity = int(capacity_regex.search(proc_line)[0].strip("capacity:\ *"))
            if item_regex.search(proc_line):
                item_line = item_regex.search(proc_line)
                item_idx = int(item_line[0].strip("item:\ *"))
                
                item_weight_line = lines[i+1].rstrip().lstrip()
                item_profit_line = lines[i+2].rstrip().lstrip()
                item_weight = weight_regex.search(item_weight_line)[0].strip("weight:\ *")
                item_fitness = profit_regex.search(item_profit_line)[0].strip("profit:\ *")
                
                item_dataset["Fitness"].iloc[item_idx-1] =  float(item_fitness)
                item_dataset["Weight"].iloc[item_idx-1] = float(item_weight)
    
    
    return num_methods, capacity, item_dataset

def bit_flip(idx, solution):
    if solution[idx] == 0:
        solution[idx] = 1
    elif solution[idx] == 1:
        solution[idx] = 0
    return solution


def get_fitness(dataset, solution):
    fitness = 0
    
    for i in range(0, len(solution)):
        if solution[i] == 1:
            fitness = fitness + dataset["Fitness"].iloc[i]
    
    return fitness

def hill_climbing(start_solution, num_methods, capacity, dataset):
    best_solution = start_solution
    best_fitness = float('-inf')
    for i in range(0, len(start_solution)):
        
        curr_solution = np.asarray(bit_flip(i, start_solution))
        curr_fitness = get_fitness(dataset, curr_solution)
        curr_weight = len(np.where(curr_solution==1)[0])
        if curr_weight <= capacity:
            if curr_fitness > best_fitness:
                best_solution = curr_solution
                best_fitness = curr_fitness
    
    return best_solution

def main():
    
    
    
    filename = str(sys.argv[1])
    output_run_path = str(sys.argv[2])
    num_iterations = int(sys.argv[3])
    num_methods, capacity, dataset = read_file(filename)
                            
    start_solution = np.zeros(shape=(len(dataset)))
    ones_idx = np.random.choice(np.arange(0, len(dataset)), np.random.choice(len(dataset)))
    for each_idx in ones_idx:
        start_solution[each_idx] = 1
    
    solution = start_solution
    for i in range(0, num_iterations):
        
        solution = hill_climbing(solution, num_methods, capacity, dataset)
    
        output_filename = output_run_path+"/results.txt"
        #Write solution to file
        fo = open(output_filename, "w+")
        fo.write("Binary String:")
        # print(solution.tolist())
        fo.write(str(solution.tolist()))
        fo.close()
        


if __name__=="__main__":
    main()
