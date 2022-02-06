This repository contains all the files required to replicate results from the paper "Search-Based Third-Party Library Migration at the Method-Level".

To be able to run this code, you need to install the MOEA framework library that can be found at "http://moeaframework.org/".
The suggested method to run search algorithms is by copying all files and folders to the "../MOEAFramework-2.13/examples/org/moeaframework/examples/ga/knapsack/files" directory.

+ The folder "dataset" contains all the input files required to run GA for each of 9 migration rules.
	+ Each migration rule has a corresponding directory where input and output files are stored. Each migration rule is associated with an id, for example, the id 		  for the migration rule logging->slf4j is 1, so all results are stored in the sub-directory "migrationRule1". The ids are as follows:

		---------------------------------------
		||  ID |   Migration Rule            ||
		---------------------------------------
		||  1  | logging ---> slf4j 	     ||
		||  2  | slf4j-api ---> log4j	     ||
		||  3  | easymock ---> mockito       ||
		||  4  | google-collect ---> guava   ||
		||  5  | gson ---> jackson           ||
		||  7  | testng ---> junit           ||
		||  8  | json ---> gson              ||
		||  10 | commons-lang ---> slf4j-api ||
		||  18 | json-simple ---> gson       ||
		---------------------------------------

	+ Each migration rule directory consists of 4 sub-directories: "groundtruth", "dataset", "input" and "output".The groundtruth directory contains the correct 		  source-target method mappings, ad the input directory contains formatted knapsack files with information with the capacity constraints, and the fitness score 	  for each potential source-target method mapping. The output directory stores the results from each search algorithm (GA, hill-climbing and random 		  	    search) run with different similarity scores (CO, DS, MS, ALL).


+ The files "API_Migration_Example.java" and "Knapsack.java" contain the knapsack code to run GA and random search using the MOEA framework. In particular, the API_Migration_Example.java file iterates over all input files to run GA and random search 30 times and store their results.
+ The file "hill_climbing.py" contains code to run stochastic hill climbing. It takes three input arguments: the file name, output path and the number of iterations* population size.
+ The python file "get_results.py" calculates the average precision and recall over 30 runs for each migration rule, similarity score and search algorithm. These results are printed out on the console.

The "capacity_ablation" folder contains the dataset and files used to generate search results or different capacity constraints. 
+ The "capacity_ablation.java" file contains code to generate GA results for different capacity constraints. This experiment is performed on the migration rule commons-lang ---> slf4j-api, with capacity values ranging from 1 to 20 (the input folders are numbered from capacity_1 to capacity_20). The results  from each of 30 runs are storedare storedand stores all results in the output directory. 
+ performed on the same dataset for migration rule 10











