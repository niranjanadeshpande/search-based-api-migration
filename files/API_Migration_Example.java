/* Copyright 2009-2019 David Hadka
 *
 * This file is part of the MOEA Framework.
 *
 * The MOEA Framework is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * The MOEA Framework is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with the MOEA Framework.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.moeaframework.examples.ga.knapsack;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.*;
import java.util.stream.Stream;
import java.io.PrintWriter;
import org.moeaframework.Executor;
import org.moeaframework.Analyzer;
import org.moeaframework.Instrumenter;
import org.moeaframework.analysis.collector.Accumulator;

import org.moeaframework.core.Solution;
import org.moeaframework.util.Vector;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Problem;

/**
 * Example of binary optimization using the {@link Knapsack} problem on the
 * {@code knapsack.100.2} instance.
 */
public class API_Migration_Example {

	/**
	 * Starts the example running the knapsack problem.
	 * 
	 * @param args the command line arguments
	 * @throws IOException if an I/O error occurred
	 */
	
	public static void main(String[] args) throws IOException {
		
		//Find all available folders (aka migration rules)
		String[] keys = {"CO","MS","DS","ALL"};
		Integer[] num_iterations = {100};//{50000,100000, 500000};//{100};////{100};//, 1000,10000, 100000, 1000000, 500000};
		Integer[] pop_arr = {100};//{100,1000,10000};//{100};//{100, 1000, 10000};//{100};//, 1000, 10000, 50000, 100000, 500000};
		String dir_name = "/home/nd7896/data/API_Migration/MOEAFramework-2.13/examples/org/moeaframework/examples/ga/knapsack/files/Input_Files_MANY";
		File file = new File(dir_name);
		String[] directories = file.list(new FilenameFilter() {
		  public boolean accept(File current, String name) {
		    return new File(current, name).isDirectory();
		  }
		});
		
		
		
		//For each migration rule, read its corresponding knapsack file
		for(int y=0; y<directories.length;y++) {
			String curr_dir = dir_name +"/"+ directories[y]; //this will go up to migration rule directories
			System.out.println(curr_dir);
			
			for(int iter_idx=0;iter_idx<num_iterations.length;iter_idx++) {
				for(int pop_idx=0; pop_idx<pop_arr.length; pop_idx++) {
					for(int key_idx=0; key_idx<keys.length; key_idx++) {
						//"num_iter:"+str(iter_num)+"pop_size:"+str(pop)
						String hyper_path = curr_dir+"/num_iter:"
								+num_iterations[iter_idx].toString()
					+"pop_size:"+pop_arr[pop_idx].toString();
						Stream<Path> paths = Files.walk(Paths.get(hyper_path+"/input/input_"+keys[key_idx])).filter(Files::isDirectory);
						//{paths.filter(Files::isRegularFile);
						        //.forEach(System.out::println);
						
						Object[] paths_arr = paths.toArray();
						System.out.println(paths_arr.length);
						
						
						//this goes from 1 to 30
						for(int k = 0; k< paths_arr.length-1;k++) {
							String curr_input_dir = paths_arr[k].toString();
							Stream<Path> inputPaths = Files.walk(Paths.get(curr_input_dir)).filter(Files::isRegularFile);
							Object[] inputPathArr = inputPaths.toArray();
							
							
							
							
							String output_filepath = hyper_path+"/output/ga_"+keys[key_idx]+"/run_"+k+"/results";
							String random_output_filepath = hyper_path+"/output/random_"+keys[key_idx]+"/run_"+k+"/results";
							
							File f_output = new File(output_filepath);
							File random_output = new File(random_output_filepath);
							if(!f_output.exists() && !f_output.isDirectory() && random_output.exists() && !random_output.isDirectory()) { 
							    // do something
								System.out.println("These particular files exist."+ output_filepath+ random_output_filepath);
							}
							else {
								
								//PrintWriter writer = new PrintWriter(output_filepath, "UTF-8");
								FileWriter writer = new FileWriter(output_filepath, false);
								FileWriter random_writer = new FileWriter(random_output_filepath, false);
								// open the file containing the knapsack problem instance
								String filepath = inputPathArr[0].toString();//paths_arr[k].toString();
								//String filepath = "/home/nd7896/data/API Migration/MOEAFramework-2.13/examples/org/moeaframework/examples/ga/knapsack/knapsack.100.2";
								System.out.println("Creating an input stream...");
								InputStream input1 = new FileInputStream(filepath);
								InputStream input2 = new FileInputStream(filepath);
								
								System.out.println(filepath);
								System.out.println("Creating a knapsack instance...");
								
								Knapsack knapsack1 = new Knapsack(input1);
								Knapsack random_knapsack = new Knapsack(input2);

								// solve using NSGA-II
								System.out.println("Running GA...");
								NondominatedPopulation result = new Executor()
										.withProblem(knapsack1)
										.withAlgorithm("ga")
										.withProperty("populationSize",100)//pop_arr[pop_idx])//250)//pop_arr[iter_idx])
										.withMaxEvaluations(50000)//num_iterations[iter_idx])//50000)//num_iterations[iter_idx])//50000)//num_iterations[iter_idx])
										.distributeOnAllCores()
										.run();
										
										//num_iterations[iter_idx]
										//pop_arr[pop_idx]
									System.out.println("Running Random...");
									NondominatedPopulation random_result = new Executor()
											.withProblem(random_knapsack)
											.withAlgorithm("random")
											.withProperty("populationSize",100)//pop_arr[pop_idx])//250)//pop_arr[iter_idx])
											.withMaxEvaluations(50000)//num_iterations[iter_idx])//50000)//num_iterations[iter_idx])//50000)//(num_iterations[iter_idx])
											.distributeOnAllCores()
											.run();
									
									
										   

								// print GA results
								System.out.println("RESULT SIZE:"+result.size());
								
								//Solution solution1 = result.get(0);
								//System.out.println(solution1.getNumberOfVariables());
								
								for (int i = 0; i < result.size(); i++) {
									Solution solution = result.get(i);
									double[] objectives = solution.getObjectives();
									
									int num_vars = objectives.length;//solution.getNumberOfVariables();//

									// negate objectives to return them to their maximized form
									objectives = Vector.negate(objectives);
											
									//System.out.println("Solution " + (i+1) + ":");
									
									for(int j = 0; j<num_vars;j++) {
										System.out.println("    Sack "+j+" Profit: " + objectives[j]);
										System.out.println("    Penalty: "+solution.getConstraint(j));
										//System.out.println("    Binary String: " + solution.getVariable(j));
									}
									System.out.println("    Capacity " + knapsack1.capacity[0]);
									System.out.println("    Number of items " + knapsack1.nitems);
									System.out.println("    Binary String: " + solution.getVariable(0));
									//System.out.println(" G:"+solution.getConstraints());
									//System.out.println("    Sack "+i+" Profit: " + objectives[i]);
									writer.write("Sack "+i+" Profit: " + objectives[i]);
									//System.out.println("    Sack 2 Profit: " + objectives[1]);
									//System.out.println(solution.getNumberOfVariables());
									//System.out.println("    Binary String: " + solution.getVariable(0));
									writer.write("Binary String: " + solution.getVariable(0));
									
									
									
									
								}
								writer.close();

								
								//print random results							
								Solution random_solution = random_result.get(0);
								System.out.println(random_solution.getNumberOfVariables());
								
								for (int i = 0; i < random_result.size(); i++) {
									Solution solution = random_result.get(i);
									double[] objectives = solution.getObjectives();
									
									int num_vars = objectives.length;//solution.getNumberOfVariables();//

									// negate objectives to return them to their maximized form
									objectives = Vector.negate(objectives);
											
									//System.out.println("Solution " + (i+1) + ":");
									
									for(int j = 0; j<num_vars;j++) {
										System.out.println("    Sack "+j+" Profit: " + objectives[j]);
										System.out.println("    Penalty: "+solution.getConstraint(j));
										//System.out.println("    Binary String: " + solution.getVariable(j));
									}
									System.out.println("    Capacity " + random_knapsack.capacity[0]);
									System.out.println("    Number of items " + random_knapsack.nitems);
									System.out.println("    Binary String: " + solution.getVariable(0));
									
									random_writer.write("Sack "+i+" Profit: " + objectives[i]);
									
									random_writer.write("Binary String: " + solution.getVariable(0));
									
									
									
									
								}
								
								
								random_writer.close();
								
								
								System.out.println("DONE.");
								paths.close();
							}
							
							
							
							
						}
					}
				}
				
				
							
								
			}
							

			
			
		}
		
		System.out.println("FINAL DONE.");

	}

}
