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
public class capacity_ablation {

	/**
	 * Starts the example running the knapsack problem.
	 * 
	 * @param args the command line arguments
	 * @throws IOException if an I/O error occurred
	 */
	
	public static void main(String[] args) throws IOException {
		
		//Find all available folders (aka migration rules)
		String[] keys = {"ALL"};
		String root_dir = "/capacity_ablation/dataset/migrationRule10/num_iter:100pop_size:100";
		String dir_name = "/capacity_ablation/dataset/migrationRule10/num_iter:100pop_size:100/input/";

		File file = new File(dir_name);
		//directories contains the capacity directories
		String[] directories = file.list(new FilenameFilter() {
		  public boolean accept(File current, String name) {
		    return new File(current, name).isDirectory();
		  }
		});
		
		
		

		for(int y=0; y<directories.length;y++) {
			System.out.println(y);
			String curr_dir = dir_name + directories[y]; //this will go up to migration rule directories
			System.out.println(curr_dir);
			//for one num iter

		
			Stream<Path> paths = Files.walk(Paths.get(curr_dir)).filter(Files::isDirectory);
			
			Object[] paths_arr = paths.toArray();
			
			
			//This loop runs each search algorithm 30 times
			for(int k = 0; k< paths_arr.length-1;k++) {
				String curr_input_dir = paths_arr[k].toString();
				Stream<Path> inputPaths = Files.walk(Paths.get(curr_input_dir)).filter(Files::isRegularFile);
				
				Object[] inputPathArr = inputPaths.toArray();
				System.out.println(inputPathArr[0]);
				

				String output_filepath = root_dir+"/output/"
						+directories[y]+"/run_"+k+"/results";

				
				File f_output = new File(output_filepath);
				

					
				FileWriter writer = new FileWriter(output_filepath, false);
				// open the file containing the knapsack problem instance
				String filepath = inputPathArr[0].toString();
				System.out.println("Creating an input stream...");
				InputStream input1 = new FileInputStream(filepath);
				InputStream input2 = new FileInputStream(filepath);
				
				System.out.println(filepath);
				System.out.println("Creating a knapsack instance...");
				
				Knapsack knapsack1 = new Knapsack(input1);

				// solve using single objective GA

				NondominatedPopulation result = new Executor()
						.withProblem(knapsack1)
						.withAlgorithm("ga")
						.withProperty("populationSize",100)
						.withMaxEvaluations(50000)
						.distributeOnAllCores()
						.run();
						
						   

				// print GA results
				//System.out.println("RESULT SIZE:"+result.size());
				
				
				for (int i = 0; i < result.size(); i++) {
					Solution solution = result.get(i);
					double[] objectives = solution.getObjectives();
					
					int num_vars = objectives.length;

					// negate objectives to return them to their maximized form
					objectives = Vector.negate(objectives);
							
					
					for(int j = 0; j<num_vars;j++) {
						System.out.println("    Sack "+j+" Profit: " + objectives[j]);
						System.out.println("    Penalty: "+solution.getConstraint(j));
					}
					System.out.println("    Capacity " + knapsack1.capacity[0]);
					System.out.println("    Number of items " + knapsack1.nitems);
					System.out.println("    Binary String: " + solution.getVariable(0));
					writer.write("Sack "+i+" Profit: " + objectives[i]);
					writer.write("Binary String: " + solution.getVariable(0));
					
						
						
						
					
					writer.close();

						
						
						
						
					}
					
					
					
					
					paths.close();
				}
				
								
			}
		
			
		}

