package MetaNet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import static experiments.data.DatasetLists.UCIContinuousFileNames;

public class CAWPEOffline {
	private static String[] basicClassifiers = {"Logistic", "C45", "SVML", "NN", "MLP"};
	private static String[] advanceClassifiers = {"MLP2", "XGBoost", "SVMQ", "RandF", "RotF"};

	public static void main(String[] args) throws IOException {
		for (String dataset : UCIContinuousFileNames) {
			int numOfClasses = -1;
			for (int fold = 0; fold < 30; fold++) {

				Scanner[] scanners = new Scanner[5];
				double[] trainAccsToPowerOfFour = new double[5];

				// get train accuracies and test scanners
				int index = 0;

				for (String classifierName : basicClassifiers) {
					File train = new File("F:/University Files/Project/Result/" + classifierName + "/Predictions/" + dataset + "/trainFold" + fold + ".csv");
					File test = new File("F:/University Files/Project/Result/" + classifierName + "/Predictions/" + dataset + "/testFold" + fold + ".csv");

					scanners[index] = new Scanner(test);
					while (!scanners[index].nextLine().equals("#")) ;
					Scanner scanner = new Scanner(train);
					for (int i = 0; i < 2; i++) {
						scanner.nextLine();
					}
					trainAccsToPowerOfFour[index++] = Math.pow(Double.parseDouble(scanner.nextLine().split(",")[0]), 4);
					if (numOfClasses == -1) {
						numOfClasses = Integer.parseInt(scanner.nextLine().split(",")[1]);
					}
				}

				// calculate CAWPE result
				File output = new File("F:/University Files/Project/Result/CAWPEOffline/result/" + dataset + "/fold" + fold + ".csv");
				File outputDirs = new File("F:/University Files/Project/Result/CAWPEOffline/result/" + dataset);
				outputDirs.mkdirs();
//				output.createNewFile();
				FileWriter fw = new FileWriter(output);
				int[] totalCounter = new int[numOfClasses];
				int[] truePredictionCounter = new int[numOfClasses];
				index = 0;
				while (scanners[index].hasNextLine()) {
					int trueClass = -1;
					double[] probabilitiesPerClass = new double[numOfClasses];
					while (index < 5) {
						String[] instance = scanners[index].nextLine().split(",");
						if (trueClass == -1) {
							trueClass = Integer.parseInt(instance[0]);
						}
						for (int i = 0; i < numOfClasses; i++) {
							probabilitiesPerClass[i] += trainAccsToPowerOfFour[index] * Double.parseDouble(instance[3 + i]);
						}
						index++;
					}
					int perdictedClass = indexOfMax(probabilitiesPerClass);
					fw.write(trueClass + "," + perdictedClass + "\n");
					totalCounter[trueClass]++;
					if (trueClass == perdictedClass) truePredictionCounter[trueClass]++;
					index = 0;
				}
				fw.write("#\n");
				fw.write("accuracy,");
				fw.write(String.valueOf(calculateAcc(totalCounter, truePredictionCounter)));
				fw.write("\n");
				fw.write("balanced accuracy,");
				fw.write(String.valueOf(calculateBalancedAcc(totalCounter, truePredictionCounter)));
				fw.write("\n");
				fw.close();
			}
		}
	}

	private static double calculateBalancedAcc(int[] totalCounter, int[] truePredictionCounter) {
		double sumAcc = 0;
		for (int i = 0; i < totalCounter.length; i++) {
			sumAcc += truePredictionCounter[i] / (double) totalCounter[i];
		}
		return sumAcc / totalCounter.length;
	}

	private static double calculateAcc(int[] totalCounter, int[] truePredictionCounter) {
		int sumTotal = 0, trueTotal = 0;
		for (int d : totalCounter) {
			sumTotal += d;
		}
		for (int d : truePredictionCounter) {
			trueTotal += d;
		}
		return trueTotal / (double) sumTotal;
	}

	private static int indexOfMax(double[] probabilitiesPerClass) {
		double max = -1;
		int index = -1;
		for (int i = 0; i < probabilitiesPerClass.length; i++) {
			if (probabilitiesPerClass[i] > max) {
				max = probabilitiesPerClass[i];
				index = i;
			}
		}
		return index;
	}
}
