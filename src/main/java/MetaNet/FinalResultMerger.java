package MetaNet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Scanner;

import static experiments.data.DatasetLists.UCIContinuousFileNames;

public class FinalResultMerger {
	private static String[] basicClassifiers = {"Logistic", "C45", "SVML", "NN", "MLP"};

	public static void main(String[] args) throws IOException {

		String resultPath = "F:/University Files/Project/Result/";
		File finalResults = new File(resultPath + "finalResults.csv");
		FileWriter fw = new FileWriter(finalResults);
		boolean first = true;
		ArrayList<String> keys = null;

		for (String dataset : UCIContinuousFileNames) {
			HashMap<String, Double> statics = new HashMap<>();

			// add basic classifiers statics
			for (String classier : basicClassifiers) {
				HashMap<String, Double> sumStatics = new HashMap<>();
				sumStatics.put("acc", 0.0);
				sumStatics.put("balancedAcc", 0.0);
				sumStatics.put("sensitivity", 0.0);
				sumStatics.put("precision", 0.0);
				sumStatics.put("recall", 0.0);
				sumStatics.put("specificity", 0.0);
				sumStatics.put("f1", 0.0);
				sumStatics.put("mcc", 0.0);
				sumStatics.put("nll", 0.0);
				sumStatics.put("meanAUROC", 0.0);
				for (int fold = 0; fold < 30; fold++) {
					File file = new File(resultPath + classier + "/Predictions/" + dataset + "/testFold" + fold + ".csv");
					Scanner scanner = new Scanner(file);
					for (int i = 0; i < 5; i++) {
						scanner.nextLine();
					}
					for (int i = 0; i < 10; i++) {
						String[] stat = scanner.nextLine().split(",");
						sumStatics.put(stat[0], sumStatics.get(stat[0]) + Double.valueOf(stat[1]));
					}
				}

				for (String key : sumStatics.keySet()) {
					statics.put(classier + "_" + key, sumStatics.get(key) / 30.0);
				}
			}

			// add CAWPEOffline result
			HashMap<String, Double> sumStatics = new HashMap<>();
			sumStatics.put("accuracy", 0.0);
			sumStatics.put("balanced accuracy", 0.0);
			for (int fold = 0; fold < 30; fold++) {
				File file = new File(resultPath + "CAWPEOffline/result/" + dataset + "/fold" + fold + ".csv");
				Scanner scanner = new Scanner(file);
				while (!scanner.nextLine().equals("#")) ;
				for (int i = 0; i < 2; i++) {
					String[] stat = scanner.nextLine().split(",");
					sumStatics.put(stat[0], sumStatics.get(stat[0]) + Double.valueOf(stat[1]));
				}
			}
			for (String key : sumStatics.keySet()) {
				statics.put("CAWPE_" + key, sumStatics.get(key) / 30.0);
			}

			// add metanet results

			sumStatics = new HashMap<>();
			sumStatics.put("accuracy", 0.0);
			sumStatics.put("balanced accuracy", 0.0);
			sumStatics.put("negetive log likelihood", 0.0);
			sumStatics.put("AUC", 0.0);

			for (int fold = 0; fold < 30; fold++) {
				File file = new File(resultPath + "MetaNet/result/C45_Logistic_MLP_NN_SVML_/" + dataset + "/stats_" + fold + ".txt");
				Scanner scanner = new Scanner(file);
				while (!scanner.nextLine().equals("=================================================================="))
					;
				for (int i = 0; i < 4; i++) {
					String[] stat = scanner.nextLine().split(":");
					sumStatics.put(stat[0], sumStatics.get(stat[0]) + Double.valueOf(stat[1]));
				}
			}
			for (String key : sumStatics.keySet()) {
				statics.put("Metanet_" + key, sumStatics.get(key) / 30.0);
			}
			if (first) {
				keys = new ArrayList<>(statics.keySet());
				Collections.sort(keys);
				for (String key : keys) {
					fw.write("," + key);
				}
				first = false;
			}
			fw.write("\n" + dataset);
			for (String key : keys) {
				fw.write("," + statics.get(key));
			}

		}
		fw.close();


	}
}
