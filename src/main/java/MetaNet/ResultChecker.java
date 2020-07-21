package MetaNet;


import java.io.File;

import static experiments.data.DatasetLists.UCIContinuousFileNames;

public class ResultChecker {
	public static void main(String[] args) {
		String[] basicClassifiers = {"Logistic", "C45", "SVML", "NN", "MLP"};
		for (String classifierName : basicClassifiers) {
			for (String dataset : UCIContinuousFileNames) {
				for (int fold = 0; fold < 30; fold++) {
					try {
						File file = new File("F:/University Files/Project/Result/" + classifierName + "/Predictions/" + dataset + "/testFold" + fold + ".csv");
						if (!file.exists()) {
							System.out.println(file.getAbsolutePath());
						} else if (file.length() == 0) {
							System.out.println("file is empty");
							System.out.println(file.getAbsolutePath());
						}
					} catch (Exception e) {
						System.out.println(e.getMessage());
					}
					try {
						File file = new File("F:/University Files/Project/Result/" + classifierName + "/Predictions/" + dataset + "/trainFold" + fold + ".csv");
						if (!file.exists()) {
							System.out.println(file.getAbsolutePath());
						} else if (file.length() == 0) {
							System.out.println("file is empty");
							System.out.println(file.getAbsolutePath());
						}
					} catch (Exception e) {
						System.out.println(e.getMessage());
					}
				}
			}
		}
	}
}
