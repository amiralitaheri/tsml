package MetaNet;

import java.io.File;
import java.io.FileWriter;
import java.util.HashMap;

public class Experiment {
    public static String[] skipDataSets = {"plant-margin", "plant-shape", "plant-texture", "connect-4"};
    public static String[] standard = {
            "XGBoostMultiThreaded", "XGBoost", "SmallTunedXGBoost", "RandF", "RotF", "PLSNominalClassifier", "BayesNet", "ED", "C45",
            "SVML", "SVMQ", "SVMRBF", "MLP", "Logistic", "CAWPE", "NN"};

    private static String[] basicClassifiers = {"Logistic", "C45", "SVML", "NN", "MLP"};
    private static String[] advanceClassifiers = {"MLP2", "XGBoost", "SVMQ", "RandF", "RotF"};

    public static void main(String[] args) {
        String[] classifiers = basicClassifiers;
        HashMap<String, Double> results = new HashMap();
        for (String dataset : new String[]{"bank"}) {
            boolean flag = true;
            for (String s : skipDataSets) {
                if (dataset.equals(s)) {
                    flag = false;
                }
            }
            double r = -1;
            if (flag) {
                MetaNet basic = new MetaNet(classifiers, dataset, 50, 1);
                basic.createData();
                try {
                    r = basic.runExperiment();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            results.put(dataset, r);
        }
        StringBuilder sb = new StringBuilder();
        for (String name : classifiers) {
            sb.append(name).append("_");
        }
        String classifiersInString = sb.toString();
        File saveFile = new File("F:/University Files/Project/Result/MetaNet/result/" + classifiersInString);
        if (saveFile.exists()) {
            saveFile.delete();
        }
        saveFile.mkdirs();
        saveFile = new File("F:/University Files/Project/Result/MetaNet/result/" + classifiersInString + "/result.csv");
        try {
            FileWriter fw = new FileWriter(saveFile);
            for (String key : results.keySet()) {
                fw.write(key + "," + results.get(key));
                fw.write(System.lineSeparator());
            }
            fw.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

    }
}
