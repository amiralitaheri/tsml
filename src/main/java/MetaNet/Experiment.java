package MetaNet;

import java.io.File;
import java.io.FileWriter;
import java.util.HashMap;

import static experiments.data.DatasetLists.UCIContinuousWithoutBigFour;

public class Experiment {
    public static String[] standard = {
            "XGBoostMultiThreaded", "XGBoost", "SmallTunedXGBoost", "RandF", "RotF", "PLSNominalClassifier", "BayesNet", "ED", "C45",
            "SVML", "SVMQ", "SVMRBF", "MLP", "Logistic", "CAWPE", "NN"};

    private static String[] basicClassifiers = new String[]{"Logistic", "C45", "SVML", "NN", "MLP"};

    public static void main(String[] args) {
        HashMap<String, Double> results = new HashMap();
        for (String dataset : UCIContinuousWithoutBigFour) {
            MetaNet basic = new MetaNet(basicClassifiers, dataset, 50, 1);
            basic.createData();
            double r = -1;
            try {
                r = basic.runExperiment();
            } catch (Exception e) {
                e.printStackTrace();
            }
            results.put(dataset, r);
        }
        StringBuilder sb = new StringBuilder();
        for (String name : basicClassifiers) {
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
