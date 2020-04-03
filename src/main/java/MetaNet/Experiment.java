package MetaNet;

public class Experiment {
    public static String[] standard = {
            "XGBoostMultiThreaded", "XGBoost", "SmallTunedXGBoost", "RandF", "RotF", "PLSNominalClassifier", "BayesNet", "ED", "C45",
            "SVML", "SVMQ", "SVMRBF", "MLP", "Logistic", "CAWPE", "NN"};

    public static void main(String[] args) {
        MetaNet basic = new MetaNet(new String[]{"Logistic", "C45", "SVML", "NN", "MLP"}, "car", 50, 1);
        basic.createData();
        try {
            basic.runExperiment();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
