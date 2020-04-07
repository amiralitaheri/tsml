package MetaNet;

import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

import static experiments.data.DatasetLists.UCIContinuousWithoutBigFour;

public class Optimizer {

    static String resultPath = "F:/University Files/Project/Result/";
    static String[] classifiersName = new String[]{"Logistic", "C45", "SVML", "NN", "MLP"};
    //    static String[] datasetNames = UCIContinuousWithoutBigFour;
//    static String[] datasetNames = new String[]{"abalone"};
    static String[] datasetNames = new String[]{"acute-inflammation", "acute-nephritis", "annealing", "arrhythmia", "audiology-std", "balance-scale", "balloons", "bank", "blood", "breast-cancer", "breast-cancer-wisc", "breast-cancer-wisc-diag", "breast-cancer-wisc-prog", "breast-tissue", "cardiotocography-10clases", "cardiotocography-3clases",
            "chess-krvkp", "congressional-voting", "conn-bench-sonar-mines-rocks", "conn-bench-vowel-deterding",
            "connect-4", "contrac", "credit-approval", "cylinder-bands", "dermatology", "echocardiogram", "ecoli", "energy-y1", "energy-y2", "fertility", "flags", "glass", "haberman-survival", "hayes-roth", "heart-cleveland", "heart-hungarian", "heart-switzerland", "heart-va", "hepatitis", "hill-valley", "horse-colic", "ilpd-indian-liver", "image-segmentation", "ionosphere", "iris", "led-display", "lenses", "letter", "libras", "low-res-spect", "lung-cancer", "lymphography", "mammographic",
            "molec-biol-promoter", "molec-biol-splice", "monks-1", "monks-2", "monks-3", "mushroom", "musk-1", "musk-2", "nursery", "oocytes_merluccius_nucleus_4d", "oocytes_merluccius_states_2f", "oocytes_trisopterus_nucleus_2f", "oocytes_trisopterus_states_5b", "optical", "ozone", "page-blocks", "parkinsons", "pendigits", "pima", "pittsburg-bridges-MATERIAL", "pittsburg-bridges-REL-L", "pittsburg-bridges-SPAN", "pittsburg-bridges-T-OR-D", "pittsburg-bridges-TYPE", "planning", "plant-margin", "plant-shape", "plant-texture", "post-operative", "primary-tumor", "ringnorm", "seeds", "semeion", "soybean", "spambase", "spect", "spectf", "statlog-australian-credit", "statlog-german-credit", "statlog-heart", "statlog-image", "statlog-landsat", "statlog-shuttle", "statlog-vehicle", "steel-plates", "synthetic-control", "teaching", "thyroid", "tic-tac-toe", "titanic", "trains", "twonorm", "vertebral-column-2clases", "vertebral-column-3clases", "wall-following", "waveform", "waveform-noise", "wine", "wine-quality-red", "wine-quality-white", "yeast", "zoo"};
    static int batchSize = 50;
    static int fold = 1;
    static int numOfEpochs = 50;

    public static void main(String[] args) {
        for (String datasetName : datasetNames) {
            MetaNet basic = new MetaNet(classifiersName, datasetName, batchSize, fold);
            basic.createData();
        }
        for (String datasetName : datasetNames) {
            try {
                optimizeLearningRateAndLayerSize(classifiersName, datasetName, fold);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    static void optimizeLearningRateAndLayerSize(String[] classifiersName, String datasetName, int fold) throws Exception {

        StringBuilder sb = new StringBuilder();
        for (String name : classifiersName) {
            sb.append(name).append("_");
        }
        String trainFileName = resultPath + "MetaNet/train/" + sb + datasetName + fold + ".csv";
        File trainFile = new File(trainFileName);


        Scanner scanner = new Scanner(trainFile);
        int numClasses = scanner.nextInt();
        int labelIndex = numClasses * classifiersName.length;

        ContinuousParameterSpace learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);
        IntegerParameterSpace layerSizeHyperparam = new IntegerParameterSpace(16, 256);

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                //These next few options: fixed values for all models
                .seed(fold)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                //Learning rate hyperparameter: search over different values, applied to all models
                .updater(new SgdSpace(learningRateHyperparam))
                .addLayer(new DenseLayerSpace.Builder()
                        //Fixed values for this layer:
                        .nIn(labelIndex)
                        //One hyperparameter to infer: layer size
                        .nOut(layerSizeHyperparam)
                        .build())
                .addLayer(new OutputLayerSpace.Builder()
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .numEpochs(numOfEpochs)
                .build();


        RandomSearchGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);

        EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);
        TerminationCondition[] terminationConditions = {/*new MaxTimeCondition(15, TimeUnit.MINUTES),*/new MaxCandidatesCondition(100)};

        Properties datasetProp = new Properties();
        datasetProp.setProperty("datasetName", datasetName);

        String baseSaveDirectory = resultPath + "MetaNet/Optimizer/" + datasetName;
        File f = new File(baseSaveDirectory);
        if (f.exists()) f.delete();
        f.mkdir();
        FileModelSaver modelSaver = new FileModelSaver(baseSaveDirectory);
        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataSource(DataSourceMetaNet.class, datasetProp)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        LocalOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());
        //Start the hyperparameter optimization
        runner.execute();

        String s = "Best score: " + runner.bestScore() + "\n" + "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" + "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);

        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();
        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();
        FileWriter fw = new FileWriter(resultPath + "MetaNet/Optimizer/" + datasetName + "/bestModel.json");
        fw.write(bestModel.getLayerWiseConfigurations().toJson());
        fw.close();
        fw = new FileWriter(resultPath + "MetaNet/Optimizer/" + datasetName + "/indexBest.txt");
        fw.write(String.valueOf(indexOfBestResult));
        fw.close();
    }

}

