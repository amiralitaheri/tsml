package MetaNet;

import experiments.Experiments;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.ConfusionMatrix;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import static experiments.Experiments.setupAndRunMultipleExperimentsThreaded;

public class MetaNet {
    private String[] classifiersName;
    private String datasetName;
    private int batchSize;
    private int fold;
    private String resultPath = "F:/University Files/Project/Result/";
    private String classifiersInString = "";
    private String trainFileName;
    private String testFileName;

    public MetaNet(String[] classifiersName, String datesetName, int batchSize, int fold) {
        Arrays.sort(classifiersName);
        this.classifiersName = classifiersName;
        this.datasetName = datesetName;
        this.batchSize = batchSize;
        this.fold = fold;

    }

    void createData() {
        //run experiments
        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();
        expSettings.dataReadLocation = "F:/University Files/Project/UCIContinuous/";
        expSettings.resultsWriteLocation = resultPath;
        expSettings.generateErrorEstimateOnTrainSet = true;
//        expSettings.forceEvaluation = true;
        System.out.println("Threaded experiment with " + expSettings);
        try {
            setupAndRunMultipleExperimentsThreaded(expSettings, classifiersName, new String[]{datasetName}, fold - 1, fold);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        //merge output files to one csv
        StringBuilder sb = new StringBuilder();
        for (String name : classifiersName) {
            sb.append(name).append("_");
        }
        classifiersInString = sb.toString();
        trainFileName = resultPath + "MetaNet/train/" + sb + datasetName + fold + ".csv";
        testFileName = resultPath + "MetaNet/test/" + sb + datasetName + fold + ".csv";
        File train = new File(trainFileName);
        File test = new File(testFileName);
        if (!train.exists() || train.length() == 0) {
            try {
                mergeFiles(classifiersName, datasetName, "train", train);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (!test.exists() || test.length() == 0) {
            try {
                mergeFiles(classifiersName, datasetName, "test", test);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    double runExperiment() throws Exception {
        int numLinesToSkip = 1;
        char delimiter = ',';
        File trainFile = new File(trainFileName);
        File testFile = new File(testFileName);


        Scanner scanner = new Scanner(trainFile);
        int numClasses = scanner.nextInt();
        int labelIndex = numClasses * classifiersName.length;


        RecordReader trainRecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        trainRecordReader.initialize(new FileSplit(trainFile));
        DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainRecordReader, batchSize, labelIndex, numClasses);

        RecordReader testRecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        testRecordReader.initialize(new FileSplit(testFile));
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSize, labelIndex, numClasses);


        //todo optimize
        int midLayerSize = (numClasses + labelIndex) / 2;
//        int midLayerSize = 500;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(fold)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.03))
//                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(labelIndex).nOut(midLayerSize)
                        .build())
//                .layer(new DenseLayer.Builder().nIn(midLayerSize).nOut(midLayerSize).activation(Activation.LEAKYRELU)
//                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(midLayerSize).nOut(numClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //model.setListeners(new ScoreIterationListener(10));

        MultiLayerNetwork bestModel = null;
        double bestAcc = -1;
        for (int i = 0; i < 1000; i++) {
            trainIterator.reset();
            while (trainIterator.hasNext()) {
                model.fit(trainIterator);
            }
            Evaluation eval = model.evaluate(testIterator);
            if (eval.accuracy() > bestAcc) {
                bestModel = model.clone();
                bestAcc = eval.accuracy();
            }
        }

        Evaluation eval = bestModel.evaluate(testIterator);

        // calculate negative log likelihood
        testIterator.reset();
        int numOfExamples = 0;
        while (testIterator.hasNext()) {
            numOfExamples += testIterator.next().numExamples();
        }
        testIterator.reset();
        double nll = bestModel.score(testIterator.next(numOfExamples));


        // calculate balancedAccuracy
        double balancedAccuracy = calculateBalancedAccuracy(eval);

        // calculate Area Under ROC
        double auc = 0;
        ROCMultiClass roc = bestModel.evaluateROCMultiClass(testIterator);
        if (numClasses > 2) {
            for (int i = 0; i < numClasses; i++) {
                auc += roc.calculateAUC(i);
            }
            auc /= numClasses;
        } else {
            if (eval.getConfusionMatrix().getActualTotal(0) > eval.getConfusionMatrix().getActualTotal(1)) {
                auc = roc.calculateAUC(1);
            } else {
                auc = roc.calculateAUC(0);
            }
        }

        saveModel(bestModel);
        logResult(eval.stats(), nll, auc, balancedAccuracy);
        return eval.accuracy();
    }

    private double calculateBalancedAccuracy(Evaluation eval) {
        ConfusionMatrix confusionMatrix = eval.getConfusionMatrix();
        double sumAccuracy = 0;
        int count = confusionMatrix.getClasses().size();
        for (int i = 0; i < count; i++) {
            int sum = 0;
            for (int j = 0; j < count; j++) {
                sum += confusionMatrix.getCount(i, j);
            }
            sumAccuracy += confusionMatrix.getCount(i, i) * 1.0 / sum;
        }
        return sumAccuracy / count;
    }

    private void saveModel(MultiLayerNetwork model) throws IOException {
        File saveFile = new File(resultPath + "MetaNet/result/" + classifiersInString + "/" + datasetName);
        if (saveFile.exists()) {
            saveFile.delete();
        }
        saveFile.mkdirs();
        saveFile = new File(resultPath + "MetaNet/result/" + classifiersInString + "/" + datasetName + "/" + fold + ".model");
        model.save(saveFile);
    }

    private void logResult(String stats, double nll, double auc, double balancedAccuracy) throws IOException {
        System.out.println(stats);
        File statsFile = new File(resultPath + "MetaNet/result/" + classifiersInString + "/" + datasetName);
        if (statsFile.exists()) {
            statsFile.delete();
        }
        statsFile.mkdirs();
        statsFile = new File(resultPath + "MetaNet/result/" + classifiersInString + "/" + datasetName + "/stats_" + fold + ".txt");
        FileWriter fw = new FileWriter(statsFile);
        fw.write(stats);
        fw.write("\nnegetive log likelihood :" + nll + "\nAUC:" + auc + "\nbalanced accuracy:" + balancedAccuracy);
        fw.close();
    }

    private void mergeFiles(String[] classifiersName, String datesetName, String path, File out) throws IOException {
        int size = classifiersName.length;
        File[] files = new File[size];
        Scanner[] scanners = new Scanner[size];
        for (int i = 0; i < size; i++) {
            files[i] = new File(resultPath + classifiersName[i] + "/Predictions/" + datesetName + "/" + path + "Fold" + (fold - 1) + ".csv");
            scanners[i] = new Scanner(files[i]);
        }
        //reading number of classes and skip first three lines
        scanners[0].nextLine();
        scanners[0].nextLine();
        int numOfClasses = Integer.valueOf(scanners[0].nextLine().split(",")[5]);
        for (int i = 1; i < size; i++) {
            scanners[i].nextLine();
            scanners[i].nextLine();
            scanners[i].nextLine();
        }

        out.createNewFile();
        FileWriter fw = new FileWriter(out);
        fw.write(String.valueOf(numOfClasses));
        fw.write(System.lineSeparator());

        while (scanners[0].hasNextLine()) {
            StringBuilder sb = new StringBuilder();
            String[] values = null;
            for (Scanner scanner : scanners) {
                values = scanner.nextLine().split(",");
                for (int i = 3; i < 3 + numOfClasses; i++) {
                    sb.append(values[i]).append(",");
                }
            }
            sb.append(values[0]); //appending real class in the end
            fw.write(sb.toString());
            fw.write(System.lineSeparator());
        }
        fw.close();
    }

}
