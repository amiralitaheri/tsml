package MetaNet;

import experiments.Experiments;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
    String[] classifiersName;
    String datesetName;
    int batchSize;
    int fold;
    String resultPath = "F:/University Files/Project/Result/";
    String trainFileName;
    String testFileName;

    public MetaNet(String[] classifiersName, String datesetName, int batchSize, int fold) {
        Arrays.sort(classifiersName);
        this.classifiersName = classifiersName;
        this.datesetName = datesetName;
        this.batchSize = batchSize;
        this.fold = fold;

    }

    void createData() {
        //run experiments
        String[] settings = new String[4];
        settings[0] = "-dp=F:/University Files/Project/UCIContinuous/";//Where to get data
        settings[1] = "-rp=" + resultPath;//Where to write results
        settings[2] = "-gtf=true"; //Whether to generate train files or not
        settings[3] = "1";
        Experiments.ExperimentalArguments expSettings = null;
        try {
            expSettings = new Experiments.ExperimentalArguments(settings);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Threaded experiment with " + expSettings);
        try {
            setupAndRunMultipleExperimentsThreaded(expSettings, classifiersName, new String[]{datesetName}, fold, fold);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        //merge output files to one csv
        StringBuilder sb = new StringBuilder();
        for (String name : classifiersName) {
            sb.append(name).append("_");
        }
        trainFileName = resultPath + "MetaNet/train/" + sb + datesetName + fold + ".csv";
        testFileName = resultPath + "MetaNet/test/" + sb + datesetName + fold + ".csv";
        File train = new File(trainFileName);
        File test = new File(testFileName);
        if (!train.exists() || train.length() == 0) {
            try {
                mergeFiles(classifiersName, datesetName, "train", train);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (!test.exists() || test.length() == 0) {
            try {
                mergeFiles(classifiersName, datesetName, "test", test);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    void runExperiment() throws Exception {
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
        DataSet trainingData = trainIterator.next();

        RecordReader testRecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        testRecordReader.initialize(new FileSplit(testFile));
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSize, labelIndex, numClasses);
        DataSet testData = testIterator.next();


        //todo optimize
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(fold)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(labelIndex).nOut(50)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(50).nOut(numClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));


        for (int i = 0; i < 1000; i++) {
            model.fit(trainingData);
        }

        Evaluation eval = new Evaluation(numClasses);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        logResult(eval.stats());
        //todo save the model
    }

    private void logResult(String stats) {
        System.out.println(stats);
        //todo save file
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
        fw.write(numOfClasses);
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
    }
}
