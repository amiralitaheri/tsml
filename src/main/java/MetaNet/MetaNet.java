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
import java.util.Arrays;

import static experiments.Experiments.setupAndRunMultipleExperimentsThreaded;

public class MetaNet {
    String[] classifiersName;
    String datesetName;
    int batchSize;
    int fold;
    String resultPath = "F:/University Files/Project/tsml/results/";
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
        trainFileName = resultPath + "MetaNet/train/" + sb + datesetName + ".csv";
        testFileName = resultPath + "MetaNet/test/" + sb + datesetName + ".csv";
        File train = new File(trainFileName);
        File test = new File(testFileName);
        if (!train.exists() || train.length() == 0) {
            mergeFiles(classifiersName, datesetName, "train");
        }
        if (!test.exists() || test.length() == 0) {
            mergeFiles(classifiersName, datesetName, "test");
        }
    }

    void runExperiment() throws Exception {
        int numLinesToSkip = 1;
        char delimiter = ',';
        File trainFile = new File(trainFileName);
        File testFile = new File(testFileName);


        int labelIndex = 0; //todo read from the first line
        int numClasses = 0;  //todo


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
    }

    private void logResult(String stats) {
        System.out.println(stats);
        //todo save file
    }

    private void mergeFiles(String[] classifiersName, String datesetName, String test) {
        //todo
    }
}
