package MetaNet;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;
import java.util.Scanner;

public class DataSourceMetaNet implements DataSource {

    DataSetIterator trainIterator;
    DataSetIterator testIterator;

    public DataSourceMetaNet() {

    }

    @Override
    public void configure(Properties properties) {
        int numLinesToSkip = 1;
        char delimiter = ',';
        StringBuilder sb = new StringBuilder();
        for (String name : Optimizer.classifiersName) {
            sb.append(name).append("_");
        }
        String trainFileName = Optimizer.resultPath + "MetaNet/train/" + sb + properties.getProperty("datasetName") + Optimizer.fold + ".csv";
        String testFileName = Optimizer.resultPath + "MetaNet/test/" + sb + properties.getProperty("datasetName") + Optimizer.fold + ".csv";
        File trainFile = new File(trainFileName);
        File testFile = new File(testFileName);


        Scanner scanner = null;
        try {
            scanner = new Scanner(trainFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        int numClasses = scanner.nextInt();
        int labelIndex = numClasses * Optimizer.classifiersName.length;


        RecordReader trainRecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        try {
            trainRecordReader.initialize(new FileSplit(trainFile));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        trainIterator = new RecordReaderDataSetIterator(trainRecordReader, Optimizer.batchSize, labelIndex, numClasses);

        RecordReader testRecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        try {
            testRecordReader.initialize(new FileSplit(testFile));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        testIterator = new RecordReaderDataSetIterator(testRecordReader, Optimizer.batchSize, labelIndex, numClasses);
    }

    @Override
    public Object trainData() {
        return trainIterator;
    }

    @Override
    public Object testData() {
        return testIterator;
    }

    @Override
    public Class<?> getDataType() {
        return DataSetIterator.class;
    }
}
