package com.autentia.wekaexamples.mains;

import com.autentia.wekaexamples.utils.DataSourceUtils;
import com.autentia.wekaexamples.utils.FileUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;

/**
 * Ejemplo con un fichero de prueba suministrado por separado al de entrenamiento.
 */
public class TestDataset {

    private final static String CONFIDENCE_PRUNING_THRESHOLD = "-C";
    private final static String CONFIDENCE_PRUNING_THRESHOLD_VALUE = "0.25";
    private final static String MIN_LEAF_INSTANCES = "-M";
    private final static String MIN_LEAF_INSTANCES_VALUE = "2";

    private final static String[] J48_OPTIONS = {CONFIDENCE_PRUNING_THRESHOLD, CONFIDENCE_PRUNING_THRESHOLD_VALUE,
            MIN_LEAF_INSTANCES, MIN_LEAF_INSTANCES_VALUE};

    public static void main(String [] args) throws Exception {

        FileUtils fileUtils = new FileUtils();
        File file = fileUtils.getFile("segment-challenge.arff");

        DataSourceUtils dataSourceUtils = new DataSourceUtils();
        Instances trainInstances = dataSourceUtils.newWekaInstances(file);

        File testFile = fileUtils.getFile("segment-test.arff");;
        Instances testInstances = dataSourceUtils.newWekaInstances(testFile);

        J48 treeClassifier = new J48();
        treeClassifier.setOptions(J48_OPTIONS);
        treeClassifier.buildClassifier(trainInstances);

        Evaluation evaluation = new Evaluation(testInstances);


        evaluation.evaluateModel(treeClassifier, testInstances);

        boolean printComplexityStatistics = false;
        System.out.println(evaluation.toSummaryString(printComplexityStatistics));

        System.out.println(evaluation.toMatrixString());
    }
}
