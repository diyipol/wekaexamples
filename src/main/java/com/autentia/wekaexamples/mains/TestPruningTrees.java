package com.autentia.wekaexamples.mains;

import com.autentia.wekaexamples.utils.DataSourceUtils;
import com.autentia.wekaexamples.utils.FileUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

import java.io.File;
import java.util.Random;

/**
 * Ejemplo extremo de diferencia de podado y no podado.
 */
public class TestPruningTrees {

    private final static String CONFIDENCE_PRUNING_THRESHOLD = "-C";
    private final static String CONFIDENCE_PRUNING_THRESHOLD_VALUE = "0.25";
    private final static String MIN_LEAF_INSTANCES = "-M";
    private final static String MIN_LEAF_INSTANCES_VALUE = "2";

    private final static String[] J48_OPTIONS = {CONFIDENCE_PRUNING_THRESHOLD, CONFIDENCE_PRUNING_THRESHOLD_VALUE,
            MIN_LEAF_INSTANCES, MIN_LEAF_INSTANCES_VALUE};

    public static void main(String [] args) throws Exception {

        FileUtils fileUtils = new FileUtils();
        File file = fileUtils.getFile("breast-cancer.arff");

        DataSourceUtils dataSourceUtils = new DataSourceUtils();
        Instances instances = dataSourceUtils.newWekaInstances(file);

        J48 prunedTreeClassifier = new J48();
        prunedTreeClassifier.setOptions(J48_OPTIONS);
        prunedTreeClassifier.buildClassifier(instances);

        System.out.println("\nPruned tree\n===");
        System.out.println(prunedTreeClassifier);

        Evaluation treeEvaluation = new Evaluation(instances);
        int numFolds = 5;
        Random random = new Random(1);
        treeEvaluation.crossValidateModel(prunedTreeClassifier, instances, numFolds, random, new Object[] {});
        System.out.println(treeEvaluation.toSummaryString());

        String classifierOptions[] = Utils.splitOptions("-U");

        J48 unprunedTreeClassifier = new J48();
        unprunedTreeClassifier.setOptions(classifierOptions);
        unprunedTreeClassifier.buildClassifier(instances);

        System.out.println("\nUnpruned tree\n===");
        System.out.println(unprunedTreeClassifier);

        treeEvaluation.crossValidateModel(unprunedTreeClassifier, instances, numFolds, random, new Object[] {});
        System.out.println(treeEvaluation.toSummaryString());

    }
}
