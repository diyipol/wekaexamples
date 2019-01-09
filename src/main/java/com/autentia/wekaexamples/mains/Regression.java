package com.autentia.wekaexamples.mains;

import com.autentia.wekaexamples.utils.DataSourceUtils;
import com.autentia.wekaexamples.utils.FileUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.util.Random;

/**
 * Diferentes ejemplos simples de regresión
 */
public class Regression {

    public static void main(String [] args) throws Exception {

        FileUtils fileUtils = new FileUtils();
        File file = fileUtils.getFile("housing.arff");

        DataSourceUtils dataSourceUtils = new DataSourceUtils();
        Instances instances = dataSourceUtils.newWekaInstances(file);

        System.out.println(instances.numInstances() + " instancias cargadas.");

        LinearRegression linearRegression = new LinearRegression();
        linearRegression.buildClassifier(instances);

        System.out.println(linearRegression);

        Evaluation linearRegressionEvaluation = new Evaluation(instances);
        int numFolds = 10;
        Random random = new Random(1);
        linearRegressionEvaluation.crossValidateModel(linearRegression, instances, numFolds, random, new Object[] {});
        System.out.println(linearRegressionEvaluation.toSummaryString());

        M5P regressionTree = new M5P();
        regressionTree.setOptions(new String[]{""});
        regressionTree.buildClassifier(instances);
        System.out.println(regressionTree);

        Evaluation regressionTreeEvaluation = new Evaluation(instances);
        regressionTreeEvaluation.crossValidateModel(regressionTree, instances, numFolds, random, new Object[] {});
        System.out.println(regressionTreeEvaluation.toSummaryString());
    }
}
