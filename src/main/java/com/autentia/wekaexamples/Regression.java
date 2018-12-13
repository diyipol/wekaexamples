package com.autentia.wekaexamples;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.util.Random;

public class Regression {

    public static void main(String [] args) throws Exception {

        FileUtils fileUtils = new FileUtils();
        File file = fileUtils.getFile("housing.arff");

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getAbsolutePath());
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);

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
