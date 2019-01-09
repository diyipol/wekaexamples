package com.autentia.wekaexamples.mains;

import com.autentia.wekaexamples.utils.DataSourceUtils;
import com.autentia.wekaexamples.utils.FileUtils;
import com.autentia.wekaexamples.models.PlayAttributeValues;
import com.autentia.wekaexamples.models.WeatherAttributesIndex;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.util.Random;

/**
 * Diferentes ejemplos simples de clasificación
 */
public class Classification {

    public static final String TREE_UNPRUNED_OPT = "-U";

    public static void main(String [] args) throws Exception {

        FileUtils fileUtils = new FileUtils();
        File file = fileUtils.getFile("weather.arff");

        DataSourceUtils dataSourceUtils = new DataSourceUtils();
        Instances instances = dataSourceUtils.newWekaInstances(file);

        System.out.println(instances.numInstances() + " instancias cargadas.");
        System.out.println(instances.toString());

        J48 tree = new J48();
        String[] options = new String[1];
        options[0] = TREE_UNPRUNED_OPT;

        tree.setOptions(options);

        tree.buildClassifier(instances);

        System.out.println(tree);

        Instance instance = new DenseInstance(4);
        instance.setDataset(instances);
        instance.setValue(WeatherAttributesIndex.OUTLOOK.getIndex(), "sunny");
        instance.setValue(WeatherAttributesIndex.TEMPERATURE.getIndex(), 65);
        instance.setValue(WeatherAttributesIndex.HUMIDITY.getIndex(), 65);
        instance.setValue(WeatherAttributesIndex.WINDY.getIndex(), "TRUE");

        int result = (int) tree.classifyInstance(instance);

        System.out.println("Resultado de clasificar la nueva instancia: " + PlayAttributeValues.newInstance(result));

        Classifier treeClassifier = new J48();
        Evaluation treeEvaluation = new Evaluation(instances);
        int numFolds = 5;
        Random random = new Random(1);
        treeEvaluation.crossValidateModel(treeClassifier, instances, numFolds, random, new Object[] {});
        System.out.println(treeEvaluation.toSummaryString());

        double[][] confusionMatrix = treeEvaluation.confusionMatrix();
        System.out.println(treeEvaluation.toMatrixString());

        Classifier naiveBayesClassifier = new NaiveBayes();
        Evaluation naiveBayesEvaluation = new Evaluation(instances);
        naiveBayesEvaluation.crossValidateModel(naiveBayesClassifier, instances, numFolds, random, new Object[] {});
        System.out.println(naiveBayesEvaluation.toSummaryString());

    }

}
