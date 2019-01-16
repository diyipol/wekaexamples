package com.autentia.wekaexamples.mains;

import com.autentia.wekaexamples.utils.DataSourceUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * <p>Recibe el fichero de entrenamiento y de tests como parámetros. Si no existe los crea con {@link weka.datagenerators.classifiers.classification.LED24}
 * con tamaño de medio giga para el de entrenamiento y de medio mega para el de testeo.</p>
 *
 * <p>Se ha comprobado que con estos tamaños J48 al no ser incremental falla. También se ha comprobado que también falla
 * el cross-validation por lo mismo, la validación debe ser con fichero de test a parte.</p>
 *
 */
public class BigData {

    public static void main(String [] args) throws Exception {

        if (args.length < 2) {
            throw new IllegalArgumentException("Se deben de pasar dos parámetros, uno con el path del fichero de " +
                    "entrenamiento y otro con el path del fichero de tests. Si los ficheros no existen se crearán con " +
                    "el generador LED24 y tamaños ??¿¿?");
        }

        String trainFilePath = args[0];
        String testFilePath = args[1];

        File trainFile = getTrainFile(trainFilePath);
        File testFile = getTestFile(testFilePath);

        DataSourceUtils dataSourceUtils = new DataSourceUtils();
        dataSourceUtils.newWekaInstances(trainFile);

        System.out.println("Cargadas las instancias del fichero de entrenamiento y testeo.");

        ArffLoader loader = new ArffLoader();
        loader.setFile(trainFile);
        Instances trainInstances = loader.getStructure();
        trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

        NaiveBayesUpdateable naiveBayesUpdatable = new NaiveBayesUpdateable();
        naiveBayesUpdatable.buildClassifier(trainInstances);

        System.out.println("Comenzamos a entrenar el modelo con Naive Bayes.");

        Instance instance;

        while ((instance = loader.getNextInstance(trainInstances))  != null) {
            naiveBayesUpdatable.updateClassifier(instance);
        }

        System.out.println("\nComenzamos a testear el modelo con Naive Bayes.");

        Instances testInstances = dataSourceUtils.newWekaInstances(testFile);
        Evaluation evaluation = new Evaluation(testInstances);

        evaluation.evaluateModel(naiveBayesUpdatable, testInstances);

        boolean printComplexityStatistics = false;
        System.out.println(evaluation.toSummaryString(printComplexityStatistics));

        System.out.println(evaluation.toMatrixString());

        /* ¡¡¡90 minutos para evaluar el IBK!!!

        IBk iBk = new IBk();
        iBk.buildClassifier(trainInstances);
        //iBk.setKNN(5);

        System.out.println("Comenzamos a entrenar el modelo con K-nearest neighbours.");

        loader.reset();
        trainInstances = loader.getStructure();
        trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

        long currentTimeMillis = System.currentTimeMillis();

        while ((instance = loader.getNextInstance(trainInstances))  != null) {
            iBk.updateClassifier(instance);
        }

        System.out.println("Modelo entrenado en [" + ((System.currentTimeMillis() - currentTimeMillis) / 1000)/60 + "] minutos.");

        currentTimeMillis = System.currentTimeMillis();

        evaluation = new Evaluation(testInstances);

        evaluation.evaluateModel(iBk, testInstances);

        System.out.println("Modelo evaluado en [" + ((System.currentTimeMillis() - currentTimeMillis) / 1000)/60 + "] minutos.");

        System.out.println(evaluation.toSummaryString(printComplexityStatistics));

        System.out.println(evaluation.toMatrixString());
        */
    }

    private static File getTrainFile(String trainFilePath) throws Exception {

        File trainFile = new File(trainFilePath);

        if (!trainFile.exists()) {
            System.out.println("No existe el fichero de entrenamiento [" + trainFilePath + "]. " +
                    "Se autogenerará uno con 10 millones de instancias.");
            trainFile = DataSourceUtils.createLed24DataSource(trainFilePath, 10000000);
        }

        return trainFile;
    }

    private static File getTestFile(String testFilePath) throws Exception {

        File testFile = new File(testFilePath);

        if (!testFile.exists()) {
            System.out.println("No existe el fichero de testeo [" + testFilePath + "]. " +
                    "Se autogenerará uno con 10 mil instancias.");
            testFile = DataSourceUtils.createLed24DataSource(testFilePath, 10000);
        }

        return testFile;
    }

}
