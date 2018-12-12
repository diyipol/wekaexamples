package com.autentia.wekaexamples;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.net.URL;

public class Classification {

    public static final String TREE_UNPRUNED_OPT = "-U";

    public static void main(String [] args) throws Exception {

        Classification classification = new Classification();
        File file = classification.getFile("weather.arff");

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getAbsolutePath());
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);

        System.out.println(instances.numInstances() + " instancias cargadas.");
        System.out.println(instances.toString());

        J48 tree = new J48();
        String[] options = new String[1];
        options[0] = TREE_UNPRUNED_OPT;

        tree.setOptions(options);

        tree.buildClassifier(instances);

        System.out.println(tree);
    }

    private File getFile(String fileName) {
        ClassLoader classLoader = getClass().getClassLoader();
        URL resource = classLoader.getResource(fileName);
        return new File(resource.getFile());
    }
}
