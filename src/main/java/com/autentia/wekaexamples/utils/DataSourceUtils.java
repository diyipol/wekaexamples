package com.autentia.wekaexamples.utils;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.datagenerators.classifiers.classification.LED24;

import java.io.File;

public class DataSourceUtils {

    public static File createLed24DataSource(String path, int numExamples) throws Exception {

        String optionsStr = new StringBuilder("-o ").append(path)
                .append(" -n ").append(numExamples)
                .toString();

        String[] options = Utils.splitOptions(optionsStr);

        LED24.runDataGenerator(new LED24(), options);

        return new File(path);
    }

    public Instances newWekaInstances(File file) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getAbsolutePath());
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

}
