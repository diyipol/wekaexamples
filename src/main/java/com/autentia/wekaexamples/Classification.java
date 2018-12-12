package com.autentia.wekaexamples;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.net.URL;

public class Classification {

    public static void main(String [] args) throws Exception {
        Classification classification = new Classification();
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(classification.getFile("weather.arff"));
        Instances data = source.getDataSet();
        System.out.println(data.numInstances() + " instances loaded.");
        System.out.println(data.toString());
    }

    private String getFile(String fileName) {
        ClassLoader classLoader = getClass().getClassLoader();
        URL resource = classLoader.getResource(fileName);
        File file = new File(resource.getFile());
        return file.getAbsolutePath();
    }
}
