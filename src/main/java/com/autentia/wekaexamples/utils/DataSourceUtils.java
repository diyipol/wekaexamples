package com.autentia.wekaexamples.utils;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;

public class DataSourceUtils {

    public Instances newWekaInstances(File file) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getAbsolutePath());
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }
}
