package com.autentia.wekaexamples.utils;

import java.io.File;
import java.net.URL;

public class FileUtils {

    public File getFile(String fileName) {
        ClassLoader classLoader = getClass().getClassLoader();
        URL resource = classLoader.getResource(fileName);
        return new File(resource.getFile());
    }
}
