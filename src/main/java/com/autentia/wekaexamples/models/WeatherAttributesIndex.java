package com.autentia.wekaexamples.models;

public enum WeatherAttributesIndex {
    OUTLOOK (0),
    TEMPERATURE (1),
    HUMIDITY (2),
    WINDY (3),
    PLAY (4);

    private int index;

    WeatherAttributesIndex(int index) {
        this.index = index;
    }

    public static WeatherAttributesIndex newInstance(int index) {
        for (WeatherAttributesIndex weatherAttributesIndex : WeatherAttributesIndex.values()) {
            if (index == weatherAttributesIndex.index) {
                return weatherAttributesIndex;
            }
        }
        throw new IllegalArgumentException("Index not found");
    }

    public int getIndex() {
        return index;
    }
}
