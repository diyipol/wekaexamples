package com.autentia.wekaexamples.models;

public enum PlayAttributeValues {
    YES (0),
    NO (1);

    private int index;

    PlayAttributeValues(int index) {
        this.index = index;
    }

    public static PlayAttributeValues newInstance(int index) {
        for (PlayAttributeValues playAttributeValues : PlayAttributeValues.values()) {
            if (index == playAttributeValues.index) {
                return playAttributeValues;
            }
        }
        throw new IllegalArgumentException("Index not found");
    }

    public int getIndex() {
        return index;
    }
}
