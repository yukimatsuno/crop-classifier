# Crop Classifier
Crop classifier using machine learning

# Overview
Machine learning model which classifies satelite imagery between rice fields and sugarcane fields.

## Data
- 10,000 segmented satelite imagery taken by Sentinel 2
- Rice field in Vietnam vs. sugarcane field in Thailand
- Imagery has 13 bands: visible (red, green, blue), NIR, red edge, SWIR, and atmospheric bands
- Sample of field imagery (Image of each band and numeric info)
![image](https://user-images.githubusercontent.com/79320522/134799239-12eb3860-e619-4679-b4a1-162caeac6c4a.png)

## Feture Engineering
Combined bands numeric data following the link : [gisgeography](https://gisgeography.com/sentinel-2-bands-combinations/)

## Learning curve
![image](https://user-images.githubusercontent.com/79320522/134799508-49db46c4-dcce-4f5c-8d3b-f8ac83c853cd.png)

## Model
- Achieved accuracy of 95%
- Train method: KNN

## App
[SKY CROP] (http://skycrop.herokuapp.com/)
