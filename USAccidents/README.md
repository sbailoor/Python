# Project: US Traffic Accidents Analysis

## Overview

This project analyzes the US Accidents dataset to identify patterns, trends, and factors contributing to traffic accidents across the United States. The goal is to provide data-driven insights and actionable recommendations to the Department of Transportation (DOT) to reduce accidents and improve road safety.

## Dataset

The dataset used in this project is the US Accidents dataset downloaded from Kaggle, specifically the "US_Accidents_March23.csv" file. It contains detailed information about millions of traffic accidents recorded across the United States.

## Business Case

The Department of Transportation (DOT) aims to develop strategies to reduce traffic accidents and enhance road safety nationwide. This analysis provides insights to support data-driven decision-making in achieving this goal.

## Project Goals

* Analyze the US Accidents dataset to identify patterns, trends, and contributing factors.
* Provide three data-driven insights that the DOT can utilize to reduce traffic accidents and improve road safety.

## Analytical Questions Explored

1. Where are the most accident-prone locations in the United States?
2. What are the most common causes or contributing factors to accidents?
3. How do accident rates vary by time (hour, day, month, season)?
4. What is the relationship between weather conditions and accident severity?
5. Are there trends in accident severity based on location or time?
6. How do traffic accidents impact traffic congestion and travel times?
7. Can we predict the likelihood of an accident occurring at a given time and place?

## Project Steps

The project followed these key steps:

1.  **Setup the Project**: Downloaded the dataset using Kagglehub.
2.  **Business Understanding**: Defined the business case, goals, data relevancy, analytical questions, benefits, and stakeholders.
3.  **Data Understanding**: Loaded the dataset, explored its structure and basic statistics, documented variables, and assessed data quality issues (missing values, outliers, inconsistencies).
4.  **Data Preparation**: Handled missing values by dropping columns with high missing percentages and imputing or removing missing values in other relevant columns. Converted data types and created derived features (time-based features, accident duration).
5.  **Exploratory Data Analysis (EDA)**: Conducted visual and statistical analysis to identify patterns in accident locations, temporal trends, weather influences, severity trends, and the distribution of binary features.
6.  **Statistical Analysis**: Identified patterns from EDA and performed statistical tests (Chi-square tests, ANOVA) to validate observed relationships between variables.
7.  **Insights and Recommendations**: Developed actionable recommendations for the DOT based on the key findings and validated patterns from the analysis.
8.  **Dashboard**: Created a Tableau dashboard to visualize key findings.

## Key Findings

*   Certain states and cities have significantly higher accident frequencies (e.g., California, Florida, Texas, Miami, Houston, Los Angeles).
*   Accidents show clear temporal patterns with peaks during weekday rush hours.
*   Weather conditions are statistically associated with accident severity.
*   Accident severity distribution varies by location and time.
*   Locations with traffic signals, junctions, and crossings are associated with a notable percentage of accidents.

## Actionable Recommendations

1.  **Targeted Interventions in High-Accident Locations**: Focus resources and safety measures on states and cities with the highest accident rates and varying severity distributions.
2.  **Time-Based Safety Measures**: Implement specific safety measures during weekday rush hours to mitigate the higher accident frequency during these periods.
3.  **Enhanced Focus on Weather-Related Accident Prevention**: Improve weather monitoring, warnings, and public education to address the association between weather conditions and accident severity.

## Tableau Dashboard

A comprehensive dashboard visualizing the key findings of this analysis is available on Tableau Public:

[Tableau Report Link](https://public.tableau.com/app/profile/sri.bailoor/viz/US_Accidents_Analysis_17573366131380/MainPage_1)

## How to Reproduce the Analysis

To reproduce this analysis, you will need:

*   Access to a Python environment with the necessary libraries installed (pandas, numpy, matplotlib, seaborn, scipy).
*   The US Accidents dataset ("US_Accidents_March23.csv") downloaded from Kaggle. The notebook includes code to download the dataset using `kagglehub`.
*   The provided Jupyter Notebook containing the code for data loading, preparation, EDA, statistical analysis, and generating insights.

Simply run the cells in the notebook sequentially to execute the analysis steps.
