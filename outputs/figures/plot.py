
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_distribution(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], kde=True, bins=50)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

def box_plot(df, y_column : None, x_column: None) -> None:
    plt.figure(figsize= (12, 10))
    sns.boxplot(data= df, y= y_column, x= x_column)
    plt.title(f'Boxplot of {y_column} and {x_column} ')
    if x_column is None:
        plt.xlabel(f"Count")
    else:
        plt.xlabel(f"{x_column}")
    if y_column is None:
        plt.ylabel(f"Count")
    else:
        plt.ylabel(f'{y_column}')
    plt.show()