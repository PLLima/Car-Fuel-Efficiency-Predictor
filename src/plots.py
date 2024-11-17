import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_violinplot(dataset, atribute, title, description, filename):
    plt.figure(figsize=(10, 6))
    sns.violinplot(dataset[atribute], orient='h', color='skyblue')
    plt.xlabel(description)
    plt.ylabel('Número de Instâncias')
    plt.title(title)
    plt.savefig(f'plots/violin_plots/{filename}.png')

def plot_piechart(dataset, atribute, title, filename):
    plt.figure(figsize=(10, 6))
    counts = dataset[atribute].value_counts()
    plt.pie(counts, labels=counts.index, 
            autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title(title)
    plt.savefig(f'plots/pie_charts/{filename}.png')

def plot_corr_heatmap(dataset, title, filename):
    numeric_df = dataset.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.savefig(f'plots/heatmaps/{filename}.png')

def plot(method, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Average Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit Line')
    plt.xlabel('Actual Combination MPG')
    plt.ylabel('Average Predicted Combination MPG')
    plt.title(f'{method.upper()} Model: Average Predicted vs Actual Combination MPG')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'src/plots/{method}_actual_vs_predicted.png')