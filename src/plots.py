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

def plot_scatterplot(model, modelname, y_pred, y_test, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label=title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Linha de Ajuste Ideal')
    plt.xlabel('Consumo Médio Real')
    plt.ylabel('Consumo Médio Predito')
    plt.title(f'{model.upper()} {modelname}: Predições vs. Consumo Médio Real')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/scatter_plots/{filename}.png')