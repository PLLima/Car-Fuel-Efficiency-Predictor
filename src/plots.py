import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(dataset, title, x_label, y_label, filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset, orient='h', color='skyblue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f'plots/box_plots/{filename}.png')

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

def plot_residuals(residuals_train, residuals_test, y_train, y_test, filename):
    df = {
        'Predições': np.concatenate((y_train, y_test)),
        'Resíduos': np.concatenate((residuals_train, residuals_test)),
        'Dataset': ['Treinamento'] * len(y_train) + ['Teste'] * len(y_test)
    }
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Resíduos', y='Predições', hue='Dataset', style='Dataset')
    plt.savefig(f'plots/scatter_plots/{filename}.png')

def plot_MSEs(random_forest_mse, linear_regression_mse, neural_networks_mse):
    plt.figure(figsize=(10, 6))
    plt.bar(['Random Forest', 'Linear Regressor', 'Neural Networt (MLP)'], [random_forest_mse, linear_regression_mse, neural_networks_mse])
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.title('Erros Quadráticos Médios dos Modelos Otimizados')
    plt.legend()
    plt.savefig('plots/Test_MSEs.png')

def plot_model_weights(names, values):
    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.title('Pesos dados pelo modelo aos atributos de entrada')
    plt.legend()
    plt.savefig('plots/atribute_weights.png')