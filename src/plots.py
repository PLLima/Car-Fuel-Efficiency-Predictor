import matplotlib.pyplot as plt

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
