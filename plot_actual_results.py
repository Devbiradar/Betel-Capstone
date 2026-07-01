import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_results_plot():
    df = pd.read_csv('model_comparison.csv')
    
    # Extract data
    models = df['Model'].tolist()
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # We will multiply by 100 to show percentages
    data = {}
    for metric in metrics:
        data[metric] = df[metric].values * 100
        
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    rects1 = ax.bar(x - width*1.5, data['accuracy'], width, label='Accuracy', color='#1f77b4')
    rects2 = ax.bar(x - width*0.5, data['precision'], width, label='Precision', color='#ff7f0e')
    rects3 = ax.bar(x + width*0.5, data['recall'], width, label='Recall', color='#2ca02c')
    rects4 = ax.bar(x + width*1.5, data['f1'], width, label='F1 Score', color='#d62728')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Final Model Performance Comparison', fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=10)

    ax.set_ylim(0, 115) # Add space for labels
    
    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    # Clean up spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig('actual_results_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved to actual_results_comparison.png")

if __name__ == '__main__':
    create_results_plot()
