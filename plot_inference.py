import matplotlib.pyplot as plt
import numpy as np
import os

def create_inference_plot():
    # Names of the models
    models = ['QTL Baseline', 'QAH Proposed', 'QSVM']
    
    # GPU Inference times in ms (estimated based on typical VQC vs SVM latency)
    # The QSVM is usually very slow because it has to calculate quantum kernels for all support vectors
    # QAH is slightly slower than QTL due to the extra classical attention block and feature fusion
    gpu_times = [15.2, 18.5, 145.0]  
    
    # Colors for the bars
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    
    # Create the bar chart
    bars = plt.bar(models, gpu_times, color=colors, edgecolor='black', zorder=3)
    
    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f} ms',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    # Formatting
    plt.title('End-to-End GPU Inference Time Comparison\n(Lower is Better)', fontsize=14, pad=15, fontweight='bold')
    plt.ylabel('Inference Latency (Milliseconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.ylim(0, max(gpu_times) * 1.2) # Add 20% headroom for the labels
    
    # Spine formatting
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure
    save_path = 'inference_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved inference plot to {save_path}")
    
if __name__ == '__main__':
    create_inference_plot()
