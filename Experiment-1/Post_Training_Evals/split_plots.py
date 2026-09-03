import re

with open('training_curves.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We only want to modify inside plot_training_metrics_comparison
start_marker = "def plot_training_metrics_comparison"
end_marker = "def compare_models"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

func_body = content[start_idx:end_idx]

# Replace axes initialization
old_init = '''    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    for ax in axes:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', labelsize=13, direction='in', length=4)

    # Increase X/Y tick label sizes explicitly for each subplot
    axes[0].tick_params(axis='x', labelsize=tt_xtick_fs)
    axes[0].tick_params(axis='y', labelsize=tt_ytick_fs)
    axes[1].tick_params(axis='both', labelsize=tt_ytick_fs)
    axes[2].tick_params(axis='both', labelsize=tt_ytick_fs)'''

new_init = '''    # Create separate figure for training time
    fig_time, ax_time = plt.subplots(1, 1, figsize=(7, 6))
    fig_time.patch.set_facecolor('white')
    ax_time.set_facecolor('white')
    ax_time.tick_params(axis='both', which='major', labelsize=13, direction='in', length=4)
    ax_time.tick_params(axis='x', labelsize=tt_xtick_fs)
    ax_time.tick_params(axis='y', labelsize=tt_ytick_fs)

    # Create figure for loss and trajectory
    fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_metrics.patch.set_facecolor('white')
    for ax in axes_metrics:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', labelsize=tt_ytick_fs, direction='in', length=4)'''

func_body = func_body.replace(old_init, new_init)

# Replace the text alignment and rotation for axes[0].set_xticklabels
old_xticklabels = "axes[0].set_xticklabels(sorted_models, fontsize=tt_xtick_fs, fontweight='bold')"
new_xticklabels = "axes[0].set_xticklabels(sorted_models, fontsize=tt_xtick_fs, fontweight='bold', rotation=25, ha='right')"
func_body = func_body.replace(old_xticklabels, new_xticklabels)

# Replace all axes indices with their new variables
func_body = func_body.replace("axes[0]", "ax_time")
func_body = func_body.replace("axes[1]", "axes_metrics[0]")
func_body = func_body.replace("axes[2]", "axes_metrics[1]")

# Fix layout and saving
old_save = '''    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "model_comparison_training_metrics"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training metrics comparison plot saved to: {output_path}.png and {output_path}.pdf")'''

new_save = '''    # Layout and save time plot
    fig_time.tight_layout()
    output_time_path = Path(output_dir) / "model_comparison_training_time"
    output_time_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_multi_format(str(output_time_path), dpi=300, bbox_inches='tight')
    plt.close(fig_time)
    
    # Layout and save metrics plot
    fig_metrics.tight_layout()
    output_metrics_path = Path(output_dir) / "model_comparison_training_metrics"
    save_plot_multi_format(str(output_metrics_path), dpi=300, bbox_inches='tight')
    plt.close(fig_metrics)
    
    print(f"✅ Training time plot saved to: {output_time_path}.png and {output_time_path}.pdf")
    print(f"✅ Training metrics plot saved to: {output_metrics_path}.png and {output_metrics_path}.pdf")'''

func_body = func_body.replace(old_save, new_save)

content = content[:start_idx] + func_body + content[end_idx:]

with open('training_curves.py', 'w', encoding='utf-8') as f:
    f.write(content)
