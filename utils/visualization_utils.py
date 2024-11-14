import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.train import zero_shot_predict

def visualize_zero_shot_predict(model, image, processor, tokenizer, captions, labels, label, prompt, title, config):
    sns.reset_orig()
    
    probs = zero_shot_predict(model, image, processor, tokenizer, captions, config)
    top_prob, top_idx = torch.topk(probs, min(5, len(captions)), dim=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [2, 2]})
    fig.suptitle(title, fontsize=20, fontweight='bold')

    ax1.imshow(image.resize((224, 224)))
    ax1.axis('off')
    ax1.title.set_text('correct label: ' +  label)

    formatted_label = prompt.format(label)
    correct_label_idx = captions.index(formatted_label)
    correct_prob = probs[0][correct_label_idx].item() * 100
    correct_rank = (probs[0] > probs[0][correct_label_idx]).sum().item() + 1
    string = f"correct rank: {correct_rank}/{len(labels)}   correct probability: {correct_prob:.2f} %"

    colors = ['dodgerblue'] * len(top_idx[0])
    if correct_rank == 1:
        colors[0] = 'forestgreen'
    else:
        colors[0] = 'tomato'
        if correct_rank <= len(top_idx[0]):
            colors[correct_rank - 1] = 'forestgreen'
             
    sorted_labels = [captions[idx] for idx in top_idx[0]]

    ax2.barh(range(len(top_prob[0])), top_prob[0].detach().cpu().numpy() * 100, color=colors)
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.title.set_text(string)
    for bar, label in zip(ax2.patches, sorted_labels):
        ax2.text(1.5, bar.get_y() + bar.get_height() / 2, label, color='black', ha='left', va='center')

    plt.tight_layout()
    plt.show()
    
def plot_few_shot_results(accuracy_dict, title):
    sns.set_style("darkgrid")
    n_shots = [0, 1, 2, 4, 8, 16]
    
    for key, value in accuracy_dict.items():
        sns.lineplot(x=n_shots, y=value, label=key, marker='o')
        
    plt.xlabel('Number of training examples per class', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(n_shots)
    plt.legend(loc='lower right')          
    plt.suptitle(title, fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
def plot_results_multiple(results, dataset_names, labels, zero_shot_results):
    _, axes = plt.subplots(2, 3, figsize=(18, 12))
    sns.set_style("darkgrid")
    n_shots = [0, 1, 2, 4, 8, 16]
    
    for i, (dataset_results, dataset_name) in enumerate(zip(results, dataset_names)):
        
        ax = axes[i // 3, i % 3]
        
        for j, model_results in enumerate(dataset_results):
            sns.lineplot(x=n_shots[1:], y=model_results, label=labels[j], marker='o', ax=ax)
            ax.plot(0, zero_shot_results[i][j], marker='*', markersize=10, color=ax.get_lines()[-1].get_color())
         
        ax.set_xlabel('Number of training examples per class', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xticks(n_shots)
        ax.legend(loc='lower right')          
        ax.set_title(dataset_name, fontsize=20, fontweight='bold')
        
    plt.tight_layout()
    plt.show()
    
def plot_per_class_accuracies(per_class_accuracies, title, num_bins=10):
    accuracies = list(per_class_accuracies.values())
    plot_title = 'Class Accuracy Distribution - ' + title
    
    colors = sns.color_palette('viridis', num_bins)
    sns.set_style("darkgrid")
    
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(accuracies, bins=[i / num_bins for i in range(num_bins + 1)], kde=False, edgecolor='black')
    
    for i, patch in enumerate(ax.patches):
        patch.set_facecolor(colors[i]) 
        
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlim(0, 1)
    plt.xticks([i/10 for i in range(11)])
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune=None))
    
    plt.suptitle(plot_title, fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.show()