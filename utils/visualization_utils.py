import torch
import matplotlib.pyplot as plt

from src.train import zero_shot_predict

def visualize_zero_shot_predict(model, image, processor, tokenizer, captions, labels, label, prompt, title, config):
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