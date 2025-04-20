import matplotlib.pyplot as plt

# Define models and data
models = ['HylerLidar', 'HyperLidar-early_exit', 'CENet', 'Cylinder3D']
training_fps = [1.17, 1.01, 36.42, 3.04]
inference_fps = [1.17, 1.18, 49.56, 1.52]
miou = [57.27, 54.88, 63.50, 56.63]

# Optional: (x, y) deltas to draw arrows (e.g., from baseline to optimized version)
arrows = {
    # 'HyperLidar-early_exit': (0.0, 2.5),  # e.g., from early_exit to full version
}

# Plot style settings
colors = ['blue', 'purple', 'green', 'orange']
markers = ['o', 's', '^', 'D']
sizes = [150] * 4

plt.figure(figsize=(12, 5))

# Highlight better region
# plt.axhspan(60, 70, facecolor='pink', alpha=0.3)
# plt.text(45, 68.5, 'Better', fontsize=12, weight='bold')

# Plot each model
plt.subplot(1, 2, 1)
for i, model in enumerate(models):
    plt.scatter(inference_fps[i], miou[i], s=sizes[i], c=colors[i], marker=markers[i], label=model)
    # plt.text(inference_fps[i] + 0.8, miou[i], f"{model}\n{miou[i]:.1f}%, {inference_fps[i]:.1f} FPS", fontsize=9)

    # Draw arrow if specified
    if model in arrows:
        dx, dy = arrows[model]
        plt.arrow(inference_fps[i], miou[i], dx, dy,
                  head_width=0.5, head_length=0.7, fc='gray', ec='gray', linestyle='--')

# Labels and formatting
plt.xlabel("Inference Runtime (FPS)", fontsize=12)
plt.ylabel("mIoU on SemanticKITTI test set (%)", fontsize=12)
plt.title("Model Trade-offs: Accuracy vs Speed", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, 60)
plt.ylim(50, 70)
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
for i, model in enumerate(models):
    plt.scatter(training_fps[i], miou[i], s=sizes[i], c=colors[i], marker=markers[i], label=model)
    # plt.text(training_fps[i] + 0.8, miou[i], f"{model}\n{miou[i]:.1f}%, {training_fps[i]:.1f} FPS", fontsize=9)

    # Draw arrow if specified
    if model in arrows:
        dx, dy = arrows[model]
        plt.arrow(training_fps[i], miou[i], dx, dy,
                  head_width=0.5, head_length=0.7, fc='gray', ec='gray', linestyle='--')

# Labels and formatting
plt.xlabel("Training Runtime (FPS)", fontsize=12)
plt.ylabel("mIoU on SemanticKITTI test set (%)", fontsize=12)
plt.title("Model Trade-offs: Accuracy vs Speed", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, 60)
plt.ylim(50, 70)
plt.legend()
plt.tight_layout()
# Save and show
plt.savefig("model_comparison_semantickitti.png", dpi=300)
print("Plot saved as model_comparison_semantickitti.png")


