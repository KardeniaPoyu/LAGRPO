
import os
from PIL import Image

def combine_vertically(top_path, bottom_path, output_path):
    if not os.path.exists(top_path) or not os.path.exists(bottom_path):
        print(f"Skipping {output_path}: one of the files missing.")
        return

    img1 = Image.open(top_path)
    img2 = Image.open(bottom_path)

    # Ensure same width
    target_width = max(img1.width, img2.width)
    
    # Resize if necessary to match widths, maintaining aspect ratio
    def resize_img(img, new_width):
        new_height = int(img.height * (new_width / img.width))
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img1 = resize_img(img1, target_width)
    img2 = resize_img(img2, target_width)

    dst = Image.new('RGB', (target_width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    dst.save(output_path)
    print(f"Created composite: {output_path}")

plot_dir = 'plots'
pairs = [
    ('ppo_vs_grpo_classic.png', 'ppo_vs_grpo_2x3_master.png', 'composite_ppo_vs_grpo.png'),
    ('grpo_g_ablation_classic.png', 'grpo_g_ablation_2x3_master.png', 'composite_grpo_ablation.png'),
    ('lagrpo_ablation_study.png', 'lagrpo_ablation_2x3_master.png', 'composite_lagrpo_ablation.png'),
    ('eval_summary_classic.png', 'eval_summary.png', 'composite_eval_summary.png'),
    ('success_rate_heatmap_classic.png', 'success_rate_heatmap.png', 'composite_success_rate_heatmap.png') # SVG needs to be PNG
]

# Note: Heatmap has SVG and classic PNG. I should check if there is a non-SVG new one.
# Let's check listing again.
# {"name":"success_rate_heatmap.png","sizeBytes":"166700"} - Yes there is a PNG version.

for top, bottom, out in pairs:
    combine_vertically(
        os.path.join(plot_dir, top),
        os.path.join(plot_dir, bottom),
        os.path.join(plot_dir, out)
    )
