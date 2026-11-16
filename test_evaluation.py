"""
Test and Benchmark Suite for Image Quality Evaluation and Processing Configurations.

This script:
1. Validates the evaluation metrics (MSE, PSNR, SSIM) on identical images, synthetic
   perturbations, and edge cases (wrong dtype, dimension mismatch, empty array).
2. Evaluates matching low-light and normal-light image pairs from the LOL training dataset
   (sample_images/our485/low/ and sample_images/our485/high/).
3. Compares six processing configurations:
   - Configuration 1: No denoising + Gamma
   - Configuration 2: Mean + Gamma
   - Configuration 3: Median + Gamma
   - Configuration 4: Gaussian + Gamma
   - Configuration 5: Gaussian + CLAHE
   - Configuration 6: Gaussian + Gamma + CLAHE
4. Measures execution time, average MSE, average PSNR, and average SSIM.
5. Saves CSV results and generates visual comparison charts in output/evaluation_tests/.
6. Prints a comparative results table and identifies top performers per metric.
"""

import os
import time
import csv
import cv2
import numpy as np

from image_processing.evaluation import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    evaluate_image,
)
from image_processing.pipeline import enhance_image

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "evaluation_tests")
OUR485_LOW = os.path.join(BASE_DIR, "sample_images", "our485", "low")
OUR485_HIGH = os.path.join(BASE_DIR, "sample_images", "our485", "high")


def render_bar_chart(
    categories: list,
    values: list,
    title: str,
    ylabel: str,
    output_path: str,
    val_format: str = "{:.2f}",
    primary_color: tuple = (41, 128, 185),  # BGR
    highlight_max: bool = True,
):
    """
    Renders a high-resolution bar chart using OpenCV and saves as PNG.
    """
    width, height = 900, 520
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)  # Off-white background

    # Margins
    margin_left = 90
    margin_right = 50
    margin_top = 80
    margin_bottom = 110

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Title
    cv2.putText(
        canvas,
        title,
        (margin_left, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    # Y-axis limits
    min_val = min(values)
    max_val = max(values)
    y_min = 0.0 if min_val >= 0 else min_val * 1.1
    y_max = max_val * 1.2 if max_val > 0 else 1.0

    # Draw gridlines & Y labels
    num_ticks = 5
    for i in range(num_ticks + 1):
        tick_val = y_min + (y_max - y_min) * (i / num_ticks)
        y_pos = int(margin_top + plot_height - (i / num_ticks) * plot_height)

        # Grid line
        cv2.line(
            canvas,
            (margin_left, y_pos),
            (margin_left + plot_width, y_pos),
            (225, 225, 225),
            1,
        )
        # Tick text
        tick_str = val_format.format(tick_val)
        cv2.putText(
            canvas,
            tick_str,
            (15, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    # Axes lines
    cv2.line(
        canvas,
        (margin_left, margin_top),
        (margin_left, margin_top + plot_height),
        (80, 80, 80),
        2,
    )
    cv2.line(
        canvas,
        (margin_left, margin_top + plot_height),
        (margin_left + plot_width, margin_top + plot_height),
        (80, 80, 80),
        2,
    )

    # Y-axis Title
    cv2.putText(
        canvas,
        ylabel,
        (margin_left, margin_top - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    # Draw bars
    num_bars = len(categories)
    bar_width = int(plot_width / (num_bars * 1.6))
    spacing = int((plot_width - (num_bars * bar_width)) / (num_bars + 1))

    best_idx = int(np.argmax(values)) if highlight_max else -1

    for i, (cat, val) in enumerate(zip(categories, values)):
        bar_x = margin_left + spacing + i * (bar_width + spacing)
        bar_h = int(((val - y_min) / (y_max - y_min)) * plot_height)
        bar_y = margin_top + plot_height - bar_h

        # Bar colour: highlight top performer
        color = (39, 174, 96) if (i == best_idx) else primary_color

        # Fill bar
        cv2.rectangle(
            canvas, (bar_x, bar_y), (bar_x + bar_width, margin_top + plot_height), color, -1
        )
        # Bar border
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + bar_width, margin_top + plot_height),
            (40, 40, 40),
            1,
        )

        # Value label above bar
        val_str = val_format.format(val)
        (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        text_x = bar_x + (bar_width - tw) // 2
        cv2.putText(
            canvas,
            val_str,
            (text_x, bar_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        # Category label below bar
        # Break category label into lines if long
        parts = cat.split(" + ")
        for line_idx, part in enumerate(parts):
            (pw, _), _ = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            px = bar_x + (bar_width - pw) // 2
            py = margin_top + plot_height + 22 + (line_idx * 18)
            cv2.putText(
                canvas,
                part,
                (px, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, canvas)


def test_unit_metrics():
    """
    Unit tests for evaluation functions: mathematical correctness, identical image
    handling (infinite PSNR), and input validation.
    """
    print("\n[Step 1] Running unit tests on evaluation functions...")

    # 1. Identical images (MSE = 0, PSNR = inf, SSIM = 1.0)
    identical_gray = np.full((120, 120), 80, dtype=np.uint8)
    res_gray = evaluate_image(identical_gray, identical_gray)
    assert res_gray["mse"] == 0.0, f"Expected MSE=0, got {res_gray['mse']}"
    assert res_gray["psnr"] == float("inf"), f"Expected PSNR=inf, got {res_gray['psnr']}"
    assert abs(res_gray["ssim"] - 1.0) < 1e-4, f"Expected SSIM=1.0, got {res_gray['ssim']}"
    print("  [PASS] Identical grayscale image: MSE=0.0, PSNR=inf, SSIM=1.0")

    identical_rgb = np.full((120, 120, 3), 100, dtype=np.uint8)
    res_rgb = evaluate_image(identical_rgb, identical_rgb)
    assert res_rgb["mse"] == 0.0
    assert res_rgb["psnr"] == float("inf")
    assert abs(res_rgb["ssim"] - 1.0) < 1e-4
    print("  [PASS] Identical RGB colour image: MSE=0.0, PSNR=inf, SSIM=1.0")

    # 2. Known degradation test
    perturbed_rgb = identical_rgb.copy()
    perturbed_rgb[0:30, 0:30, :] = 200
    res_perturbed = evaluate_image(identical_rgb, perturbed_rgb)
    assert res_perturbed["mse"] > 0, "Expected positive MSE"
    assert res_perturbed["psnr"] < 100, "Expected finite PSNR"
    assert 0 <= res_perturbed["ssim"] < 1.0, "Expected SSIM in [0, 1)"
    print(
        f"  [PASS] Perturbed image metrics: MSE={res_perturbed['mse']:.2f}, "
        f"PSNR={res_perturbed['psnr']:.2f} dB, SSIM={res_perturbed['ssim']:.4f}"
    )

    # 3. Input validation & error handling
    validation_cases = [
        ("Dimension mismatch", np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 120), dtype=np.uint8), ValueError),
        ("Channel mismatch", np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8), ValueError),
        ("Wrong dtype float32", np.zeros((100, 100), dtype=np.float32), np.zeros((100, 100), dtype=np.float32), TypeError),
        ("Empty array", np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8), ValueError),
        ("Non-numpy input", [[1, 2]], np.zeros((2, 2), dtype=np.uint8), TypeError),
    ]

    for label, ref, proc, expected_err in validation_cases:
        try:
            evaluate_image(ref, proc)
            raise AssertionError(f"Failed to raise {expected_err.__name__} for {label}")
        except expected_err:
            print(f"  [PASS] Caught {expected_err.__name__} for {label}")


def run_benchmark_and_comparisons():
    """
    Evaluates 5 matching low-light and normal-light image pairs from our485
    across the 6 processing configurations.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print(" NIGHTVISION: Processing Configuration Quality Benchmark")
    print("=" * 72)

    # Select 5 image pairs from our485
    sample_names = ["10.png", "100.png", "101.png", "102.png", "103.png"]

    print(f"\n[Step 2] Selected {len(sample_names)} test pairs from 'sample_images/our485/':")
    for s in sample_names:
        print(f"  - Low: our485/low/{s}  <-->  Reference: our485/high/{s}")

    # Load all 5 pairs into memory
    dataset_pairs = []
    for name in sample_names:
        low_p = os.path.join(OUR485_LOW, name)
        high_p = os.path.join(OUR485_HIGH, name)

        bgr_low = cv2.imread(low_p)
        bgr_high = cv2.imread(high_p)

        if bgr_low is None or bgr_high is None:
            raise FileNotFoundError(f"Failed to load image pair for {name}")

        rgb_low = cv2.cvtColor(bgr_low, cv2.COLOR_BGR2RGB)
        rgb_high = cv2.cvtColor(bgr_high, cv2.COLOR_BGR2RGB)
        dataset_pairs.append((name, rgb_low, rgb_high))

    # Evaluate baseline: Raw Low-Light vs Ground Truth
    baseline_mse = []
    baseline_psnr = []
    baseline_ssim = []
    for name, low, high in dataset_pairs:
        m = evaluate_image(high, low)
        baseline_mse.append(m["mse"])
        baseline_psnr.append(m["psnr"])
        baseline_ssim.append(m["ssim"])

    print("\n[Baseline Metric - Raw Low-Light vs Normal-Light Reference]:")
    print(
        f"  Average MSE:  {np.mean(baseline_mse):.2f}\n"
        f"  Average PSNR: {np.mean(baseline_psnr):.2f} dB\n"
        f"  Average SSIM: {np.mean(baseline_ssim):.4f}"
    )

    # Define the 6 processing configurations specified in Task 3
    configurations = [
        {
            "name": "No Denoising + Gamma",
            "params": {"denoise_method": "none", "gamma": 0.6, "use_clahe": False},
        },
        {
            "name": "Mean + Gamma",
            "params": {"denoise_method": "mean", "kernel_size": 3, "gamma": 0.6, "use_clahe": False},
        },
        {
            "name": "Median + Gamma",
            "params": {"denoise_method": "median", "kernel_size": 3, "gamma": 0.6, "use_clahe": False},
        },
        {
            "name": "Gaussian + Gamma",
            "params": {"denoise_method": "gaussian", "kernel_size": 3, "gamma": 0.6, "use_clahe": False},
        },
        {
            "name": "Gaussian + CLAHE",
            # gamma=1.0 acts as identity luminance transformation
            "params": {"denoise_method": "gaussian", "kernel_size": 3, "gamma": 1.0, "use_clahe": True, "clip_limit": 2.0},
        },
        {
            "name": "Gaussian + Gamma + CLAHE",
            "params": {"denoise_method": "gaussian", "kernel_size": 3, "gamma": 0.6, "use_clahe": True, "clip_limit": 2.0},
        },
    ]

    print("\n[Step 3] Benchmarking 6 processing configurations...")

    # Warmup run to avoid cold-start overhead (JIT/import/memory allocation) in timing
    if dataset_pairs:
        _ = enhance_image(dataset_pairs[0][1], denoise_method="gaussian", kernel_size=3, gamma=0.6, use_clahe=True)
        _ = evaluate_image(dataset_pairs[0][2], dataset_pairs[0][1])

    benchmark_summary = []

    for cfg in configurations:
        cfg_name = cfg["name"]
        cfg_params = cfg["params"]

        mse_list = []
        psnr_list = []
        ssim_list = []
        time_list = []

        for name, low, high in dataset_pairs:
            # Measure processing execution time
            t_start = time.perf_counter()
            enhanced = enhance_image(low, **cfg_params)
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            # Calculate quantitative metrics against normal-light ground truth
            metrics = evaluate_image(high, enhanced)

            mse_list.append(metrics["mse"])
            psnr_list.append(metrics["psnr"])
            ssim_list.append(metrics["ssim"])
            time_list.append(t_elapsed_ms)

        avg_mse = float(np.mean(mse_list))
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
        avg_time = float(np.mean(time_list))

        benchmark_summary.append({
            "configuration": cfg_name,
            "avg_mse": avg_mse,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_time_ms": avg_time,
        })

        print(
            f"  -> {cfg_name:<28} | MSE: {avg_mse:>7.2f} | "
            f"PSNR: {avg_psnr:>6.2f} dB | SSIM: {avg_ssim:>6.4f} | Time: {avg_time:>5.2f} ms"
        )

    # -------------------------------------------------------------
    # Step 4: Save Results to CSV
    # -------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "configuration_comparison.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Average_MSE", "Average_PSNR_dB", "Average_SSIM", "Average_Time_ms"])
        for row in benchmark_summary:
            writer.writerow([
                row["configuration"],
                f"{row['avg_mse']:.2f}",
                f"{row['avg_psnr']:.2f}",
                f"{row['avg_ssim']:.4f}",
                f"{row['avg_time_ms']:.2f}",
            ])
    print(f"\n[Artifact] Saved comparison CSV to: {csv_path}")

    # -------------------------------------------------------------
    # Step 5: Generate Visual Comparison Charts
    # -------------------------------------------------------------
    cfg_labels = [row["configuration"] for row in benchmark_summary]
    psnr_vals = [row["avg_psnr"] for row in benchmark_summary]
    ssim_vals = [row["avg_ssim"] for row in benchmark_summary]
    time_vals = [row["avg_time_ms"] for row in benchmark_summary]

    # Chart 1: PSNR Comparison
    chart_psnr_path = os.path.join(OUTPUT_DIR, "psnr_comparison_chart.png")
    render_bar_chart(
        categories=cfg_labels,
        values=psnr_vals,
        title="Average Peak Signal-to-Noise Ratio (PSNR) by Configuration",
        ylabel="PSNR (dB) - Higher is Better",
        output_path=chart_psnr_path,
        val_format="{:.2f} dB",
        primary_color=(192, 57, 43),  # Deep Red/Orange
        highlight_max=True,
    )

    # Chart 2: SSIM Comparison
    chart_ssim_path = os.path.join(OUTPUT_DIR, "ssim_comparison_chart.png")
    render_bar_chart(
        categories=cfg_labels,
        values=ssim_vals,
        title="Average Structural Similarity Index (SSIM) by Configuration",
        ylabel="SSIM (0 to 1) - Higher is Better",
        output_path=chart_ssim_path,
        val_format="{:.4f}",
        primary_color=(41, 128, 185),  # Cobalt Blue
        highlight_max=True,
    )

    # Chart 3: Execution Time Comparison
    chart_time_path = os.path.join(OUTPUT_DIR, "processing_time_chart.png")
    render_bar_chart(
        categories=cfg_labels,
        values=time_vals,
        title="Average Execution Time per 400x600 Image",
        ylabel="Processing Time (ms) - Lower is Faster",
        output_path=chart_time_path,
        val_format="{:.2f} ms",
        primary_color=(142, 68, 173),  # Purple
        highlight_max=False,
    )

    print(f"[Artifact] Saved PSNR chart to: {chart_psnr_path}")
    print(f"[Artifact] Saved SSIM chart to: {chart_ssim_path}")
    print(f"[Artifact] Saved Execution Time chart to: {chart_time_path}")

    # -------------------------------------------------------------
    # Step 6: Reporting and Top Performer Identification
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" NIGHTVISION: Processing Configuration Performance Summary")
    print("=" * 80)
    print(f"{'Configuration':<28} | {'Avg MSE':<10} | {'Avg PSNR (dB)':<14} | {'Avg SSIM':<10} | {'Avg Time (ms)':<14}")
    print("-" * 80)
    for row in benchmark_summary:
        print(
            f"{row['configuration']:<28} | "
            f"{row['avg_mse']:>10.2f} | "
            f"{row['avg_psnr']:>14.2f} | "
            f"{row['avg_ssim']:>10.4f} | "
            f"{row['avg_time_ms']:>14.2f}"
        )
    print("-" * 80)

    # Identify top performers separately per metric
    best_psnr_row = max(benchmark_summary, key=lambda x: x["avg_psnr"])
    best_ssim_row = max(benchmark_summary, key=lambda x: x["avg_ssim"])
    fastest_row = min(benchmark_summary, key=lambda x: x["avg_time_ms"])

    print("\n[Metric-Specific Analysis]")
    print(f" * Highest PSNR Performer: {best_psnr_row['configuration']}")
    print(f"   -> Average PSNR: {best_psnr_row['avg_psnr']:.2f} dB (Average MSE: {best_psnr_row['avg_mse']:.2f})")
    print(f" * Highest SSIM Performer: {best_ssim_row['configuration']}")
    print(f"   -> Average SSIM: {best_ssim_row['avg_ssim']:.4f}")
    print(f" * Fastest Execution Time: {fastest_row['configuration']}")
    print(f"   -> Average Latency: {fastest_row['avg_time_ms']:.2f} ms")
    print("=" * 80)

    return True


if __name__ == "__main__":
    test_unit_metrics()
    run_benchmark_and_comparisons()
