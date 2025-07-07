import sys
import os
from correlation_tracker import CorrTracker
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
from sequence_utils import VOTSequence
import cv2



workspace_path = "../workspace-dir"
tracker_id = "tracker_cf"
results_dir = os.path.join(workspace_path, "results", tracker_id)
analysis_dir = os.path.join(workspace_path, "analysis", tracker_id)
toolkit_path = "../toolkit-dir"  
dataset_dir = "../workspace-dir/sequences"

# parameter values
alphas = np.linspace(0.0, 0.5, 11)
sigmas = np.linspace(0.5, 5, 11)
lambdas = np.linspace(0.1, 2000, 11)
enlarge_factors =  np.linspace(1.0, 1.5, 11)

# evaluate
def run_eval(alpha=None, sigma=None, lmbd=None, enlarge_factor = None):


    if alpha is not None:
        os.environ["ALPHA"] = str(alpha)
    if sigma is not None:
        os.environ["SIGMA"] = str(sigma)
    if lmbd is not None:
        os.environ["LMBDA"] = str(lmbd)
    if enlarge_factor is not None: 
        print(enlarge_factor)
        os.environ["ENLFCT"] = str(enlarge_factor)

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    if os.path.exists(analysis_dir):
        shutil.rmtree(analysis_dir)

    subprocess.run([
        "python", os.path.join(toolkit_path, "evaluate_tracker.py"),
        "--workspace_path", workspace_path,
        "--tracker", tracker_id
    ], check=True)

    out = subprocess.check_output([
        "python", os.path.join(toolkit_path, "compare_trackers.py"),
        "--workspace_path", workspace_path,
        "--tracker", tracker_id
    ]).decode()
    
    iou = float(out.split("Average overlap:")[1].split("\n")[0].strip())
    fails = float(out.split("Total failures:")[1].split("\n")[0].strip())
    fps = float(out.split("Average speed:")[1].split("FPS")[0].strip())
    print(iou, fails)
  
    return iou, fails, fps
  

def plot_metric(values, results, metric_idx, ylabel, xlabel, filename, is_max_best):
    metric = results[:, metric_idx]
    best_idx = np.argmax(metric) if is_max_best else np.argmin(metric)

    plt.figure(figsize=(8, 5))
    for i in range(len(values)):
        plt.scatter(values[i], metric[i], color="#b0b0b0", s=30)
    plt.scatter(values[best_idx], metric[best_idx], color="#e74c3c", s=60)
    plt.plot(values, metric, color="#b0b0b0", alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{filename}.pdf")
    plt.close()

def measure_sequence_times():

    sequences = sorted(
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    )

    per_sequence_times = []
    for seq_name in sequences:
        sequence = VOTSequence(dataset_dir, seq_name)
        tracker = CorrTracker()
        tracker.time_flag = True

        frame_idx = 0
        init_frame = 0
        init_times = []
        track_times = []
        frame_count = 0

        while frame_idx < sequence.length() and frame_idx < len(sequence.gt):
            img = cv2.imread(sequence.frame(frame_idx))
            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')

            start = time.time()
            if frame_idx == init_frame:
                tracker.initialize(img, gt_bb)
                init_times.append(time.time() - start)
                predicted_bbox = gt_bb
            else:
                predicted_bbox = tracker.track(img)
                track_times.append(time.time() - start)

            frame_count += 1
            overlap = sequence.overlap(predicted_bbox, gt_bb)
            frame_idx += 1 + (overlap == 0) * 4
            if overlap == 0:
                init_frame = frame_idx

        # compute average fps
        init_fps = 1 / np.mean(init_times) if init_times else 0
        track_fps = 1 / np.mean(track_times) if track_times else 0
        total_fps = frame_count / (sum(init_times) + sum(track_times) + 1e-6)

        per_sequence_times.append({
            "sequence": seq_name,
            "init_fps": init_fps,
            "track_fps": track_fps,
            "total_fps": total_fps
        })
    return per_sequence_times

def plot_timing(results):
   
    sequences = [r["sequence"] for r in results]
    init_fps = np.array([r["init_fps"] for r in results])
    track_fps = np.array([r["track_fps"] for r in results])
    total_fps = np.array([r["total_fps"] for r in results])
    x = np.arange(len(sequences))


    plt.figure(figsize=(10, 5))

    plt.plot(x, init_fps, marker='o', color='red', label='Initialization FPS', alpha = 0.6)
    plt.plot(x, track_fps, marker='s', color='blue', label='Tracking FPS', alpha = 0.6)
    plt.xticks(x, sequences, rotation=45, ha='right')
    plt.xlabel("Sequence")
    plt.ylabel("FPS")
    plt.legend(loc='best')  
    plt.tight_layout()
    plt.savefig("init_track_fps_linechart.pdf", bbox_inches='tight')
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(x, total_fps, marker='o', color='blue', label='Total FPS', alpha = 0.6)
    plt.xticks(x, sequences, rotation=45, ha='right')
    plt.xlabel("Sequence")
    plt.ylabel("FPS")
    plt.tight_layout()
    plt.savefig("total_fps_linechart.pdf", bbox_inches='tight')
    plt.close()

# alpha
# alpha_results = np.array([run_eval(alpha=a) for a in alphas])
# print(alpha_results)
# print(alphas)
# plot_metric(alphas, alpha_results, 1, "Failures", "alpha_value", "alpha_failures", is_max_best=False)
# plot_metric(alphas, alpha_results, 0, "Overlap", "alpha_value","alpha_overlap" ,is_max_best=True)
# plot_metric(alphas, alpha_results, 2, "FPS", "alpha_value","alpha_fps" ,is_max_best=True)

# sigma 
# sigma_results = np.array([run_eval(sigma=s) for s in sigmas])
# print(sigma_results)
# print(sigmas)
# plot_metric(sigmas, sigma_results, 1, "Failures", "sigma_value", "sigma_failures" ,is_max_best=False)
# plot_metric(sigmas, sigma_results, 0, "Overlap", "sigma_value", "sigma_overlap",is_max_best=True)
# plot_metric(sigmas, sigma_results, 2, "FPS", "sigma_value", "sigma_fps",is_max_best=True)

# lambda 
#lambda_results = np.array([run_eval(lmbd=l) for l in lambdas])
#print(lambda_results)
#print(lambdas)
#plot_metric(lambdas, lambda_results, 1, "Failures", "lambda_value", "lambda_value", is_max_best=False)
#plot_metric(lambdas, lambda_results, 0, "Overlap", "lambda_value","lambda_overlap", is_max_best=True)
#plot_metric(lambdas, lambda_results, 2, "FPS", "lambda_value", "lambda_fps",is_max_best=True)

# enlarge factor
# enlarge_results = np.array([run_eval(enlarge_factor=ef) for ef in enlarge_factors])
# print(enlarge_results)
# print(enlarge_factors)
# plot_metric(enlarge_factors, enlarge_results, 1, "Failures", "enlarge_value", "enlarge_value", is_max_best=False)
# plot_metric(enlarge_factors, enlarge_results, 0, "Overlap", "enlarge_value","enlarge_overlap", is_max_best=True)
# plot_metric(enlarge_factors, enlarge_results, 2, "FPS", "enlarge_value", "enlarge_fps",is_max_best=True)


timing_results = measure_sequence_times()
plot_timing(timing_results)