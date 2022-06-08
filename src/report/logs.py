import pandas as pd
import os
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

def print_and_save_logs(metrics, is_training, iteration, total_length, start_time, model_name=None, dataset_name=None, keep_print=False, save=True):
    stage = "train" if is_training else "val"  
    current_time = time.time()
    elapsed_time = current_time-start_time
    progress_percent = (iteration/total_length)*100
    d = {
        "Loss": metrics.get_loss(),
        "Pos F1": metrics.get_pos_f1(),
        "Px Acc": metrics.get_px_acc(),
        "Rx Acc": metrics.get_rx_acc(),
        "S div": metrics.silence_divergence,
        "H div": metrics.px_similarity_div,
        "GS div": metrics.groove_similarity_div,
        "Iteration": iteration,
        "Progress": f"{int(progress_percent)}%",
        "Stage": stage,
        "Elapsed Time": int(elapsed_time),
        "Expected End Time": int(elapsed_time*100/progress_percent) if progress_percent > 0 else -1
    }
    df = pd.DataFrame(columns=["Loss", "Pos F1", "Px Acc", "Rx Acc", "S div", "H div", "GS div", "Iteration", "Progress"])
    df = df.append(d, ignore_index=True)
    columns = df.to_string().split("\n")[0]
    values = df.to_string().split("\n")[1]
    if iteration == 0: 
        print(columns)
    #print(columns)
    if keep_print or iteration-1 == total_length:
        print(values)
    else:
        print(values, end="\r")

    if save:
        df_path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}.csv")
        # read existing df
        if os.path.isfile(df_path):
            old_df = pd.read_csv(df_path)
            df = old_df.append(df, ignore_index=True)
        #write
        df.to_csv(df_path, index=False)