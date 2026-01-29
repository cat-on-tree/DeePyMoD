""" Constains a tool to convert from Tensorboard to Pandas DataFrame """

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from natsort import natsorted
import matplotlib.pyplot as plt


def load_tensorboard(path: str) -> pd.DataFrame:
    """Loads tensorboard files into a pandas dataframe. Assumes one run per folder!

    Args:
        path (string): path of folder with tensorboard files.

    Returns:
        DataFrame: Pandas dataframe with all run data.
    """

    event_paths = [
        file
        for file in os.walk(path, topdown=True)
        if file[2][0][: len("events")] == "events"
    ]

    df = pd.DataFrame()
    steps = None  # steps are the same for all files

    for event_idx, path in enumerate(event_paths):
        summary_iterator = EventAccumulator(os.path.join(path[0], path[2][0])).Reload()
        tags = summary_iterator.Tags()["scalars"]
        data = [
            [event.value for event in summary_iterator.Scalars(tag)] for tag in tags
        ]
        if steps is None:
            steps = [event.step for event in summary_iterator.Scalars(tags[0])]

        # Adding to dataframe
        tags = [tag.replace("/", "_") for tag in tags]  # for name consistency
        if (
            event_idx > 0
        ):  # We have one file in the top level, so after we need to use folder name
            tags = [path[0].split("/")[-1]]

        for idx, tag in enumerate(tags):
            try:
                df[tag] = data[idx]
            except ValueError:  # more debugging info
                print(
                    f"Warning: Either the {tag = } of `df` or {idx = } of `data` do not exist! Check for pre-existing saved files. "
                )
        df.index = steps
    return df


def plot_history(foldername: str):
    """Plots the training history of the model."""
    history = load_tensorboard(foldername)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for history_key in history.keys():
        history_key_parts = history_key.split("_")

        # === 1. 处理 loss ===
        if history_key_parts[0] == "loss":
            if history_key_parts[-1] == "0":
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1] + "_" + history_key_parts[-1],
                    linestyle="--",
                )
            elif history_key_parts[-1] == "1":
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1] + "_" + history_key_parts[-1],
                    linestyle=":",
                )
            else:
                axs[0].semilogy(
                    history[history_key],
                    label=history_key_parts[1] + "_" + history_key_parts[-1],
                    linestyle="-",
                )
            # 修正：remaining 逻辑通常不应嵌套在 loss 内部，或者需要更严格的判断
            if history_key_parts[0] == "remaining" and len(history_key_parts) > 4:
                axs[0].semilogy(
                    history[history_key],
                    label=f"{history_key_parts[1]}_{history_key_parts[3]}_{history_key_parts[4]}",
                    linestyle="-.",
                )

        # === 2. 处理 coeffs (报错点) ===
        if history_key_parts[0] == "coeffs":
            # 动态构建 label，防止越界
            # 如果长度够长，使用原来的格式
            if len(history_key_parts) >= 5:
                label_str = f"{history_key_parts[2]}_{history_key_parts[3]}_{history_key_parts[4]}"
            else:
                # 长度不够，就把剩下的部分全拼起来
                label_str = "_".join(history_key_parts[2:])

            # 确保索引 2 存在
            if len(history_key_parts) > 2:
                if history_key_parts[2] == "0":
                    axs[1].plot(history[history_key], label=label_str, linestyle="--")
                elif history_key_parts[2] == "1":
                    axs[1].plot(history[history_key], label=label_str, linestyle=":")
                else:
                    axs[1].plot(history[history_key], label=label_str, linestyle="-")

        # === 3. 处理 unscaled ===
        if history_key_parts[0] == "unscaled":
            if len(history_key_parts) >= 6:
                label_str = f"{history_key_parts[3]}_{history_key_parts[4]}_{history_key_parts[5]}"
            else:
                label_str = "_".join(history_key_parts[3:])

            if len(history_key_parts) > 3:
                if history_key_parts[3] == "0":
                    axs[2].plot(history[history_key], label=label_str, linestyle="--")
                elif history_key_parts[3] == "1":
                    axs[2].plot(history[history_key], label=label_str, linestyle=":")
                else:
                    axs[2].plot(history[history_key], label=label_str, linestyle="-")

        # === 4. 处理 estimator ===
        if history_key_parts[0] == "estimator":
            if len(history_key_parts) >= 6:
                label_str = f"{history_key_parts[3]}_{history_key_parts[4]}_{history_key_parts[5]}"
            else:
                label_str = "_".join(history_key_parts[3:])

            if len(history_key_parts) > 3:
                if history_key_parts[3] == "0":
                    axs[3].plot(history[history_key], label=label_str, linestyle="--")
                elif history_key_parts[3] == "1":
                    axs[3].plot(history[history_key], label=label_str, linestyle=":")
                else:
                    axs[3].plot(history[history_key], label=label_str, linestyle="-")

    # axs[0].set_ylim([-2, 2])
    axs[1].set_ylim([-2, 2])
    axs[2].set_ylim([-2, 2])
    axs[3].set_ylim([-2, 2])

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    plt.show()