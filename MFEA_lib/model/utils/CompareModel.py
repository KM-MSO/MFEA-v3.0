import matplotlib.pyplot as plt
import numpy as np
from .. import AbstractModel
import os
import pandas as pd
from typing import List, Tuple
from . import loadModel

class CompareModel():
    # TODO so sánh
    def __init__(self, models: List[AbstractModel.model], label: List[str] = None) -> None:
        self.models = models
        if label is None:
            label = [m.name for m in self.models]
        else:
            assert len(self.models) == len(label)
            for idx in range(len(label)):
                if label[idx] == Ellipsis:
                    label[idx] = self.models[idx].name

        self.label = label

    def render(self, shape: tuple = None, min_cost=0, nb_generations: int = None, step=1, figsize: Tuple[int, int] = None, dpi=200, yscale: str = None, re=False, label_shape=None, label_loc=None):
        assert np.all([len(self.models[0].tasks) == len(m.tasks)
                      for m in self.models])
        nb_tasks = len(self.models[0].tasks)
        for i in range(nb_tasks):
            assert np.all([self.models[0].tasks[i] == m.tasks[i]
                          for m in self.models])

        if label_shape is None:
            label_shape = (1, len(self.label))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.label)

        if label_loc is None:
            label_loc = 'lower center'

        if shape is None:
            shape = (nb_tasks // 3 + np.sign(nb_tasks % 3), 3)
        else:
            assert shape[0] * shape[1] >= nb_tasks

        if nb_generations is None:
            nb_generations = min([len(m.history_cost) for m in self.models])
        else:
            nb_generations = min(nb_generations, min(
                [len(m.history_cost) for m in self.models]))

        if figsize is None:
            figsize = (shape[1] * 6, shape[0] * 5)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.suptitle("Compare Models\n", size=15)
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        if step >= 10:
            marker = 'o'
        else:
            marker = None

        for idx_task, task in enumerate(self.models[0].tasks):
            for idx_model, model in enumerate(self.models):
                fig.axes[idx_task].plot(
                    np.append(np.arange(0, nb_generations, step),
                              np.array([nb_generations - 1])),
                    np.where(
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task] >= min_cost,
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task],
                        0
                    ),
                    label=self.label[idx_model],
                    marker=marker,
                )
                # plt.legend()
                if yscale is not None:
                    fig.axes[idx_task].set_yscale(yscale)
            fig.axes[idx_task].set_title(task.name)
            fig.axes[idx_task].set_xlabel("Generations")
            fig.axes[idx_task].set_ylabel("Cost")

        for idx_blank_fig in range(idx_task + 1, shape[0] * shape[1]):
            fig.delaxes(fig.axes[idx_task + 1])

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, loc=label_loc, ncol=label_shape[1])
        plt.show()
        if re:
            return fig

    def summarizing_compare_result(self, path=None, idx_main_algo=0, min_value=0, combine=True, nb_task=50, ls_benchmark=None):
        #
        if path is None:
            result_table = np.zeros(shape=(len(self.label)-1, 3), dtype=int)
            name_row = []
            name_column = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(self.label)):
                if idx_algo != idx_main_algo:
                    name_row.append(
                        str(self.label[idx_main_algo] + " vs " + self.label[idx_algo]))
                    result = np.where(self.models[idx_main_algo].history_cost[-1] > min_value, self.models[idx_main_algo].history_cost[-1], min_value)\
                        - np.where(self.models[idx_algo].history_cost[-1] > min_value,
                                   self.models[idx_algo].history_cost[-1], min_value)
                    # Better
                    result_table[count_row][0] += len(np.where(result < 0)[0])
                    # Equal
                    result_table[count_row][1] += len(np.where(result == 0)[0])
                    # Worse
                    result_table[count_row][2] += len(np.where(result > 0)[0])
                    count_row += 1

            result_table = pd.DataFrame(
                result_table, columns=name_column, index=name_row)
        else:
            list_algo = os.listdir(path)
            print(list_algo)
            benchmarks = [name_ben.split(
                "_")[-1].split(".")[0] for name_ben in os.listdir(os.path.join(path, list_algo[0]))]
            ls_model_cost = [np.zeros(
                shape=(len(benchmarks), nb_task)).tolist() for i in range(len(list_algo))]
            # print(ls_model_cost)
            for idx_algo in range(len(list_algo)):
                path_algo = os.path.join(path, list_algo[idx_algo])
                # count_benchmark = 0

                for benchmark_mso in os.listdir(path_algo):
                    count_benchmark = benchmark_mso.split(".")[0]
                    count_benchmark = count_benchmark.split("_")[-1]
                    count_benchmark = int(count_benchmark) - 1

                    model = loadModel(os.path.join(
                        path_algo, benchmark_mso), ls_benchmark[count_benchmark])

                    ls_model_cost[idx_algo][count_benchmark] = model.history_cost[-1]
                    # count_benchmark += 1

            result_table = np.zeros(
                shape=(len(benchmarks), len(list_algo)-1, 3), dtype=int)
            name_row = []
            name_col = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(list_algo)):
                if idx_main_algo != idx_algo:
                    name_row.append(
                        list_algo[idx_main_algo] + " vs " + list_algo[idx_algo])
                    for idx_benchmark in range(len(benchmarks)):
                        result = np.where(ls_model_cost[idx_main_algo][idx_benchmark] > min_value, ls_model_cost[idx_main_algo][idx_benchmark], min_value) \
                            - np.where(ls_model_cost[idx_algo][idx_benchmark] > min_value,
                                       ls_model_cost[idx_algo][idx_benchmark], min_value)

                        result_table[idx_benchmark][count_row][0] += len(
                            np.where(result < 0)[0])
                        result_table[idx_benchmark][count_row][1] += len(
                            np.where(result == 0)[0])
                        result_table[idx_benchmark][count_row][2] += len(
                            np.where(result > 0)[0])
                    count_row += 1
            if combine is True:
                result_table = pd.DataFrame(
                    np.sum(result_table, axis=0), columns=name_col, index=name_row)
        return result_table

    def detail_compare_result(self, min_value=0, round = 100):
        name_row = [str("Task" + str(i + 1))
                    for i in range(len(self.models[0].tasks))]
        name_col = self.label
        data = []
        for model in self.models:
            data.append(model.history_cost[-1])

        data = np.array(data).T
        data = np.round(data, round)
        pre_data = pd.DataFrame(data)
        end_data = pd.DataFrame(data).astype(str)

        result_compare = np.zeros(shape=(len(name_col)), dtype=int).tolist()
        for task in range(len(name_row)):
            argmin = np.argmin(data[task])
            min_value_ = max(data[task][argmin], min_value)
            # for col in range(len(name_col)):
            #     if data[task][col] == data[task][argmin]:
            #         result_compare[col] += 1
            #         end_data.iloc[task][col]= str("(≈)") + pre_data.iloc[task][col].astype(str)
            #     elif data[task][col] > data[task][argmin]:
            #         end_data.iloc[task][col]= str("(-)") + pre_data.iloc[task][col].astype(str)
            #     else:

            for col in range(len(name_col)):
                if data[task][col] <= min_value_:
                    result_compare[col] += 1
                    end_data.iloc[task][col] = str(
                        "(+)") + end_data.iloc[task][col]

        for col in range(len(name_col)):
            result_compare[col] = str(
                result_compare[col]) + "/" + str(len(name_row))

        result_compare = pd.DataFrame([result_compare], index=[
                                      "Compare"], columns=name_col)
        end_data.columns = name_col
        end_data.index = name_row
        end_data = pd.concat([end_data, result_compare])

        # assert data.shape == (len(name_row), len(name_col))

        return end_data

