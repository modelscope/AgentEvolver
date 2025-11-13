import json
import sys


def time_cost_analysis(file_path: str):
    result_time = {}
    result_len = {}
    all_time = []
    all_len = []
    with open(file_path) as f:
        for line in f:
            if not line:
                continue

            if "info_dict=" not in line:
                continue

            if "time_cost" not in line:
                continue

            line = line.split("info_dict=")[1]
            line_dict = json.loads(line)
            act_step = line_dict["act_step"]
            time_cost = line_dict["time_cost"]
            char_len = len(line_dict["llm_output"]["content"])

            if act_step not in result_time:
                result_time[act_step] = []
            result_time[act_step].append(time_cost)
            all_time.append(time_cost)

            if act_step not in result_len:
                result_len[act_step] = []
            result_len[act_step].append(char_len)
            all_len.append(char_len)

    for k, v in sorted(result_time.items(), key=lambda x: x[0]):
        len_list = result_len[k]
        print(f"act_step={k} time_cost={sum(v) / len(v):.2f} "
              f"count={len(v)} "
              f"len={sum(len_list) / len(len_list):.2f} "
              f"efficient={sum(v) * 1000 / sum(len_list):.2f}")

    print(f"time_cost={sum(all_time) / len(all_time):.2f} "
          f"count={len(all_time)} "
          f"len={sum(all_len) / len(all_len):.2f} "
          f"efficient={sum(all_time) * 1000 / sum(all_len):.2f}")


if __name__ == "__main__":
    for file in sys.argv[1:]:
        time_cost_analysis(file)
