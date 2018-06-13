__author__ = 'HyNguyen'

def xxx():
    rouge_save = {}
    groups = [85,130,180,220,270,340]
    for group in groups:
        rouge_save[str(group)] = []
        with open("settings.{0}.txt".format(group),mode="r") as f:
            lines = f.readlines()
            for n_method in range(5):
                line_rouge1 = lines[n_method*12 + 3]
                line_rouge2 = lines[n_method*12 + 7]
                line_rougesu4 = lines[n_method*12 + 11]
                rouge1_score = line_rouge1.split()[3]
                rouge2_score = line_rouge2.split()[3]
                rougesu4_score = line_rougesu4.split()[3]
                rouge_save[str(group)].append({"1":rouge1_score, "2":rouge2_score, "su4":rougesu4_score})

    hyhy = ""
    list_rouge_name = ["1","2","su4"]
    for rouge_name in list_rouge_name:
        hyhy += rouge_name + "\n\n"
        for group in groups:
            for i in range(5):
                hyhy+= rouge_save[str(group)][i][rouge_name] +","
            hyhy=hyhy[:-1]+"\n"
        hyhy += "\n"

    with open("result.csv", mode="w") as f:
        f.write(hyhy)

import os
import numpy as np
from collections import OrderedDict

if __name__ == "__main__":

    list_summary = [ "simple-mmr"]
    list_feature = ["unitfidf", "bitfidf","w2v_bow_mean","w2v_sg_mean","glove_mean","d2v_dm","d2v_dbow"]

    list_dir = [name for name in os.listdir("./") if name.find("simple-mmr") != -1 and os.path.isdir(name)]

    fo = open("result_max.csv", mode="w")

    for name in list_dir:
        summary_name, feature_name = name.split("_")
        summary_feature = "{0}_{1}".format(summary_name,feature_name)
        files_result = [filename for filename in os.listdir(summary_feature) if filename.find("output.")!=-1]
        result_dict = {}
        for file_result in files_result:
            with open(summary_feature+"/"+file_result, mode="r") as f:
                lines = f.readlines()
            group = file_result.split(".")[2]
            n_times = int(len(lines)/12)
            result_array = []
            for idx in range(n_times):
                xxx_lines = lines[idx*12: idx*12+12]
                ROUGE1 = xxx_lines[3].split()[3]
                ROUGE2 = xxx_lines[7].split()[3]
                ROUGESU4 = xxx_lines[11].split()[3]
                result_array.append([ np.round(float(ROUGE1),4), np.round(float(ROUGE2),4), np.round(float(ROUGESU4),4)])
            result_dict[int(group)] = np.array(result_array)

        # result_dict = OrderedDict(sorted(result_dict.items(), key=lambda t:t))

        xxx = []
        for key, value in result_dict.iteritems():
            xxx.append(value)
        xxx = np.array(xxx,dtype=np.float32).mean(0)

        idx_max = np.argmax(xxx[:,1])
        final_result = xxx[idx_max]*100
        fo.write("{0},{1},{2},{3},{4}\n".format(summary_name,feature_name,"%.2f"%final_result[0],"%.2f"%final_result[1],"%.2f"%final_result[2]))



        # with open(summary_feature + "/results.csv",mode="w") as f:
        #     for i in range(3):
        #         average_matrix = []
        #         for key,value in result_dict.iteritems():
        #             if summary_name == "simple-mmr":
        #                 f.write("{0},{1},{2},{3},{4},{5}\n".format(key,value[0,i], value[1,i], value[2,i], value[3,i], value[4,i]))
        #                 average_matrix.append([value[0,i], value[1,i], value[2,i], value[3,i], value[4,i]])
        #             else:
        #                 f.write("{0},{1}\n".format(key,value[0,i]))
        #                 average_matrix.append([value[0,i]])
        #         average_matrix = np.array(average_matrix,dtype=np.float32)
        #         average_matrix = np.mean(average_matrix, axis=0)
        #         average_matrix = np.round(average_matrix,4)
        #         if summary_name == "simple-mmr":
        #             f.write("{0},{1},{2},{3},{4},{5}\n".format("average",average_matrix[0], average_matrix[1], average_matrix[2], average_matrix[3], average_matrix[4]))
        #         else:
        #             f.write("{0},{1}\n".format("average",np.round(average_matrix[0],4)))

    fo.close()