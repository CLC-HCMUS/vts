__author__ = 'HyNguyen'

import itertools

def change_name(token):
    if token == "unigramtfidf":
        return "BOW TF-IDF"
    elif token == "w2vmdssim":
        return "Word2Vec"
    elif token == "pvdm":
        return "PV DM"
    elif token == "pvdbow":
        return "PV DBOW"
    elif token == "trcomparer":
        return "TR-Comparer"
    elif token == "filtered":
        return "Filtered"
    elif token == "sktmdssim":
        return "Skip-Thought Vectors"
    else:
        return "Hyhy"


def check_exist_model(exist_model={}, element = ""):
    tokens_element = element.split("-")
    for key,value in exist_model.iteritems():
        tokens = key.split("-")
        if len(set(tokens) & set(tokens_element)) == len(tokens_element):
            if len(tokens) == len(tokens_element):
                return value
    return None


def change_feature_name(feature_name):
    result = []
    for token in feature_name.split("-"):
        changed_token = change_name(token)
        result.append(changed_token)
    return " \& ".join(result)

def create_list_aggregate(aggregate_features = []):
    list_feature = []
    for aggregate_feature in aggregate_features:
        for L in range(2, len(aggregate_feature)+1):
            for subset in itertools.combinations(aggregate_feature, L):
                list_feature.append("-".join(subset))
    return list_feature


def create_data(xxx = []):
    result = []
    groups = [["trcomparer","filtered"],["unigramtfidf"],["pvdm","pvdbow","sktmdssim"]]
    group_1 =groups[0]
    group_2 =groups[1]
    group_3 =groups[2]
    if len(xxx) == 3:
        for xxx_1 in group_1:
            for xxx_2 in group_2:
                for xxx_3 in group_3:
                    result.append(xxx_1+"-"+xxx_2+"-"+xxx_3)

    elif len(xxx) == 2:
        group_a = groups[xxx[0]]
        group_b = groups[xxx[1]]
        for xxx_1 in group_a:
            for xxx_2 in group_b:
                result.append(xxx_1+"-"+xxx_2)

    return result

if __name__ == "__main__":

    list_feature = []
    aggregate_feature = ["filtered","trcomparer","unigramtfidf","pvdm","pvdbow","w2vmdssim","sktmdssim"]
    except_aggregate_feature1 = ["trcomparer","filtered","unigramtfidf"]
    except_aggregate_feature2 = ["trcomparer","filtered","w2vcbow","pvdm","pvdbow","sktduc04sim"]
    except_aggregate_feature3 = ["unigramtfidf","w2vcbow","pvdm","pvdbow","sktduc04sim"]
    except_aggregate_feature = create_list_aggregate([except_aggregate_feature1,except_aggregate_feature2,except_aggregate_feature3])

    piority = ""
    for L in range(1, 2):
        for subset in itertools.combinations(aggregate_feature, L):
            list_feature.append("-".join(subset))

    with open("result_max.csv", mode="r") as f:
        lines = f.readlines()

    exist_model = {}

    for line in lines:
        tokens = line.strip().split(",")
        exist_model[tokens[1]] = ("%.2f"%float(tokens[2]), "%.2f"%float(tokens[3]), "%.2f"%float(tokens[4]))

    fo = open("check.csv", mode="w")

    list_feature = create_data([0,1,2])

    for feature in list_feature:
        # if piority != "":
        #     if feature.find(piority) == -1:
        #         continue
        # if feature in except_aggregate_feature:
        #     continue

        rouge1, rouge2, rougesu4 = check_exist_model(exist_model, feature)
        changed_feature = change_feature_name(feature)
        if len(changed_feature) >= 25 and len(changed_feature) < 50:
            print "\parbox[c][1.5cm][c]{6.5cm}{",changed_feature, "} &" , rouge1, " & " , rouge2,  " & " , rougesu4, "\\\\ \\hline"
        elif len(changed_feature) >= 50 and len(changed_feature) < 75:
            print"\parbox[c][2cm][c]{6.5cm}{",changed_feature, "} &" , rouge1, " & " , rouge2,  " & " , rougesu4, "\\\\ \\hline"
        elif len(changed_feature) >= 75 and len(changed_feature) < 100:
            print "\parbox[c][2.75cm][c]{6.5cm}{",changed_feature, "} &" , rouge1, " & " , rouge2,  " & " , rougesu4, "\\\\ \\hline"
        else:
            print "\parbox[c][1cm][c]{6.5cm}{",changed_feature, "} &" , rouge1, " & " , rouge2,  " & " , rougesu4, "\\\\ \\hline"
        fo.write("{0},{1},{2},{3}\n".format(changed_feature,rouge1,rouge2,rougesu4))
    fo.close()