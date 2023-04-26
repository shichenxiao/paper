
tags = [("B-PER","I-PER"),("B-LOC","I-LOC"),("B-ORG","I-ORG"),("B-MISC","I-MISC")]
def find_tag(labels,B_label="B-PER",I_label="I-PER"):
    result = []
    if isinstance(labels,str): # 如果labels是字符串
        labels = labels.strip().split() # 将labels进行拆分
        labels = ["O" if label =="0" else label for label in labels] # 如果标签是O就就是O，否则就是label
        # print(labels)
    for num in range(len(labels)): # 遍历Labels
        if labels[num] == B_label:
            song_pos0 = num # 记录B_SONG的位置
        if labels[num] == I_label and labels[num-1] == B_label: # 如果当前lable是I_SONG且前一个是B_SONG
            lenth = 2 # 当前长度为2
            for num2 in range(num,len(labels)): # 从该位置开始继续遍历
                if labels[num2] == I_label and labels[num2-1] == I_label: # 如果当前位置和前一个位置是I_SONG
                    lenth += 1 # 长度+1
                if labels[num2] == "O": # 如果当前标签是O
                    result.append((song_pos0,lenth)) #z则取得B的位置和长度
                    break # 退出第二个循环
    return result


def find_all_tag(labels):

    result = {}
    for tag in tags:
        res = find_tag(labels,B_label=tag[0],I_label=tag[1])
        result[tag[0].split("-")[1]] = res # 将result赋值给就标签
    return result

# res = find_all_tag(pre)

def Precision(pre_labels,true_labels):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    pre = []
    if isinstance(pre_labels,str):
        pre_labels = pre_labels.strip().split() # 字符串转换为列表
        pre_labels = ["O" if label =="0" else label for label in pre_labels]
    if isinstance(true_labels,str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label =="0" else label for label in true_labels]

    pre_result = find_all_tag(pre_labels) # pre_result是一个字典，键是标签，值是一个元组，第一位是B的位置，第二位是长度
    for name in pre_result: # 取得键，也就是标签
        for x in pre_result[name]: # 取得值：也就是元组，注意元组可能有多个
            if x: # 如果x存在
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]: # 判断对应位置的每个标签是否一致
                    pre.append(1) # 一致则结果添加1
                else:
                    pre.append(0) # 不一致则结果添加0
    if len(pre) != 0:
        return sum(pre) / len(pre)
    else:
        return 0 #为1的个数/总个数




def Recall(pre_labels,true_labels):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    recall = []
    if isinstance(pre_labels,str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label =="0" else label for label in pre_labels]
    if isinstance(true_labels,str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label =="0" else label for label in true_labels]

    true_result = find_all_tag(true_labels)
    for name in true_result: # 取得键，也就是标签，这里注意和计算precision的区别，遍历的是真实标签列表
        for x in true_result[name]: # 以下的基本差不多
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    recall.append(1)
                else:
                    recall.append(0)
    if len(recall) != 0:
        return sum(recall) / len(recall)
    else:
        return 0


def F1_score(precision,recall):

    if (precision + recall) != 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0 # 有了precision和recall，计算F1就简单了