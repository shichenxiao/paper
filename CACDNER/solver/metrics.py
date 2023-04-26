class Metrics(object):
    """用于实体级别评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, std_tags, predict_tags):
        """
        初始化对照文件中的标签列表、预测文件的标签列表、以及
        :param std_tags:
        :param predict_tags:
        """
        #将按句的标签列表转化成按字的标签列表 如 [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.std_tags = flatten_lists(std_tags)     #将已知标签和预测的标签拼成列表
        self.predict_tags = flatten_lists(predict_tags)

        #计数器
        self.std_entity_counter = self.count_entity_dict(self.std_tags)  #标准结果的各实体个数
        self.predict_entity_counter = self.count_entity_dict(self.predict_tags)  #预测结果的各实体个数
        print("标准各实体个数", self.std_entity_counter)
        print("预测各实体个数", self.predict_entity_counter)

        self.std_entity_number = self.count_entity(self.std_tags)    #标准结果的实体总个数
        # self.predict_entity_number = self.count_entity(self.predict_tags)  #预测结果的实体总个数
        print("标准实体数", self.std_entity_number)
        # print("预测实体数", self.predict_entity_number)

        self.corrent_entity_number = self.count_correct_entity()
        print("正确的实体", self.corrent_entity_number)

        self.entity_set = set(self.std_entity_counter)
        print("实体集合", self.entity_set)

        # 计算精确率
        self.precision_scores = self.cal_precision()
        print("各个实体的准确率", self.precision_scores)

        # 计算召回率
        self.recall_scores = self.cal_recall()
        print("各个实体的召回率", self.recall_scores)

        # 计算F1分数
        self.f1_scores = self.cal_f1()
        print("各个实体的f1值", self.f1_scores)

        # 计算加权均值
        self.wighted_average = self._cal_wighted_average()
        print("各项指标的加权均值", self.wighted_average)

    def cal_precision(self):
    #计算每类实体的准确率
        precision_scores = {}
        #某实体正确的个数  /  预测中某实体所有的个数
        for entity in self.entity_set:  #这里计算结果可能有问题
            if entity in self.predict_entity_counter:
                precision_scores[entity] = self.corrent_entity_number.get(entity, 0) / max(1e-10,
                                                                                           self.predict_entity_counter[
                                                                                               entity])
            else:
                precision_scores[entity] = 0
        return precision_scores

    def cal_recall(self):
    #计算每类尸体的召回率
        recall_scores ={}
        for entity in self.entity_set:
            recall_scores[entity] = self.corrent_entity_number.get(entity, 0) / max(1e-10, self.std_entity_counter[entity])
        return recall_scores


    def cal_f1(self):
    #计算f1值
        f1_scores = {}
        for entity in self.entity_set:
            p, r = self.precision_scores[entity], self.recall_scores[entity]
            f1_scores[entity] = 2 * p * r / (p + r + 1e-10)
        return f1_scores

    def report_scores(self):
        """
        将结果用表格的形式打印出来
        :return:
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))
        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个实体的p, r, f
        for entity in self.entity_set:
            print(row_format.format(
                entity,
                self.precision_scores[entity],
                self.recall_scores[entity],
                self.f1_scores[entity],
                self.std_entity_counter[entity]   #这部分是support的值
            ))
        #计算并打印平均值
        avg_metrics = self._cal_wighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            self.std_entity_number
        ))

    def _cal_wighted_average(self):
        #计算加权均值

        weighted_average = {}
        total = self.std_entity_number  #标准实体的总数

        #计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for entity in self.entity_set:
            size = self.std_entity_counter[entity]  #标准结果各个实体的个数
            weighted_average['precision'] += self.precision_scores[entity] * size
            weighted_average['recall'] += self.recall_scores[entity] * size
            weighted_average['f1_score'] += self.f1_scores[entity] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

	#以下注释掉的代码为将B-company E-book这种BMSE格式符合情况但标签前后不一致，不计入预测结果的实体数中
    # def count_entity_dict(self, tag_list):
    #     """
    #     计算每个实体对应的个数，注意BME、BE、S结构才为实体，其余均不计入,B-company和E-book也不计入
    #     :param tag_list:
    #     :return:
    #     """
    #     enti_dict = {}
    #     flag = 0  # 初始状态设置为0
    #     B_word = ''  #初始状态B设为空
    #     for tag in tag_list:
    #         if 'B-' in tag and flag == 0:  #当B-出现时，将状态变为1
    #             flag = 1
    #             B_word = tag[2:]
    #         if 'M-' in tag and flag == 1:
    #             M_word = tag[2:]
    #             if M_word != B_word:   #当M和B标签不同时，不为实体将B_word设为空
    #                 B_word = ''
    #                 flag = 0
    #         if 'E-' in tag and flag == 1: #E前有B则可以判定为一个实体
    #             flag = 0  #状态回归初始
    #             E_word = tag[2:]
    #             tag = tag[2:]
    #             if E_word == B_word:
    #                 if tag not in enti_dict:
    #                     enti_dict[tag] = 1
    #                 else:
    #                     enti_dict[tag] += 1
    #             B_word = ''
    #         if 'S-' in tag:  #当S-出现，直接加一
    #             B_word = ''
    #             flag = 0
    #             tag = tag[2:]
    #             if tag not in enti_dict:
    #                 enti_dict[tag] = 1
    #             else:
    #                 enti_dict[tag] += 1
    #         if 'O' in tag: #出现O-时，将状态设为0 B_word设为0
    #             B_word = ''
    #             flag = 0
    #     return enti_dict

    def count_entity_dict(self, tag_list):
        """
        计算每个实体对应的个数，只有 B-I 结构才被视为一个实体
        :param tag_list: List[str]，BIO 标注的实体序列
        :return: Dict[str, int]，每种实体类型出现的次数
        """
        enti_dict = {}
        flag = 0  # 初始状态设置为0
        enti_type = None  # 初始实体类型设置为 None
        for tag in tag_list:
            if 'B-' in tag and flag == 0:  # 当 B 开始，将状态变为 1
                flag = 1
                enti_type = tag[2:]
            if 'I-' in tag and flag == 1:  # I 说明这个实体还在进行中
                if tag[2:] != enti_type:
                    # 实体类型出现了不一致的情况，说明实体结束了
                    if enti_type not in enti_dict:
                        enti_dict[enti_type] = 1
                    else:
                        enti_dict[enti_type] += 1
                    enti_type = None  # 重新开始一个实体
                    flag = 0
                else:
                    # 实体类型一致，继续识别
                    pass
            if tag in ['O', 'B-']:  # 如果是 O/B-, 那么实体结束了
                if enti_type is not None:
                    if enti_type not in enti_dict:
                        enti_dict[enti_type] = 1
                    else:
                        enti_dict[enti_type] += 1
                    enti_type = None
                    flag = 0  # 状态回归初始
        return enti_dict

    def count_correct_entity(self):
        """
        计算每种实体被正确预测的个数
        B: 开始标记
        I: 内部标记
        O: 非实体标记
        :return:
        """
        correct_enti_dict = {}
        prev_tag = 'O'
        entity_length = 0
        for std_tag, predict_tag in zip(self.std_tags, self.predict_tags):
            if std_tag == 'O':
                if entity_length > 0:
                    entity_type = prev_tag[2:]
                    if entity_type not in correct_enti_dict:
                        correct_enti_dict[entity_type] = 1
                    else:
                        correct_enti_dict[entity_type] += 1
                    entity_length = 0
                prev_tag = 'O'
            elif 'B-' in std_tag:
                if entity_length > 0:
                    entity_type = prev_tag[2:]
                    if entity_type not in correct_enti_dict:
                        correct_enti_dict[entity_type] = 1
                    else:
                        correct_enti_dict[entity_type] += 1
                entity_length = 1
                prev_tag = std_tag
            elif 'I-' in std_tag:
                if std_tag == predict_tag:
                    entity_length += 1
                    prev_tag = std_tag
                else:
                    entity_length = 0
                    prev_tag = 'O'
        if entity_length > 0:
            entity_type = prev_tag[2:]
            if entity_type not in correct_enti_dict:
                correct_enti_dict[entity_type] = 1
            else:
                correct_enti_dict[entity_type] += 1
        return correct_enti_dict

    def count_entity(self, tag_list):
        """
        计算标准列表中的实体个数，因为标准结果中无错误分类，所以实体的个数可以直接计算为B和I标签数目之和
        :return:
        """
        entity_count = 0  # 记录实体数量
        for i in range(len(tag_list)):  # 遍历所有标签
            if tag_list[i][0] == 'B':  # 如果当前标签以'B'开头，说明找到了一个实体
                entity_count += 1
                # 继续查找以'I'开头的标签，直到查找到不是以'I'开头的标签或者到达序列的末尾
                j = i + 1
                while j < len(tag_list) and tag_list[j][0] == 'I':
                    j += 1
                # 更新i的值，跳过已经查找过的标签
                i = j - 1
        return entity_count


def flatten_lists(lists):
    """
    将列表的列表拼成一个列表
    :param lists:
    :return:
    """
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list