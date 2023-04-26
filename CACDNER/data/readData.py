import codecs


def read_data(file):
    with open(file,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")
    all_text=[]
    all_label = []
    text = []
    label = []
    for data in all_data:
        #去掉两边空格
        if data == "":
            all_text.append(text)
            all_label.append(label)
            text=[]
            label =[]
        else:
            t,l = data.split()
            #assert len(word)>=2
            text.append(t)
            label.append(l)
    #循环走完判断防止最后一个句子没有进入到句子集合

    return all_text, all_label