import xml.etree.ElementTree as ET
import os
import sys
import re
import jieba
import math
import json
import random
import time

def get_name(basename):
    t = time.localtime()
    v = time.strftime("%y%m%d%H%M%S",t)
    return basename+'_'+v
def XML2TXT_extract(root_dic,dis_file=None):
    fl = os.listdir(root_dic)
    if dis_file == None:
        dis_file = get_name('DOC_SEG')+".txt"
    data_file = open(dis_file,'w',encoding='utf-8')
    count = 0
    fl = [root_dic+'/'+f for f in fl]

    length_map = {}
    for file in fl:
        if not file.endswith('.xml'):
            try:
                fl_a = os.listdir(file)
                fl_a = [file+'/'+f for f in fl_a]
                fl += fl_a
            except Exception:
                pass
        else:

            try:
                if (count+1) % 100 == 0:
                    print("[INFO] Now reading file : %d "%(count+1))
                stree = ET.ElementTree(file = file)
                qw = next(stree.iter('QW')).attrib['value']
                sens = re.split(r"[,、，。；：\n]",qw)
                patterns = [
                    r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
                    r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
                    # r"[\s]",
                    r"[a-zA-Z《》【】（）\s]+",
                ]
                res = []
                for sen in sens:
                    for p in patterns:
                        sen = re.sub(p,'',sen)
                    if len(sen)<3:
                        continue
                    cutres = jieba.lcut(sen)
                    lc = len(cutres)
                    if lc not in length_map:
                        length_map[lc]=0
                    length_map[lc] += 1
                    sen = ' '.join(cutres)
                    res.append(sen)

                data_file.write(','.join(res))
                data_file.write('\n')

            except StopIteration:
                pass
            #
            # if count>10:
            #     break

            count += 1

    print("[INFO] 信息提取完毕，总共提取文书%d篇 句子长度统计如下"%count)


    # for kv in enumerate(length_map):

    ll = sorted(length_map.keys(),key = lambda x:x)
    for k in ll:
        print("k = %d : %d"%(k,length_map[k]))

    data_file.close()


def sent_format(sentence):
    patterns = [
        r"\(\d+\)[，、．.,\s]*",
        r"\([一二三四五六七八九十]+\)[，、．.,\s]*",
        r"（[一二三四五六七八九十]+）[，、．.,\s]*",
        r"（\d+）[，、．.,\s]*",
        r"[一二三四五六七八九十]+[，、．.,\s]+",
        r"\d+[，、．.,\s]+",
        r"[⑼]",
        "被告人的供述与辩解：*",
        "被告人.*?供述([与和]辩解)?：*",
        "证人证言及辨认笔录：*",
        "\A(证人)*.*?(证词|证言)?证实：*",
        "证人.*?证言：*",
        ".*?的证言(。|，)证实",
        "未到庭笔录证明",
        "辨认笔录及照片",
        "证明：",
        "经审理查明"
    ]
    for p in patterns:
        sentence = re.sub(p,'',sentence)
    return sentence
if __name__ == '__main__':
    arg = sys.argv
    XML2TXT_extract(arg[1])