
# coding: utf-8

import re2 as re
import json
import os
import collections
import unicodedata
import sys
import tqdm
from ufal.udpipe import Model, Pipeline



data_dir_list = [sys.argv[1]]
output_dir = sys.argv[2]
tmp_dir = 'tmp'
results_dir = 'results'
udpipe_model = sys.argv[3]
files_n=100 #



postfix = os.path.split(data_dir_list[-1])[-1]
tmp_dir = os.path.join(output_dir, tmp_dir)
results_dir = os.path.join(output_dir, results_dir)
results_dir = os.path.join(results_dir, postfix)
tmp_dir = os.path.join(tmp_dir, postfix)
os.makedirs(tmp_dir, exist_ok = True)
os.makedirs(results_dir, exist_ok = True)

json_files = []
for data_dir in data_dir_list:
    json_files.extend([os.path.join(data_dir,file_path) for file_path in  os.listdir(data_dir)])



topics = []
for file in json_files:
    for line in open(file, 'r', encoding="utf-8"):
        topics.append(json.loads(line))



texts = [topic['text'] for topic in topics]



def counters_merge(counters):
    if len(counters) < 3:
        share_counter = collections.Counter()
        for counter in counters:
            share_counter += counter
        return share_counter
    else:
        split_point = len(counters)//2
        l_c = counters_merge(counters[:split_point])
        r_c = counters_merge(counters[split_point:])
        return l_c + r_c


def preproc(text_list, modelfile):
    model = Model.load(modelfile)
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'horizontal')
    combining_characters = dict.fromkeys([c for c in range(sys.maxunicode)
                                                   if unicodedata.combining(chr(c))])
    sanitized_texts = []
    for text in tqdm.tqdm(text_list):
        ref_name_tag_regexp = r'<ref name=.*?>'
        div_class_tag_regexp = r'<div class=.*?>'
        other_tag_regexp = r'</?\s?[^(math)(/math)(…)\.А-Яа-я0-9\"].{0,150}?>'
        repeat_math_tag_regexp = r'(</?math>[^А-Яа-я]{0,150}?)</?math>'
        math_tag_regexp = r'</?math*?>'
        url_regexp = r'(?i)\b((?:(https?|ftp)://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb]))'
        text = re.sub(ref_name_tag_regexp, r" ", text) #remove <ref name=
        text = re.sub(div_class_tag_regexp, r" ", text) #remove <div class=
        text = re.sub(other_tag_regexp, r" ", text) #remove other tags
        text = '<neli>'.join(text.split('\n'))
        text = re.sub(repeat_math_tag_regexp, r" ", text) # degree of nesting 3 math tags
        text = re.sub(repeat_math_tag_regexp, r" ", text)
        text = re.sub(repeat_math_tag_regexp, r" ", text)
        text = '\n'.join(text.split('<neli>'))
        text = re.sub(math_tag_regexp, r" ", text) #remove all math tags
        #----------------
        text = re.sub(r'…', r'...', text) # … -> ... (2026)
        text = re.sub(r'\?{1,}![\?!]+', r'?!', text) # ?!???!! -> ?!
        text = re.sub(r'!{1,}\?[\?!]+', r'?!', text) # ?!???!! -> ?!
        text = re.sub(r'!{3,}', r'!!', text) # !!!!!!!! -> !!
        text = re.sub(r'\?{3,}', r'??', text) # ???? -> ??
        text = re.sub(r'\.{4,}', r'...', text) # ....... -> ...
        text = re.sub(r'[“”«»]', r'"', text) #
        text = re.sub(r"’", r"'", text) #
        text = re.sub(r"[`']{2,}", r'"', text) #
        text = re.sub(r'[‐‑‒–—―-]{1,}', r'-', text) # (2010)(2011)(2012)(2013)(2014)(2015)(2016) -> - (2012)
        #----------------
        text = re.sub(url_regexp, r" bracketURLbracket ", text)
        text = re.sub(r"\s+", r" ", text)
        text = re.sub(r'­', r'', text) # remove u00ad
        sentences = pipeline.process(text).split('\n')
        sanitized_sentences = []
        for sentence in sentences:
            tokens = sentence.split()
            tokens = [token if token != 'bracketURLbracket' else '<URL>' for token in tokens]
            sanitized_tokens = []
            for token in tokens:
                decomposed_token = unicodedata.normalize('NFD', token)
                sanitized_token = decomposed_token.translate(combining_characters)
                # Move bak reversed N with hat
                sanitized_token = ''.join(ch_san if ch not in 'йЙ' else ch for ch, ch_san in zip(token, sanitized_token))
                sanitized_token = unicodedata.normalize('NFC', sanitized_token)

                sanitized_tokens.append(sanitized_token)
            sanitized_sentences.append(sanitized_tokens)
        sanitized_texts.append(sanitized_sentences)
    return sanitized_texts

sanitized_texts = preproc(texts, udpipe_model)


articles_per_file = max(len(sanitized_texts)//files_n,1)
split_points = list(range(0,len(sanitized_texts),articles_per_file))
split_points[-1] = None

def write_articles_to_file(artcs, file_path, eoa_tag):
    with open(file_path,'wt', encoding="utf-8") as f:
        for artc in artcs[:-1]:
            for line in artc[:-1]:
                f.write(' '.join(line)+'\n')
            f.write(' '.join(artc[-1]))
            f.write(eoa_tag+'\n')
        # write last article without tag
        for line in artcs[-1][:-1]:
                f.write(' '.join(line)+'\n')
        # write last line without newline
        f.write(' '.join(artcs[-1][-1]))


for indx, (start, end) in enumerate(zip(split_points[:-1], split_points[1:]), 1):
    file_path = os.path.join(results_dir, 'prts_{}.txt'.format(str(indx).zfill(6)))
    if sanitized_texts[start:end]:
        write_articles_to_file(sanitized_texts[start:end], file_path, '<NEXT_PAPER>')
