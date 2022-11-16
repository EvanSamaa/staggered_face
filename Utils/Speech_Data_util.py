import numpy as np
import math
import os


class Check_Pauses():
    def __init__(self):
        self.filled_pauses = set(["um", "ah", "uh", "eh"])

    def __call__(self, word):
        if word == ".":
            return "SP"  # pause
        if word[1:].lower() in self.filled_pauses:
            return "FP"  # filled pause
        else:
            return "NP"  # no pause
class Sentence_word_phone_parser:
    """
    This class parse a praatscript and a accompanied textgrid into sentences, and store them in easily usable
    Format.
    phone_list, word_list, sentence_list stores lists of phonemes, words and sentences respectively
    phone_to_word and phone_to_sentences use the same index as phone_list, and they store index of the word and sentence
    they point to.
    Similarly word_to_phone, word_to_sentences, sentence_to_words, sentence_to_phones works similarly. points to index
    to the phone, word, sentence lists respectively


    """
    def __init__(self, praat_file_path, text_file_path):
        phone_list, phone_intervals, word_list, word_intervals = self.load_praatoutput(praat_file_path)
        sentence_list, sentence_intervals = self.phone_to_sentences(text_file_path,
                                                               phone_list, phone_intervals, word_list, word_intervals)
        self.phone_list = phone_list
        self.phone_intervals = phone_intervals
        self.phone_to_word = word_list
        self.phone_to_sentence = sentence_list

        self.word_list = []
        self.word_intervals = []
        self.word_to_phone = []
        self.word_to_sentence = []

        self.sentence_list = []
        self.sentence_intervals = []
        self.sentence_to_phone = []
        self.sentence_to_word = []

        # get all word level pointers
        start = 0
        for i in range(1, len(word_list)):
            if word_list[i - 1] != word_list[i]:
                self.word_list.append(word_list[i - 1])
                self.word_intervals.append(word_intervals[i - 1])
                word_to_phone_temp = []
                for j in range(start, i):
                    word_to_phone_temp.append(j)
                start = i
                self.word_to_phone.append(word_to_phone_temp)
                self.word_to_sentence.append(self.phone_to_sentence[i - 1])

            if i == len(word_list) - 1:

                self.word_list.append(word_list[i])
                self.word_intervals.append(word_intervals[i])
                word_to_phone_temp = []
                for j in range(start, i + 1):
                    word_to_phone_temp.append(j)
                start = i
                self.word_to_phone.append(word_to_phone_temp)
                self.word_to_sentence.append(self.phone_to_sentence[i])
        # get all sentence level pointers
        self.sentence_list = []
        for i in range(0, self.phone_to_sentence[-1] + 1):
            self.sentence_list.append(i)
            for j in range(0, len(sentence_intervals)):
                if self.phone_to_sentence[j] == i:
                    self.sentence_intervals.append(sentence_intervals[j])
                    break
        start = 0
        for i in range(1, len(self.phone_to_sentence)):
            if self.phone_to_sentence[i - 1] != self.phone_to_sentence[i]:
                sentence_to_phone_temp = []
                for j in range(start, i):
                    sentence_to_phone_temp.append(j)
                start = i
                self.sentence_to_phone.append(sentence_to_phone_temp)

            if i == len(self.phone_to_sentence) - 1:
                sentence_to_phone_temp = []
                for j in range(start, i + 1):
                    sentence_to_phone_temp.append(j)
                start = i
                self.sentence_to_phone.append(sentence_to_phone_temp)

        start = 0
        for i in range(1, len(self.word_to_sentence)):
            if self.word_to_sentence[i - 1] != self.word_to_sentence[i]:
                sentence_to_word_temp = []
                for j in range(start, i):
                    sentence_to_word_temp.append(j)
                start = i
                self.sentence_to_word.append(sentence_to_word_temp)

            if i == len(self.word_to_sentence) - 1:
                sentence_to_word_temp = []
                for j in range(start, i + 1):
                    sentence_to_word_temp.append(j)
                start = i
                self.sentence_to_word.append(sentence_to_word_temp)
    def load_praatoutput(self, file_name):
        """
        Given a praatscript produced by Jsync, this function loads the praatscript
        and generates timing for words and phonemes
        :param file_name: absolute/relative path to the *_PraatOutput.txt file
        :return: phone_list, phone_intervals, word_list, merged_word_intervals, each
        is of the size (number of phones)
        """
        phone_list = []
        phone_intervals = []
        word_list = []
        stats = []
        f = open(file_name)
        garb = f.readline()
        arr = f.readlines()
        for i in range(0, len(arr)):
            content = arr[i].split("\t")
            start = float(content[0])
            end = float(content[1])
            phone = content[21]
            word = content[26]
            phone_list.append(phone)
            word_list.append(word)
            phone_intervals.append([start, end])
        merged_word_intervals = []
        prev_word = word_list[0]
        merged_word_intervals.append([phone_intervals[0][0]])
        prev_k = 1
        for i in range(1, len(word_list)):
            if word_list[i] != prev_word:
                merged_word_intervals[-1].append(phone_intervals[i - 1][1])
                for j in range(0, prev_k - 1):
                    merged_word_intervals.append(merged_word_intervals[-1])
                prev_word = word_list[i]
                merged_word_intervals.append([phone_intervals[i][0]])
                prev_k = 0
            if i == len(word_list) - 1:
                merged_word_intervals[-1].append(phone_intervals[i - 1][1])
                for j in range(0, prev_k):
                    merged_word_intervals.append(merged_word_intervals[-1])
            prev_k = prev_k + 1
        return phone_list, phone_intervals, word_list, merged_word_intervals
    def phone_to_sentences(self, text_file_path, phone_list, phone_intervals, word_list, word_intervals):
        # here I will label each phone to be part of a sentence
        punctuations = set([",", ".", "!", "?", ";", "\n", "...", "......", ":", ""])
        pause_checker = Check_Pauses()
        text_file = open(text_file_path, "r").readlines()
        # process the script to add punctuation to it
        all_script = " ".join(text_file)
        all_script = self.detag(all_script)
        raw_script = all_script.split(" ")
        raw_script_copy = []
        for i in range(0, len(raw_script)):
            end_with_nextline = False
            switched_sentece = False
            if len(raw_script) <= 0:
                continue
            if raw_script[i][-1] == "\n":
                end_with_nextline = True
            raw_script[i] = raw_script[i].strip()
            if len(raw_script[i]) == 0:
                continue
            if raw_script[i][0] == "<" and raw_script[i][-1] == ">":
                continue
            if raw_script[i] in punctuations:
                raw_script_copy.append(raw_script[i])
                switched_sentece = True
            elif raw_script[i][0] == "<" and raw_script[i][-1] in punctuations:
                raw_script_copy.append(raw_script[i][-1])
                switched_sentece = True
            elif len(raw_script[i]) >= 2 and raw_script[i][-1] in punctuations and raw_script[i][-1] == raw_script[i][0]:
                raw_script_copy.append(raw_script[i][1:-1])
            elif raw_script[i][-1] in punctuations:
                raw_script_copy.append(raw_script[i][:-1])
                raw_script_copy.append(raw_script[i][-1])
                switched_sentece = True
            elif raw_script[i][0] in punctuations:
                raw_script_copy.append(raw_script[i][0])
                raw_script_copy.append(raw_script[i][1:])
                switched_sentece = True
            else:
                raw_script_copy.append(raw_script[i])
            if not switched_sentece and end_with_nextline:
                raw_script_copy.append(",")
        raw_script = raw_script_copy
        sentence_number = 0
        sentence_number_tags = []
        sentence_intervals = []

        # two pointers, one for the textfile, the other one for phoneme list
        pt1 = 0
        pt2 = 0
        start = 0
        end = 0
        prev_word = "."
        # first the very first word in the list
        for i in range(len(word_list)):
            if word_list[i] != ".":
                prev_word = word_list[i][1:].lower()
                break

        # iteratively give each phoneme a sentence tag
        added = False
        while pt2 < len(phone_list):
            word_with_punctuation = raw_script[pt1].lower()
            # remove the underscore at the beginning of words
            if len(word_list[pt2]) > 1:
                word_from_praatscript = word_list[pt2].lower()[1:]
            else:
                word_from_praatscript = word_list[pt2].lower()

            if word_from_praatscript != "." and word_from_praatscript != prev_word:
                pt1 += 1
                pt1 = min(pt1, len(raw_script) - 1)
                word_with_punctuation = raw_script[pt1].lower()
                if word_with_punctuation in punctuations:
                    pt1 += 1
                    pt1 = min(pt1, len(raw_script) - 1)
                    prev_word = raw_script[pt1].lower()
                    if not added:
                        sentence_number += 1
                else:
                    prev_word = word_from_praatscript
                    added = False

            elif word_from_praatscript != "." and word_from_praatscript == prev_word:
                added = False

            # if the current word form the praatscript label is silence, then go to a new sentence
            if word_from_praatscript == ".":
                # if the file starts with a pause, don't do anything
                if sentence_number == 0 and pt2 == 0:
                    pass
                elif word_intervals[pt2][1] - word_intervals[pt2][0] > 0.4:
                    # otherwise increment the sentence number by 1
                    sentence_number += 1
                    added = True
            sentence_number_tags.append(sentence_number)
            pt2 += 1
        # using the sentence tag to give each phoneme a sentence interval
        start = word_intervals[0][0]
        current_sentence = 0
        prev_index = 0
        for i in range(0, len(word_list)):
            if sentence_number_tags[i] != current_sentence or i == len(word_list) - 1:
                end = word_intervals[i][0]
                if i == len(word_list) - 1:
                    for j in range(prev_index, i + 1):
                        sentence_intervals.append([start, end])
                else:
                    for j in range(prev_index, i):
                        sentence_intervals.append([start, end])
                start = end
                prev_index = i
                current_sentence += 1

        return sentence_number_tags, sentence_intervals
    def detag(self, input_text):
        new_string = ""
        in_tag = False
        for i in range(0, len(input_text)):
            if input_text[i] == "<":
                in_tag = True
                continue
            elif input_text[i] == ">":
                in_tag = False
                continue
            if (not in_tag):
                new_string = new_string + input_text[i]
        return new_string
class XSampa_phonemes_dicts():
    """
    A dictionary for all the phones that shows up in the XSampa pronunciation dictionary.
    """
    def strip(self, phone):
        try:
            float(phone[-1])
            return phone[:-1]
        except:
            return phone
    def __init__(self):
        self.vocabs = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                  'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH',
                  'UW', 'V', 'W', 'Y', 'Z', 'ZH', "sil", "sp"])
        self.vowels = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
                  'IH', 'IY', 'OW', 'OY', 'UH', 'UW', ])
        self.voiced = set(['M', 'N', "L", "NG"]).union(self.vowels)
        self.consonants = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG',
                              'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'])
        self.consonants_no_jaw = self.consonants
        self.lip_closer = set(["B", "F", "M", "P", "S", "V"])
        self.lip_rounder = set(["B", "F", "M", "P", "V"])
        self.nasal_obtruents = set(['L', 'N', 'NG', 'T', 'D', 'G', 'K', 'F', 'V', 'M', 'B', 'P'])
        self.fricative = set(["S", "Z", "ZH", "SH", "CH", "F", "V", 'TH'])
        self.plosive = set(["P", "B", "D", "T", "K", "G"])
        self.lip_heavy = set(["W", "OW", "UW", "S", "Z", "Y", "JH", "OY"])
        self.sibilant = set(["S", "Z", "SH", "CH", "ZH"])


if __name__ == "__main__":
    input_folder = "F:/MASC/JALI_neck/data/neck_rotation_values/sarah_connor"
    input_file_name = "audio"
    input_praat_script = os.path.join(input_folder, input_file_name + "_PraatOutput.txt")
    input_txt_script = os.path.join(input_folder, input_file_name + ".txt")
    tim = Sentence_word_phone_parser(input_praat_script, input_txt_script)
    # for i in range(len(tim.phone_list)):
    #     print(tim.phone_list[i], tim.phone_intervals[i], tim.phone_to_sentence[i])
    for i in range(len(tim.word_list)):
        print(tim.word_list[i], tim.word_intervals[i], tim.word_to_sentence[i])


