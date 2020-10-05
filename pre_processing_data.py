from datetime import datetime
import random
from keras.preprocessing.sequence import pad_sequences
from underthesea import pos_tag
import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tqdm import tqdm, trange
from underthesea import word_tokenize
import pandas as pd
import io
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle
import argparse
device = torch.device("cpu")
from transformers import RobertaConfig, RobertaModel,ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,BertPreTrainedModel
from fairseq.models.roberta import RobertaModel
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from vncorenlp import VnCoreNLP
NUM_WORDS = 1500
def getData(file_path):
    data = pd.read_csv(file_path)
    return data
dic={}

def getLabels(data):
    file_path = 'data_intent_17_update.csv'
    d = getData(file_path)
    intents = d.intent.unique()
    texts = data.sentence
    for i,intent in enumerate(intents):
        dic[intent] = i
    labels = data.intent.apply(lambda x:dic[x])
    return texts,labels
def txtTokenizer(data):
    sentences = data.sentence
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    return tokenizer,word_index
def processing_cnn_17(test_sentence, model):
    file_path = 'data_intent_17_update.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256, dtype="long", truncating="post", padding="post")
    import time
    start_time = time.time()
    # with open('model_cnn.pkl', 'rb') as fp:
    #     model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent
def processing_cnn_45(test_sentence, model):
    file_path = 'data_intent_45.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256, dtype="long", truncating="post", padding="post")
    import time
    start_time = time.time()

    # with open('model_cnn_45_intents.pkl', 'rb') as fp:
    #     model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent
def processing_lstm_17(test_sentence):
    file_path = 'data_intent_17_update.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256)
    import time
    start_time = time.time()
    with open('model_lstm.pkl_', 'rb') as fp:
        model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent
def processing_lstm_45(test_sentence):
    file_path = 'data_intent_45.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256)
    import time
    start_time = time.time()
    with open('model_lstm_45_intents.pkl_', 'rb') as fp:
        model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    for x in unique_list:
        return x

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# def processing_phobert_17(test_sentence):
#     MAX_LEN = 256
#     file_path = 'data_intent_17_update.csv'
#     data = getData(file_path)
#     intents = data.intent.unique()
#     vocab = Dictionary()
#     vocab.add_from_file("../Demo/models/PhoBERT_base_transformers/dict.txt")
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--bpe-codes',
#                         default="../Demo/models/PhoBERT_base_transformers/bpe.codes",
#                         required=False,
#                         type=str,
#                         help='path to fastBPE BPE'
#                         )
#     args, unknown = parser.parse_known_args()
#     bpe = fastBPE(args)
#
#     # Load the dictionary
#
#
#     with open('model_extraction_phobert.pt', 'rb') as fp:
#         model_phoBert = torch.load(fp, map_location="cpu")
#     with open('model_phobert.pt', 'rb') as fp:
#         model = torch.load(fp, map_location="cpu")
#     sent = '<s> ' + bpe.encode(test_sentence) + ' </s>'
#     sent_ids = vocab.encode_line(sent, append_eos=False, add_if_not_exist=False).long().tolist()
#     tokenized_sentence = pad_sequences([sent_ids], maxlen=MAX_LEN, dtype="long", value=1.0, truncating="post",
#                                        padding="post")
#     input_ids = torch.tensor(tokenized_sentence)
#     mask = [[float(m != 1) for m in val] for val in input_ids]
#     mask = torch.tensor(mask)
#     data_loader = DataLoader(TensorDataset(input_ids, mask), batch_size=1)
#     features = []
#     for batch in data_loader:
#         batch = tuple(t.to(device) for t in batch)
#         X, mask = batch
#         with torch.no_grad():
#             features = model_phoBert(X, mask)
#         features = features.detach().cpu().numpy()
#     predicts = model.predict(features)
#     predicts = np.argmax(predicts, axis=-1)
#     for indx, label in enumerate(intents):
#         if (predicts == indx):
#             intent = label
#     print(intent)

def processing_NER(sentence):

    pos = pos_tag(sentence)
    sentence = word_tokenize(sentence)
    X = [sent2features(pos)]
    import time
    start_time = time.time()
    with open('model_CRF_NER.pkl', 'rb') as fp:
        crf = pickle.load(fp)
    pred = crf.predict(X)
    pred = np.array(pred)
    pred = pred.flatten()
    print(pred)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    # Product perform by upper word
    sentence2string = ''
    words = []
    tag = []
    i = 0
    if(len(pred)>=2):
        for word, label in list(zip(sentence, pred)):
            if label[0] == 'B':
                sentence2string = ''
                sentence2string += (word)
                tag.append(label[2:])
            if label[0] == 'I' and word!=',':
                sentence2string += (' '+word)
            if label[0] == 'I' and word==',':
                sentence2string += (word)
            if label[0] == 'I' and (i+1 == len(pred)):
                words.append(sentence2string)
            if((i+1)>len(pred)):
                break
            if ((i + 1) < len(pred)):
                if label[0] == 'I' and pred[i + 1][0] == 'O':
                    words.append(sentence2string)
                if label[0] == 'I' and pred[i + 1][0] == 'B':
                    words.append(sentence2string)
                if label[0] == 'B' and pred[i + 1][0] == 'O':
                    words.append(sentence2string)
                if label[0] == 'B' and pred[i + 1][0] == 'B':
                    words.append(sentence2string)
            if ((i + 1) == len(pred)):
                if label[0] == 'B':
                    words.append(sentence2string)
            i = i + 1
        return words, tag
    if (len(pred) < 2):
        for word, label in list(zip(sentence, pred)):
            if label[0] == 'B':
                tag = []
                sentence2string = ''
                sentence2string += (word + ' ')
                tag.append(label[2:])
                words.append(sentence2string)
        return words, tag



def read_Basic_point(conn,tennganh,nam):
    print(tennganh)
    print("Read")
    cursor = conn.cursor()
    if (nam is None):
        cursor.execute(f"select DiemChuan,TenNganh,Nam from basic_point where TenNganh LIKE N'%{tennganh}%'")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Điểm chuẩn ngành {row[1]} năm {row[2]} là {row[0]}'
    if (tennganh is None):
        cursor.execute(f"select DiemChuan,TenNganh,Nam from basic_point where Nam LIKE {nam};")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Điểm chuẩn ngành {row[1]} năm {row[2]} là {row[0]}'
    if tennganh is not None and nam is not None:
        cursor.execute(f"select DiemChuan,TenNganh,Nam from basic_point where TenNganh LIKE N'%{tennganh}%' AND Nam LIKE {nam};")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Điểm chuẩn ngành {row[1]} năm {row[2]} là {row[0]}'
    return answer,nums
def read_Major_Infor(conn,tennganh):
    print("Read")
    cursor = conn.cursor()
    cursor.execute(f"select MaNganh,TenNganh from major_infor where TenNganh LIKE N'%{tennganh}%';")
    for row in cursor:
        answer = f'Mã ngành {row[1]} là {row[0]}'
    return answer
def read_Contact_point_Dept(conn,campus_name):
    print("Read")
    cursor = conn.cursor()
    cursor.execute(f"select Phong,SDT from contact_point_dept where Phong LIKE N'%{campus_name}%';")

    for row in cursor:

        answer = f'Số điện thoại {row[0]} là {row[1]}'
    return answer
def read_Contact_point_Teacher(conn,teacher_name,dept_name):
    print("Read")
    teacher_name_tmp = teacher_name.replace("thầy ", "")
    teacher_name_tmp = teacher_name_tmp.replace("cô ", "")
    teacher_name_tmp = teacher_name_tmp.replace("thây ", "")
    teacher_name_tmp = teacher_name_tmp.replace("thày ", "")
    teacher_name_tmp = teacher_name_tmp.replace("co ", "")
    cursor = conn.cursor()
    if(dept_name is None):
        cursor.execute(f"select SDT,Email from contact_point_teacher where HoVaTen LIKE N'%{teacher_name_tmp}%';")
        nums = 0
        for row in cursor:
            nums = nums+1
            answer = f'Số điện thoại của {teacher_name} là {row[0]} và email là {row[1]}'
    else:
        cursor.execute(f"select SDT,Email,BoPhan from contact_point_teacher where HoVaTen LIKE N'%{teacher_name_tmp}%' AND BoPhan LIKE N'%{dept_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Số điện thoại của {teacher_name} ở {row[2]} là {row[0]} và email là {row[1]}'
    return answer,nums

def read_Major_Fee(conn, major_name):
    intent = "học phí"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_name is None):
        answer = random_MajorName_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa,TenNganh from major_fee where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Thông tin về học phí ngành {row[1]} như sau: {row[0]}'
    return answer


def read_Credit_Fee(conn, major_name,datetime):
    intent = "học phí (tín chỉ)"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_name is None):
        answer = random_MajorName_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa,TenNganh from credit_fee where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Thông tin về học phí (tín chỉ) ngành {row[1]} như sau: {row[0]}'
    return answer

def read_Major_Description(conn, major_name):
    intent = "mô tả"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_name is None):
        answer = random_MajorName_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa,TenNganh from major_description where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Thông tin chung về ngành {row[1]} như sau: {row[0]}'
    return answer

def read_Major_training_time(conn, major_name):
    intent = "thời gian đào tạo"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_name is None):
        answer = random_MajorName_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa,TenNganh from major_training_time where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Thông tin về thời gian đào tạo ngành {row[1]}: {row[0]}'
    return answer

def read_Mode_of_Study(conn, major_mode):
    intent = "chế độ học tập"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_mode is None):
        answer = random_Program_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa,TenNganh from major_description where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Thông tin chung về chương trình {row[1]} như sau: {row[0]}'
    return answer


def read_Job_Opportunities(conn, major_name):
    intent = "cơ hội nghề nghiệp"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_name is None):
        answer = random_MajorName_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa,TenNganh from career_opportunity_major where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Cơ hội nghề nghiệp của ngành {row[1]}: {row[0]}'
    return answer


def read_Enroll_Docs(conn, doc_name):
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (doc_name is None):
        answer = f'Giấy tờ nhập học bao gồm: CV, học bạ cấp ba, Giấy chuyển nghĩa vụ quân sự, Sổ đoàn, Giấy chứng nhận kết quả thi, Giấy báo nhập học, ' \
                 f'Giấy báo trúng tuyển, Giấy chứng nhận tốt nghiệp THPT.'
    else:
        cursor.execute(
            f"select MoTa,Doc_Name from enroll_document where Doc_Name LIKE N'%{doc_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'Về {row[1]}: {row[0]}'
    return answer

def read_NoCredit_Major(conn, major_name):
    intent = "số tín chỉ"
    # major_name = major_name.replace("ngành ", "")
    # major_name = major_name.replace("nganh ", "")
    cursor = conn.cursor()
    answer = ""
    if (major_name is None):
        answer = random_MajorName_Answer(intent)
    else:
        cursor.execute(
            f"select MoTa from nocredit_major where TenNganh LIKE N'%{major_name}%';")
        nums = 0
        for row in cursor:
            nums = nums + 1
            answer = f'{row[0]}'
    return answer
def random_MajorName_Answer(intent):
    value = random.randint(0, 4)
    if value == 0:
        answer = "Bạn muốn hỏi "+intent+" của ngành nào ?"
    if value == 1:
        answer = "Bạn đang quan tâm tới "+intent+" của ngành nào vậy ?"
    if value == 2:
        answer = "Bạn muốn biết thông tin "+intent+" của ngành nào nhỉ ?"
    if value == 3:
        answer = "Bạn muốn hỏi thông tin "+intent+" của ngành nào ạ ?"
    if value == 4:
        answer = "Xin lỗi mình chưa hiểu ý bạn muốn hỏi thông tin "+intent+" của ngành nào vậy ạ ?"
    return answer

def random_Program_Answer(intent):
    value = random.randint(0, 4)
    if value == 0:
        answer = "Bạn muốn hỏi "+intent+" của chương trình nào ?"
    if value == 1:
        answer = "Bạn đang quan tâm tới "+intent+" của chương trình nào vậy ?"
    if value == 2:
        answer = "Bạn muốn biết thông tin "+intent+" của chương trình nào nhỉ ?"
    if value == 3:
        answer = "Bạn muốn hỏi thông tin "+intent+" của chương trình nào ạ ?"
    if value == 4:
        answer = "Xin lỗi mình chưa hiểu ý bạn muốn hỏi thông tin "+intent+" của chương trình nào vậy ạ ?"
    return answer

def random_KYear_Answer(intent):
    value = random.randint(0, 4)
    if value == 0:
        answer = "Bạn muốn hỏi "+intent+" của năm nào ?"
    if value == 1:
        answer = "Bạn đang quan tâm tới "+intent+" của năm nào vậy ?"
    if value == 2:
        answer = "Bạn muốn biết thông tin "+intent+" của năm nào nhỉ ?"
    if value == 3:
        answer = "Bạn muốn hỏi thông tin "+intent+" của năm nào ạ ?"
    if value == 4:
        answer = "Xin lỗi mình chưa hiểu ý bạn muốn hỏi thông tin "+intent+" của năm nào vậy ạ ?"
    return answer

def random_TeacherName_Answer(teacher_name):
    value = random.randint(0, 4)
    if value == 0:
        answer = "Bạn muốn hỏi "+teacher_name+" nào ?"
    if value == 1:
        answer = "Bạn muốn biết thông tin "+teacher_name+" nào nhỉ ?"
    if value == 2:
        answer = "Bạn cho mình biết tên đầy đủ của "+teacher_name+" nhé"
    if value == 3:
        answer = "Bạn muốn liên hệ với "+teacher_name+" nào nhỉ ?"
    if value == 4:
        answer = "Sorry, bạn có biết "+teacher_name+" bạn hỏi làm ở phòng/ban nào không ?"
    return answer

def random_Username_Answer():
    value = random.randint(0, 4)
    if value == 0:
        answer = "Tên bạn là gì ?"
    if value == 1:
        answer = "Bạn cho mình biết tên nhé!"
    if value == 2:
        answer = "Bạn để lại tên để tư vấn viên gọi nhé!"
    if value == 3:
        answer = "Tên của bạn là gì vậy ạ ?"
    if value == 4:
        answer = "Sorry, bạn cho mình biết tên bạn được không?, bên mình sẽ phản hồi lại bạn sau ạ!"
    return answer

def random_PhoneNumber_Answer():
    value = random.randint(0, 4)
    if value == 0:
        answer = "Vậy khoa có thể gọi cho em theo số điện thoại nào ?"
    if value == 1:
        answer = "Em dùng số điện thoại nào nhỉ ?"
    if value == 2:
        answer = "Số điện thoại của em là gì ?"
    if value == 3:
        answer = "Khoa sẽ liên lạc với em sau qua số điện thoại nào được nhỉ ?"
    if value == 4:
        answer = "Em để lại số điện thoại để tư vấn viên gọi nhé!"
    return answer

def random_Greeting_Answer():
    value = random.randint(0, 4)
    if value == 0:
        answer = "Chào bạn, bạn cần tư vấn gì ạ ?"
    if value == 1:
        answer = "Chào bạn"
    if value == 2:
        answer = "Hi bạn, bạn cần tôi giúp gì nào :)"
    if value == 3:
        answer = "Chào bạn, bạn cần hỗ trợ gì ạ!"
    if value == 4:
        answer = "Chào bạn, bạn muốn hỏi gì nào :)"
    return answer


def random_Bye_Answer():
    value = random.randint(0, 6)
    if value == 0:
        answer = "Tạm biệt bạn, chúc bạn sức khỏe!"
    if value == 1:
        answer = "Bye!"
    if value == 2:
        answer = "Cảm ơn bạn đã ghé thăm!"
    if value == 3:
        answer = "Tạm biệt bạn!"
    if value == 4:
        answer = "Bye bye!"
    if value == 5:
        answer = "Tạm biệt bạn, cảm ơn bạn đã ghé thăm"
    if value == 6:
        answer = "OK, chúc bạn sức khỏe nhé!"
    return answer

def random_Other_Answer():
    value = random.randint(0, 4)
    if value == 0:
        answer = "Rất tiếc. Ý của bạn là gì?"
    if value == 1:
        answer = "Xin lỗi. Bạn có thể nhắc lại ý định của bạn được không?"
    if value == 2:
        answer = "Xin lỗi, mình chưa trả lời được câu hỏi của bạn, vui lòng đợi TVV trong giây lát"
    if value == 3:
        answer = "Xin lỗi. Tôi không hiểu câu nói của bạn."
    if value == 4:
        answer = "Tôi không hiểu ý bạn lắm. Bạn có thể nhắc lại ý định của bạn được không?"
    return answer

# if __name__ == '__main__':
#     d ="ngành Ngôn Ngữ Hàn bao giờ cập nhật khung chương trình  ?"
#     d1 = [d]
#     a = processing_cnn_45(d1)
#     b  = processing_NER(d)
#     print(a)
#     print(b)
#
#     teacher_name  = None
#     campus_name  = None
#     dept_name = None
#     major_name = None
#     year = None
#     major_mode = None
#     doc_name = None
#     for i in range(len(b[1])):
#         if(b[1][i]=='Major_Name'):
#             major_name = b[0][i].strip()
#             major_name = major_name.replace("ngành ","")
#             major_name = major_name.replace("nganh ", "")
#
#         if (b[1][i] == 'Major_Mode'):
#             major_mode = b[0][i].strip()
#
#         if (b[1][i] == 'datetime'):
#             year = b[0][i].strip()
#             if(len(year)>4):
#                 year = year.split(' ',1)[1]
#         if (b[1][i] == 'Teacher_Name'):
#             teacher_name = b[0][i].strip()
#             print(teacher_name)
#         if (b[1][i] == 'Campus_Name' or b[1][i] == 'Dept_Name'):
#             campus_name = b[0][i].strip()
#             print(campus_name)
#             campus_name = campus_name.replace("cs ", "")
#             campus_name = campus_name.replace("cơ sở ", "")
#         if (b[1][i] == 'Dept_Name'):
#             dept_name = b[0][i].strip()
#         if (b[1][i] == 'Docs_Name'):
#             doc_name = b[0][i].strip()
#             doc_name = doc_name.replace("giay ", "")
#             doc_name = doc_name.replace("giấy ", "")
#             doc_name = doc_name.replace("Giấy ", "")
#             doc_name = doc_name.replace("Giây ", "")
#     # import pyodbc
#     # server = 'THANG-NGUYEN'
#     # database = 'ChatBotDB'
#     # username = 'sa'
#     # password = '123'
#     # driver = '{SQL Server Native Client 11.0}'  # Driver you need to connect to the database
#     # port = '1433'
#     # cnn = pyodbc.connect(
#     #     'DRIVER=' + driver + ';PORT=port;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username +
#     #     ';PWD=' + password)
#     import pymysql
#     cnn = pymysql.connect(
#         host="localhost",
#         user="root",
#         password="",
#         database="chatbotdb"
#     )
#     if a=="Basic_point":
#         print(read_Basic_point(cnn,major_name,year))
#     if a=="Major_infor":
#         print(read_Major_Infor(cnn,major_name))
#     if a=="contact_point" and campus_name is not None and teacher_name is None:
#         print(read_Contact_point_Dept(cnn,campus_name))
#     if a=="contact_point" and teacher_name is not None:
#         print(read_Contact_point_Teacher(cnn,teacher_name,dept_name))
#     if a=="Major_Fee":
#         ans = read_Major_Fee(cnn,major_name)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Ngành này chưa cập nhật học phí, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Credit_Fee":
#         ans = read_Credit_Fee(cnn,major_name,year)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Ngành này chưa cập nhật học phí, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Major_Description":
#         ans = read_Major_Description(cnn,major_name)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Ngành này chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Mode_of_Study":
#         ans = read_Mode_of_Study(cnn,major_mode)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Chương trình này chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Major_training_time":
#         ans = read_Major_training_time(cnn,major_name)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Job_Opportunities":
#         ans = read_Job_Opportunities(cnn,major_name)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Enroll_Docs" or a=="Enroll_Procedure":
#         ans = read_Enroll_Docs(cnn,doc_name)
#         if(ans!=""):
#             print(ans)
#         else:
#             print("Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé")
#
#     if a=="Enroll_Location" or a=="Enroll_Procedure":
#        print("Địa điểm nhập học là Nhà C, Làng sinh viên Hacinco, số 79 Ngụy Như Kon Tum, Nhân Chính, Thanh Xuân, Hà Nội.")
#
#     if a=="Curriculum":
#         ans = read_NoCredit_Major(cnn, major_name)
#         if (ans != ""):
#             print(ans)
#         else:
#             print("Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé")
