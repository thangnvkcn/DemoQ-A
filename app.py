from flask import Flask, render_template, request,json,redirect
import pickle
from pre_processing_data import *
from dictionary import *
app = Flask(__name__)
with open('model_cnn.pkl', 'rb') as fp:
    cnn_17 = pickle.load(fp)
with open('model_cnn_45_intents.pkl', 'rb') as fp:
    cnn_45 = pickle.load(fp)
import time
start_time = time.time()
# with open('model_lstm.pkl_', 'rb') as fp:
#     lstm_17 = pickle.load(fp)
# with open('model_lstm_45_intents.pkl_', 'rb') as fp:
#     lstm_45 = pickle.load(fp)
end_time = time.time()
print('total run-time: %f ms' % ((end_time - start_time) * 1000))
# with open('model_CRF_NER.pkl', 'rb') as fp:
#     crf = pickle.load(fp)
import pymysql
mydb = pymysql.connect(
            host="db4free.net",
            user="thang2020",
            password="chatbot2020",
            database="chatbotdb2020"
        )
# mydb = pymysql.connect(
#             host="localhost",
#             user="root",
#             password="",
#             database="chatbotdb"
#         )
# import pyodbc
# server = 'THANG-NGUYEN'
# database = 'ChatBotDB'
# username = 'sa'
# password = '123'
# driver = '{SQL Server Native Client 11.0}'  # Driver you need to connect to the database
# port = '1433'
# mydb = pyodbc.connect(
#             'DRIVER=' + driver + ';PORT=port;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username +
#             ';PWD=' + password)
# app.config['MYSQL_HOST'] = ''
# app.config['MYSQL_USER'] = ''
# app.config['MYSQL_PASSWORD'] =''
# app.config['MYSQL_DB'] = ''

@app.route('/')
def a():
    return render_template("session.html")


# @app.route('/process', methods=['GET','POST'])
# def process():
#     if request.method == 'POST':
#         choice = request.form['model_option']
#         rsawtext = request.form['rawtext']
#         text = rawtext
#
#
#     return render_template("index.html", results_17=results_17,results_45=results_45,text = text )

@app.route('/process', methods=['GET','POST'])
def vote():
    if request.method == 'POST':
        teacher_name = None
        campus_name = None
        dept_name = None
        major_name = None
        year = None
        major_mode = None
        doc_name = None
        rawtext=request.get_json()['text']
        rawtext = find_in_dictionary(rawtext)
        test_sentence = [rawtext]
        print(test_sentence)
        results_17 = processing_cnn_17(test_sentence,cnn_17)
        results_45 = processing_cnn_45(test_sentence,cnn_45)
        print(results_45)
        # choice = request.get_json()['model']
        # print(choice)
        # print(rawtext)
        # if choice == 'cnn':

        #     results_17 = processing_cnn_17(test_sentence,cnn_17)

        # elif choice == 'lstm':
        #     test_sentence = [rawtext]
        #     results_17 = processing_lstm_17(test_sentence)
        #     results_45 = processing_lstm_45(test_sentence)
        results_NER = processing_NER(rawtext)
        print(results_NER)
        for i in range(len(results_NER[1])):
            if (results_NER[1][i] == 'Major_Name'):
                major_name = results_NER[0][i].strip()
                major_name = major_name.replace("ngành ", "")
                major_name = major_name.replace("nganh ", "")

            if (results_NER[1][i] == 'datetime'):
                year = results_NER[0][i].strip()
                if (len(year) > 4):
                    year = year.split(' ', 1)[1]
            if (results_NER[1][i] == 'Teacher_Name'):
                teacher_name = results_NER[0][i].strip()
            if (results_NER[1][i] == 'Campus_Name' or results_NER[1][i] == 'Dept_Name'):
                campus_name = results_NER[0][i].strip()
                campus_name = campus_name.replace("cs ", "")
                campus_name = campus_name.replace("cơ sở ", "")
            if (results_NER[1][i] == 'Dept_Name'):
                dept_name = results_NER[0][i].strip()
            if (results_NER[1][i] == 'Major_Mode'):
                major_mode = results_NER[1][i].strip()
            if (results_NER[1][i] == 'Docs_Name'):
                doc_name = results_NER[0][i].strip()
                doc_name = doc_name.replace("giay ", "")
                doc_name = doc_name.replace("giấy ", "")
                doc_name = doc_name.replace("Giấy ", "")
                doc_name = doc_name.replace("Giây ", "")


        if results_45=="Basic_point":
            intent = "điểm chuẩn"
            if year is None and major_name is None:
                answer = random_MajorName_Answer(intent)
            else:
                _,nums = read_Basic_point(mydb,major_name,year)
                if(0<nums<=1):
                    answer = read_Basic_point(mydb,major_name,year)[0]
                    teacher_name = None
                    campus_name = None
                    dept_name = None
                    major_name = None
                    year = None
                    major_mode = None
                    doc_name = None
                elif nums>=2 and year is None:
                    answer = random_KYear_Answer(intent)
                elif nums>=2 and major_name is None:
                    answer = random_MajorName_Answer(intent)
                else:
                    answer = "Ngành này chưa cập nhật điểm chuẩn, bạn vui lòng liên hệ sau nhé"

        if results_45=="Major_Code":
            answer = read_Major_Infor(mydb, major_name)
        if results_45=="contact_point" and campus_name is not None and teacher_name is None:
            answer = read_Contact_point_Dept(mydb, campus_name)
        if results_45=="contact_point" and teacher_name is not None:
            _,nums = read_Contact_point_Teacher(mydb,teacher_name,dept_name)
            if(0<nums<=1):
                answer = read_Contact_point_Teacher(mydb, teacher_name, dept_name)[0]
            else:
                answer = random_TeacherName_Answer(teacher_name)

        if results_45 == "Major_Fee":
            ans = read_Major_Fee(mydb, major_name)
            if (ans != ""):
                answer = ans
            else:
                answer = "Ngành này chưa cập nhật học phí, bạn vui lòng liên hệ sau nhé"

        if results_45 == "Credit_Fee":
            ans = read_Credit_Fee(mydb, major_name, year)
            if (ans != ""):
                answer = ans
            else:
                answer = "Ngành này chưa cập nhật học phí (tín chỉ), bạn vui lòng liên hệ sau nhé"

        if results_45 == "Major_Description":
            ans = read_Major_Description(mydb, major_name)
            if (ans != ""):
                answer = ans
            else:
                answer = "Ngành này chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé"

        if results_45 == "Mode_of_Study":
            ans = read_Mode_of_Study(mydb, major_mode)
            if (ans != ""):
                answer = ans
            else:
                answer = "Chương trình này chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé"

        if results_45 == "Major_training_time":
            ans = read_Major_training_time(mydb, major_name)
            if (ans != ""):
                answer = ans
            else:
                answer = "Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé!"

        if results_45 == "Job_Opportunities":
            ans = read_Job_Opportunities(mydb, major_name)
            if (ans != ""):
                answer = ans
            else:
                answer = "Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé"

        if results_45 == "Enroll_Docs" or results_45 == "Enroll_Procedure":
            ans = read_Enroll_Docs(mydb, doc_name)
            if (ans != ""):
                answer = ans
            else:
                answer = "Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé"

        if results_45 == "Enroll_Location" or results_45 == "Enroll_Procedure":
            answer = "Địa điểm nhập học là Nhà C, Làng sinh viên Hacinco, số 79 Ngụy Như Kon Tum, Nhân Chính, Thanh Xuân, Hà Nội."

        if results_45 == "Curriculum":
            ans = read_NoCredit_Major(mydb, major_name)
            if (ans != ""):
                answer = ans
            else:
                answer = "Hiện tại chưa cập nhật thông tin, bạn vui lòng liên hệ sau nhé"
        if results_45 == "Greetings ":
            answer = random_Greeting_Answer()

        if results_45 == "Bye":
            answer = random_Bye_Answer()

        if results_45 == "Other":
            answer = random_Other_Answer()
        if results_45!="Basic_point" and results_45!="contact_point" and results_45!="Major_Code" and results_45!="Major_Fee" and results_45!="Credit_Fee" and results_45!="Major_Description" and results_45!="Mode_of_Study" and results_45!="Major_training_time" and results_45!="Job_Opportunities" and results_45!="Enroll_Docs" and results_45!="Enroll_Location" and results_45!="Enroll_Procedure" and results_45!="Greetings " and results_45!="Bye":
            answer = random_Other_Answer()

        # if results_17 == "Basic_point":
        #     answer = read_Basic_point(mydb,major_name,year)
        # if results_17 == "Major_infor":
        #     answer = read_Major_Infor(mydb, major_name)
        # if results_17 == "contact_point" and campus_name is not None and teacher_name is None:
        #     answer = read_Contact_point_Dept(mydb, campus_name)
        # if results_17 == "contact_point" and teacher_name is not None:
        #     answer = read_Contact_point_Teacher(mydb, teacher_name, dept_name)



        st1 = results_NER[0]
        st2 = results_NER[1]
        text=""
        p='<p align="center" class="tag_{}">{} <br> {}</p>'
        ner = []
        for i in range(len(st2)):
            label=st2[i]
            word=st1[i]
            if label=="O":
                text += p.format(label, word, "")
            else:
                text+=p.format(label,word,label)
                ner.append(word+" - "+label)
        text={'text':text,'results_17':results_17,'results_45':results_45,'answer':answer,'results_NER':ner}
    return json.dumps({'success': True, 'text_tagged': text}), 200, {'Content-Type': 'application/json; charset=UTF-8'}
if __name__ == '__main__':
    app.run(host='127.0.0.1',threaded=False,port=8080)