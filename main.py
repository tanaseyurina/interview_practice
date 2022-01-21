import numpy as np
import cv2

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import font

import Ai_program  #自動分析プログラムの読み込み
import csv #csvファイルの読み込み AIデータ保存

# グラフ
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# ファイル参照
import os
# 音読
import pyttsx3
# 録音
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import queue
import sys
import wave #WAVファイルの読み書き
# スクショ
import tkcap #スクショ保存

import threading #スレッドによる並列処理を管理
import time #時間を扱う関数
import datetime #日付や時刻を操作するためのクラスを提供
from PIL import ImageTk, Image #画像ファイルの読み込み

camera=cv2.VideoCapture(0)

tt=0 #Ai_Data_aggregate():用
txt_row=0 #def question_play():用
flg = True #cam():用

def cam_start():
    thread = threading.Thread(target=cam)
    thread.start()

def cam():
    global flg
    cap = cv2.VideoCapture(0)
    fps = 30

    # 録画する動画のフレームサイズ（webカメラと同じにする）
    size = (640, 480)
    now = datetime.datetime.now()
    filename= './video/' + now.strftime('%Y%m%d%H%M') + '.avi' 
    # 出力する動画ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(filename, fourcc, fps, size)

    while (cap.isOpened()):
        if flg == False:
            flg =True
            break
        else:
            ret, frame = cap.read()
            print("1")
            print("2")
            
            # 画面表示
            #cv2.imshow('frame', frame)
            #disp_image()

            # 書き込み
            video.write(frame)       

    # 終了処理
    cap.release()
    video.release()
    cv2.destroyAllWindows()

def cam_stop():
    global flg
    flg = False
    
    print("stop")
    
def change_a(window):
    window.tkraise()


def Ai_start():
    thread = threading.Thread(target=Ai_Data_aggregate)
    thread.start()

def Ai_Data_aggregate():
    global flg
    
    start=time.time()

    num=0
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    g=0
    h=0
    i=0
    with open('data.csv', 'a',newline='') as z:
        while 1:
            if flg == False:
                flg = True
                break
            else:
                tt=time.time()-start
                if tt>=1:

                        writer = csv.writer(z)
                        writer.writerow((num,a,b,c,d,e,f,g,h,i)) 
                        start=time.time()
                        num+=1
                        a=0
                        b=0
                        c=0
                        d=0
                        e=0
                        f=0
                        g=0
                        h=0
                        i=0

                else:
                    a,b,c,d,e,f,g,h,i = Ai_program.Ai(a,b,c,d,e,f,g,h,i)
                    print(Ai_program.Ai(a,b,c,d,e,f,g,h,i))

def Ai_stop():
    global flg
    flg = False

#Aiのデータをリセット##################################################
def Ai_Data_reset():
    with open('data.csv', 'w',newline='') as z:
        writer = csv.writer(z)
        header = ['tt', 'a', 'b', 'c', 'd','e','f','g','h','i']
        writer.writerow(header)
#######################################################################

#棒グラフ設定##########################################################
def bar_graph_set(x, y):
    #　Figureインスタンスを生成する
    fig = plt.Figure(figsize=(10,7))
 
    # 目盛を内側にする
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
 
    # Axes(軸)を作り、グラフの上下左右に目盛線を付ける
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.set_xticklabels(["顔_正面","視線_正面","うなずき","笑顔","顔_右向き","顔_左向き","視線_右向き","視線_左向き","視線_下向き"],rotation=30,fontname="MS Gothic")
    
    # 軸のラベルを設定する
    ax.set_ylabel('count')

    ax.tick_params()
    ax.set_xticks( np.arange(0, 9, step=1) )
    ax.bar(x, y,width=0.3,color=['gray','gray','gray','gray','red','red','blue','blue','blue'])

    return fig
#######################################################################

#棒グラフ##############################################################
def bar_graph():

    df = pd.read_csv('data.csv',index_col=0)
    a=df['a'].sum() # 顔_右向き
    b=df['b'].sum() # 顔_左向き
    c=df['c'].sum() # 顔_正面
    d=df['d'].sum() # 視線_右向き
    e=df['e'].sum() # 視線_左向き
    f=df['f'].sum() # 視線_正面
    g=df['g'].sum() # 視線_下向き
    h=df['h'].sum() # うなずき
    i=df['i'].sum() # 笑顔
    
    x = np.arange(0, 9)
    y = c,f,h,i,a,b,d,e,g

    with open('data.csv') as z:
        print(z.read())

    fig = bar_graph_set(x, y)
    canvas = FigureCanvasTkAgg(fig, frame_bar_graph)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0 )
#######################################################################

#折れ線グラフ設定######################################################
def line_graph_set_cfhi(x, yc,yf,yh,yi):
    df = pd.read_csv('data.csv')
    # Figureインスタンスを生成する
    fig = plt.Figure(figsize=(18,7))

    # 目盛を内側にする
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Axes（軸）を作り、グラフの上下左右に目盛線を付ける
    axc = fig.add_subplot(411)
    axc.yaxis.set_ticks_position('both')
    axc.xaxis.set_ticks_position('both')
    # 軸の目盛設定
    axc.tick_params()
    axc.set_xticks( np.arange(0,len(df),1) )
    axc.set_yticks( np.arange(0,len(df),1) )
    axc.set_xticklabels([])    
    # 軸のラベルを設定する
    axc.set_ylabel("顔/正面",rotation=360,labelpad=30,fontname="MS Gothic")
    # データを線でつなぐ
    axc.plot(x, yc,color="gray",linewidth=2)

    axf = fig.add_subplot(412)
    axf.yaxis.set_ticks_position('both')
    axf.xaxis.set_ticks_position('both')
    axf.tick_params()
    axf.set_xticks( np.arange(0,len(df),1) )
    axf.set_yticks( np.arange(0,len(df),1) )
    axf.set_xticklabels([])
    axf.set_ylabel("視線/正面",rotation=360,labelpad=30,fontname="MS Gothic")
    axf.plot(x, yf,color="gray",linewidth=2)
    
    axh = fig.add_subplot(413)
    axh.yaxis.set_ticks_position('both')
    axh.xaxis.set_ticks_position('both')
    axh.tick_params()
    axh.set_xticks( np.arange(0,len(df),1) )
    axh.set_yticks( np.arange(0,len(df),1) )
    axh.set_xticklabels([])
    axh.set_ylabel("うなずき",rotation=360,labelpad=30,fontname="MS Gothic")
    axh.plot(x, yh,color="gray",linewidth=2)
    
    axi = fig.add_subplot(414)
    axi.yaxis.set_ticks_position('both')
    axi.xaxis.set_ticks_position('both')
    axi.tick_params()
    axi.set_xticks( np.arange(0,len(df),1) )
    axi.set_yticks( np.arange(0,len(df),1) )
    axi.set_xticklabels(df['tt'],fontname="MS Gothic")
    axi.set_xlabel('time')
    axi.set_ylabel("笑顔",rotation=360,labelpad=30,fontname="MS Gothic")
    axi.plot(x, yi,color="gray",linewidth=2)

    fig.align_ylabels()

    return fig
#######################################################################

#折れ線グラフ##########################################################
def line_graph_cfhi(): #折れ線グラフ
    df = pd.read_csv('data.csv')
    x=np.arange(0,len(df),1)
    yc=df['c']
    yf=df['f']
    yh=df['h']
    yi=df['i']

    fig = line_graph_set_cfhi(x, yc,yf,yh,yi)
    canvas = FigureCanvasTkAgg(fig, frame_line_graph_true)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0)
#######################################################################

#折れ線グラフ設定######################################################
def line_graph_set_ab(x,ya,yb):
    df = pd.read_csv('data.csv')
    # Figureインスタンスを生成する
    fig = plt.Figure(figsize=(18,7))
    
    # 目盛を内側にする
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Axes（軸）を作り、グラフの上下左右に目盛線を付ける
    axa = fig.add_subplot(211)
    axa.yaxis.set_ticks_position('both')
    axa.xaxis.set_ticks_position('both')
    # 軸の目盛設定
    axa.tick_params()
    axa.set_xticks( np.arange(0,len(df),1) )
    axa.set_yticks( np.arange(0,len(df),1) )
    axa.set_xticklabels([])
    # 軸のラベルを設定する
    axa.set_ylabel('顔/右向き',rotation=360,labelpad=30,fontname="MS Gothic")
    # データを線でつなぐ
    axa.plot(x, ya,color="red",linewidth=2)

    axb = fig.add_subplot(212)
    axb.yaxis.set_ticks_position('both')
    axb.xaxis.set_ticks_position('both')
    axb.tick_params()
    axb.set_xticks( np.arange(0,len(df),1) )
    axb.set_yticks( np.arange(0,len(df),1) )
    axb.set_xticklabels(df['tt'],fontname="MS Gothic")
    axb.set_xlabel('time')
    axb.set_ylabel('顔/左向き',rotation=360,labelpad=30,fontname="MS Gothic")
    axb.plot(x, yb,color="red",linewidth=2)
    
    fig.align_ylabels() 

    return fig
#######################################################################

#折れ線グラフ##########################################################
def line_graph_ab(): #折れ線グラフ
    df = pd.read_csv('data.csv')
    x=np.arange(0,len(df),1)
    ya=df['a']
    yb=df['b']
    fig = line_graph_set_ab(x, ya,yb)
    canvas = FigureCanvasTkAgg(fig, frame_line_graph_face)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0)
#######################################################################

#折れ線グラフ設定######################################################
def line_graph_set_deg(x, yd,ye,yg):
    df = pd.read_csv('data.csv')
    # Figureインスタンスを生成する
    fig = plt.Figure(figsize=(18,7))

    # 目盛を内側にする
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Axes（軸）を作り、グラフの上下左右に目盛線を付ける
    axd = fig.add_subplot(311)
    axd.yaxis.set_ticks_position('both')
    axd.xaxis.set_ticks_position('both')
    axd.tick_params()
    axd.set_xticks( np.arange(0,len(df),1) )
    axd.set_yticks( np.arange(0,len(df),1) )
    axd.set_xticklabels([])
    # 軸のラベルを設定する
    axd.set_ylabel('視線/右向き',rotation=360,labelpad=30,fontname="MS Gothic")
    # データを線でつなぐ
    axd.plot(x, yd,color="blue",linewidth=2)

    axe = fig.add_subplot(312)
    axe.yaxis.set_ticks_position('both')
    axe.xaxis.set_ticks_position('both')
    axe.tick_params()
    axe.set_xticks( np.arange(0,len(df),1) )
    axe.set_yticks( np.arange(0,len(df),1) )
    axe.set_xticklabels([])
    axe.set_ylabel('視線/左向き',rotation=360,labelpad=30,fontname="MS Gothic")
    axe.plot(x, ye,color="blue",linewidth=2)

    axg = fig.add_subplot(313)
    axg.yaxis.set_ticks_position('both')
    axg.xaxis.set_ticks_position('both')
    axg.tick_params()
    axg.set_xticks( np.arange(0,len(df),1) )
    axg.set_yticks( np.arange(0,len(df),1) )
    axg.set_xticklabels(df['tt'],fontname="MS Gothic")
    axg.set_xlabel('time')
    axg.set_ylabel('視線/下向き',rotation=360,labelpad=30,fontname="MS Gothic")
    axg.plot(x, yg,color="blue",linewidth=2)
    
    fig.align_ylabels()

    return fig
#######################################################################

#折れ線グラフ##########################################################
def line_graph_deg(): #折れ線グラフ
    df = pd.read_csv('data.csv')
    x=np.arange(0,len(df),1)
    yd=df['d']
    ye=df['e']
    yg=df['g']

    fig = line_graph_set_deg(x, yd,ye,yg)
    canvas = FigureCanvasTkAgg(fig, frame_line_graph_eye)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0)
#######################################################################

# ファイル指定の関数###################################################
def filedialog_check():
    fTyp = [("", "*")]
    iFile = os.path.abspath(os.path.dirname(__file__))
    iFilePath = filedialog.askopenfilename(filetype = fTyp, initialdir = iFile)

    entry.set(iFilePath)
#######################################################################

# ファイル指定の関数###################################################
def filedialog_check2():
    fTyp = [("", "*")]
    iFile = os.path.abspath(os.path.dirname(__file__))
    iFilePath = filedialog.askopenfilename(filetype = fTyp, initialdir = iFile)

    Read_aloud_entry.set(iFilePath)
#######################################################################

#音読##################################################################
def Read_aloud(): #Read_aloud 音読
    text = ""

    filePath = Read_aloud_entry.get()
    if filePath:
        text += "ファイルパス：" + filePath

    if text:
        engine = pyttsx3.init()
        f = open(filePath, 'r', encoding='UTF-8')
        data = f.read()
        f.close()

        # rateは、1分あたりの単語数で表した発話速度(基本200)
        rate = engine.getProperty("rate")
        engine.setProperty("rate",150)

        # ボリュームは、0.0~1.0の間で設定
        volume = engine.getProperty('volume')
        engine.setProperty('volume',1.0)

        #参照した言葉の出力
        engine.say(data)
        engine.runAndWait()
    else:
        messagebox.showerror("error", "パスの指定がありません")
#######################################################################

# 面接官の関数#########################################################
def interviewer_window():
    global img
    image = Image.open("2.png")
    
    root2 = tk.Toplevel()
    root2.geometry(str(image.width)+"x"+str(image.height))
    canvas = tk.Canvas(root2, width=image.width, height=image.height)
    canvas.pack()

    img = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0,anchor=tk.NW, image=img)
    print("end")
#######################################################################

#質問再生##############################################################
def question_play(): #question_play 質問再生
    global txt_row
    global data

    txt_data=data[txt_row%len(data)]

    engine = pyttsx3.init()

    rate = engine.getProperty("rate")
    engine.setProperty("rate",150)

    # ボリュームは、0.0~1.0の間で設定します
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)

    #参照した言葉の出力
    engine.say(txt_data)

    engine.runAndWait()

    txt_row += 1
#######################################################################

def question():
    interviewer_window()
    question_play()


#毎回読み込ませない方法################################################
def read_txt(): #質問再生用　読み込み
    text = ""

    filePath = entry.get()
    if filePath:
        text += "ファイルパス：" + filePath

    if text:
        with open(filePath, 'r', encoding='UTF-8') as f:
            data = f.read()
            data=data.split("\n")
            for data in f:
                print(data)
            
        return data

    else:
        messagebox.showerror("error", "パスの指定がありません")
#######################################################################

#スクショ保存##########################################################
def screenshot_save():
    now = datetime.datetime.now()
    time = './picture/' + now.strftime('%Y%m%d%H%M') + '.png'
    cap= tkcap.CAP(root)
    cap.capture(time)  
#######################################################################

#文字起こし用　録音####################################################

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

q = queue.Queue()

flg = True

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def sentence_output_rec_start():
    thread = threading.Thread(target=sentence_output_rec)
    thread.start()

def sentence_output_rec():
    global flg
    channels=1
    device=None
    now = datetime.datetime.now()
    filename= './record/' + now.strftime('%Y%m%d%H%M') + '.wav'
    samplerate=None
    subtype=None

    if samplerate is None:
        device_info = sd.query_devices(device, 'input')
        samplerate = int(device_info['default_samplerate'])

    with sf.SoundFile(filename, mode='x', samplerate=samplerate,channels=channels, subtype=subtype) as file:
        with sd.InputStream(samplerate=samplerate, device=device,channels=channels, callback=callback):
            while True:              
                if flg == False:
                    flg =True
                    break
                else:
                    file.write(q.get())

def sentence_output_rec_stop():
    global flg
    flg = False
#######################################################################

#文字起こし用　文字起こし##############################################
def sentence_output():
    text = ""

    filePath = sentence_output_entry.get()
    if filePath:
        text += "ファイルパス：" + filePath

    if text:
        r = sr.Recognizer()
        with sr.AudioFile(filePath) as source:
            audio = r.record(source)

        aa=tk.Label(frame_sentence_output,text=r.recognize_google(audio, language='ja'))
        aa.grid(row=2,column=0,columnspan=3,sticky=tk.W)

    else:
        messagebox.showerror("error", "パスの指定がありません")

#######################################################################

#フィードバック機能####################################################
def feedback():
    df = pd.read_csv('data.csv',index_col=0)
    a=df['a'].sum() # 顔_右向き
    b=df['b'].sum() # 顔_左向き
    c=df['c'].sum() # 顔_正面
    d=df['d'].sum() # 視線_右向き
    e=df['e'].sum() # 視線_左向き
    f=df['f'].sum() # 視線_正面
    g=df['g'].sum() # 視線_下向き
    #h=df['h'].sum() # うなずき
    i=df['i'].sum() # 笑顔

    sum_face=a+b+c
    sum_eye=d+e+f+g
    p_face=(c*100)/sum_face
    p_eye=(f*100)/sum_eye
    p_eye_d=(f*100)/sum_eye

    p_smile=(i*100)/sum_face

    if sum_face==c:
        feedback_face="<顔の向き>しっかり正面に向けています。練習していない質問や自信が無くなった時などに、顔が下に向きやすくなるので、気を付けましょう。"
    elif 100>=p_face>90:
        feedback_face="<顔の向き>よそを向いたタイミングを折れ線グラフや動画で振り返りましょう。その時のあなたはどういう心境・状況でしたか？"
    elif 90>=p_face:
        feedback_face="<顔の向き>まだまだこれからです！面接官に思いを届けるイメージで顔を正面に向けましょう。自信がつくまで繰り返し練習です！"
    
    if 100>p_eye>90:
        feedback_eye="<視線>しっかりカメラを捉えられています。面接官に思いを伝えるイメージで練習してみましょう。"
    elif 90>=p_eye>80:
        feedback_eye="<視線>視線がずれたタイミングを動画や折れ線グラフで振り返りましょう。予想外の質問などがありましたか？回答する時には視線をカメラに向けましょう。"
    elif 80>=p_eye:
        feedback_eye="<視線>折れ線グラフと動画を振り返ってください。自分自身を見てあなたはどういう印象をもちましたか？自信がつくまで振り返り練習あるのみです！！"
    elif p_eye_d>=50:
        feedback_eye="<視線>下を向きすぎです。PC画面を見ていてもカメラ越しの相手には下を向いているように見えます。自分自身で動画を確認し相手にどう見られているか確認しましょう。"

    if 5<=p_smile:
        feedback_smile="笑顔が程よくあり良いと思います。動画を振り返り自然な流れの笑顔か自分自身で見返し、好印象が持てるかチェックしましょう。"
    else:
        feedback_smile="笑顔が全くないですね。最初の挨拶や笑顔を作れるタイミングがあれば笑顔になりましょう。円滑なコミュニケーションをとるために笑顔は大切です。" #https://employment.en-japan.com/tenshoku-daijiten/42123/

    feedback_comment="お疲れさまでした！グラフや録画・録音を利用し、あなたの行動が面接官にどういう印象を与えているか振り返りながら練習していきましょう！！"
    
    lbl_Read_aloud1 = ttk.Label(frame_feedback, text=feedback_face)
    lbl_Read_aloud1.grid(row=1,column=0,sticky=tk.W)
    lbl_Read_aloud11 = ttk.Label(frame_feedback, text=feedback_eye)
    lbl_Read_aloud11.grid(row=2,column=0,sticky=tk.W)
    lbl_Read_aloud2 = ttk.Label(frame_feedback, text=feedback_smile)
    lbl_Read_aloud2.grid(row=3,column=0)
    lbl_Read_aloud22 = ttk.Label(frame_feedback, text=feedback_comment)
    lbl_Read_aloud22.grid(row=4,column=0)
#######################################################################

# ファイル指定の関数###################################################
def filedialog_check3():
    fTyp = [("", "*")]
    iFile = os.path.abspath(os.path.dirname(__file__))
    iFilePath = filedialog.askopenfilename(filetype = fTyp, initialdir = iFile)
    sentence_output_entry.set(iFilePath)
#######################################################################

#Tkinter初期設定#######################################################
root = tk.Tk()
root.title("window")
root.geometry("1900x1060")
#######################################################################

#ボタン枠　メイン######################################################
frame_btn_main = Frame(root, bd=6, relief=GROOVE) #ボタンの枠
frame_btn_main.grid(row=0,column=0,sticky=tk.W+tk.E+tk.N+tk.S) #ボタンの枠配置設定

#title
lbl_practice = ttk.Label(frame_btn_main,text="<  面接練習  >")
lbl_practice.grid(row=0,column=0,columnspan=2)

btn_start = ttk.Button(frame_btn_main,text="開始",command=Ai_start)
#btn_start.grid(row=1,sticky=tk.W,padx=100,columnspan=3)
btn_start.grid(row=1,column=0,sticky=tk.W)
#btn_start.grid_anchor(tk.CENTER)
btn_stop = ttk.Button(frame_btn_main,text="停止",command=Ai_stop)
#btn_stop.grid(row=1,sticky=tk.W,padx=150,columnspan=3)
btn_stop.grid(row=1,column=1)
#btn_stop.grid_anchor(tk.CENTER)
btn_reset = ttk.Button(frame_btn_main,text="データリセット",command=Ai_Data_reset)
#btn_reset.grid(row=1,sticky=tk.W,padx=200,columnspan=3)
btn_reset.grid(row=1,column=2)
#btn_reset.grid_anchor(tk.CENTER)

#title 予想質問ラベル
lbl_question_play = ttk.Label(frame_btn_main, text="<  質疑対応練習  >")
lbl_question_play.grid(row=2,column=0,columnspan=2)

# 「ファイル参照」エントリーの作成　入力枠
entry = StringVar()
IFileEntry = ttk.Entry(frame_btn_main, textvariable=entry, width=20)
IFileEntry.grid(row=3,column=2,columnspan=2,sticky=tk.W)
entry.set("質問練習用.txt") #z.txtは他に変えてもいい

# 「ファイル参照」ボタンの作成　
btn_filedialog_check = ttk.Button(frame_btn_main, text="参照", command=filedialog_check)
btn_filedialog_check.grid(row=3,column=1)

data=read_txt() #質問スタートボタンより前にする

btn_question_play = ttk.Button(frame_btn_main, text="質問スタート", command=question)#lambda:[interviewer_window(),question_play()]
btn_question_play.grid(row=3,column=0)

#title 録音
lbl_sentence_output_rec = ttk.Label(frame_btn_main, text="<  録音  >")
lbl_sentence_output_rec.grid(row=4,column=0,columnspan=2)

# 録音 ボタンの作成　
btn_sentence_output_rec = ttk.Button(frame_btn_main, text="録音", command=sentence_output_rec_start) #▶
btn_sentence_output_rec.grid(row=5,column=0,sticky=tk.W)
btn_sentence_output_stop = ttk.Button(frame_btn_main, text="停止", command=sentence_output_rec_stop) #■
btn_sentence_output_stop.grid(row=5,column=1,sticky=tk.W)

#title 録画
lbl_enpty = ttk.Label(frame_btn_main, text="")
lbl_enpty.grid(row=4,column=1,columnspan=1)
lbl_video_rec = ttk.Label(frame_btn_main, text="<  録画  >")
lbl_video_rec.grid(row=4,column=4,columnspan=2)
# 録画 ボタンの作成　
btn_video_rec = ttk.Button(frame_btn_main, text="録音", command=cam_start)
btn_video_rec.grid(row=5,column=4,sticky=tk.W)
btn_video_stop = ttk.Button(frame_btn_main, text="停止", command=cam_stop)
btn_video_stop.grid(row=5,column=5,sticky=tk.W)
#######################################################################

#アドバイス文##########################################################
frame_feedback = Frame(root, bd=6, relief=GROOVE) #ボタンの枠
frame_feedback.grid(row=0, column=1,sticky=tk.W+tk.E+tk.N+tk.S) #ボタンの枠配置設定
lbl_Read_aloud = ttk.Button(frame_feedback, text="<  アドバイス  >", command=feedback)
lbl_Read_aloud.grid(row=0,column=0, ipadx=180)
#######################################################################

#ボタン枠　サブ########################################################
frame_btn_sub = Frame(root, bd=6, relief=GROOVE) #ボタンの枠
frame_btn_sub.grid(row=0, column=2,sticky=tk.W+tk.E+tk.N+tk.S, ipadx=130) #ボタンの枠配置設定

# うなずき練習（音読）ラベル
lbl_Read_aloud = ttk.Label(frame_btn_sub, text="<  うなずき練習  >")
lbl_Read_aloud.grid(row=0,column=0,columnspan=3)

# 「ファイル参照」エントリーの作成　入力枠
Read_aloud_entry = StringVar()
Read_aloud_IFileEntry = ttk.Entry(frame_btn_sub, textvariable=Read_aloud_entry, width=20)
Read_aloud_IFileEntry.grid(row=1,column=2,sticky=tk.W)
Read_aloud_entry.set("うなずき練習用.txt")

# 「ファイル参照」ボタンの作成　
btn_filedialog_check2 = ttk.Button(frame_btn_sub, text="参照", command=filedialog_check2)
btn_filedialog_check2.grid(row=1,column=1)

#音読スタートボタン
btn_Read_aloud = ttk.Button(frame_btn_sub, text="音読スタート", command=Read_aloud)
btn_Read_aloud.grid(row=1,column=0)

#######################################################################

#切り替え先frame#######################################################
  # 棒グラフ #
frame_bar_graph = tk.Frame(root,height=800)
frame_bar_graph.grid(row=2, column=0,columnspan=4, sticky=tk.W+tk.E+tk.N+tk.S)

  # 折れ線グラフ 正 #
frame_line_graph_true = tk.Frame(root,bg="light grey")
frame_line_graph_true.grid(row=2, column=0, columnspan=4, sticky="nsew")
frame_line_graph_true.grid_propagate(0) #height=500を有効にするもの

  # 折れ線グラフ 顔 #
frame_line_graph_face = tk.Frame(root,bg="LightPink",height=800)
frame_line_graph_face.grid(row=2, column=0, columnspan=4, sticky="nsew")

  # 折れ線グラフ 目 #
frame_line_graph_eye = tk.Frame(root,bg="light steel blue",height=800)
frame_line_graph_eye.grid(row=2, column=0, columnspan=4, sticky="nsew")

  # 文字起こし #
frame_sentence_output = tk.Frame(root,bg="dark gray",height=800)
frame_sentence_output.grid(row=2, column=0, columnspan=4, sticky="nsew")

  # 面接官 #
frame_interviewer = tk.Frame(root,bg="dark gray",height=800)
frame_interviewer.grid(row=2, column=0, columnspan=4, sticky="nsew")

#  # スクショ #
#frame_screenshot = tk.Frame(root,bg="dark gray")
#frame_screenshot.grid(row=2, column=0, columnspan=4, sticky="nsew")

#######################################################################

#画面切り替え##########################################################

entry_1 = ttk.Entry(frame_bar_graph)
entry_2 = ttk.Entry(frame_line_graph_true)
entry_3 = ttk.Entry(frame_line_graph_face)
entry_4 = ttk.Entry(frame_line_graph_eye)
entry_5 = ttk.Entry(frame_sentence_output)
entry_6 = ttk.Entry(frame_interviewer)

frame_btn_switch = Frame(root, bd=6, relief=SUNKEN)
frame_btn_switch.grid(row=1, column=0,columnspan=7, sticky=W)

button_1 = ttk.Button(frame_btn_switch, text="棒", command=lambda: change_a(frame_bar_graph))
button_2 = ttk.Button(frame_btn_switch, text="折れ線（正）", command=lambda: change_a(frame_line_graph_true))
button_3 = ttk.Button(frame_btn_switch, text="折れ線（顔）", command=lambda: change_a(frame_line_graph_face))
button_4 = ttk.Button(frame_btn_switch, text="折れ線（目）", command=lambda: change_a(frame_line_graph_eye))
button_5 = ttk.Button(frame_btn_switch, text="文字起こし", command=lambda: change_a(frame_sentence_output))

button_1.grid(row=0,column=1)
button_2.grid(row=0,column=2)
button_3.grid(row=0,column=3)
button_4.grid(row=0,column=4)
button_5.grid(row=0,column=5)

btn_screenshot=ttk.Button(frame_btn_switch, text="保存", command=screenshot_save)
btn_screenshot.grid(row=0,column=6,padx=20)
#######################################################################

#グラフ出力ボタン######################################################
  # 棒グラフ #
btn_bar_graph = Button(frame_bar_graph, text='棒グラフ 出力', command=bar_graph)
btn_bar_graph.grid(row=0,column=0, sticky=tk.W,ipadx=40)

fig = bar_graph_set(0, 0)
canvas = FigureCanvasTkAgg(fig, frame_bar_graph)

  # 折れ線グラフ 正 #
btn_line_graph_true = Button(frame_line_graph_true, text='折れ線グラフ(正)　出力', command=line_graph_cfhi)
btn_line_graph_true.grid(row=0, column=0, sticky=tk.W)

fig_cfhi = line_graph_set_cfhi(0, 0, 0, 0, 0)
canvas_cfhi = FigureCanvasTkAgg(fig_cfhi, frame_line_graph_true)

  # 折れ線グラフ 顔 #
btn_line_graph_face = Button(frame_line_graph_face, text='折れ線グラフ(顔)　出力', command=line_graph_ab)
btn_line_graph_face.grid(row=0, column=0, sticky=tk.W)

fig_ab = line_graph_set_ab(0, 0, 0)
canvas_ab = FigureCanvasTkAgg(fig_ab, frame_line_graph_face)

  # 折れ線グラフ 目 #
btn_line_graph_eye = Button(frame_line_graph_eye, text='折れ線グラフ(目)　出力', command=line_graph_deg)
btn_line_graph_eye.grid(row=0, column=0, sticky=tk.W)

fig_deg = line_graph_set_deg(0, 0, 0, 0)
canvas_deg = FigureCanvasTkAgg(fig_deg, frame_line_graph_eye)
#######################################################################

#文字起こし############################################################
# 「ファイル参照」エントリーの作成　入力枠
sentence_output_entry = StringVar()
sentence_output_IFileEntry = tk.Entry(frame_sentence_output, textvariable=sentence_output_entry, width=40)
sentence_output_IFileEntry.grid(row=0,padx=240,sticky=tk.W)
sentence_output_entry.set(".wavファイルのみ読み込み") #z.txtは他に変えてもいい

# 「ファイル参照」ボタンの作成　
btn_filedialog_check2 = ttk.Button(frame_sentence_output, text="参照", command=filedialog_check3)
btn_filedialog_check2.grid(row=0,padx=120,sticky=tk.W)

button_5 = ttk.Button(frame_sentence_output, text="文字起こし", command=sentence_output)
button_5.grid(row=0,sticky=tk.W)
#######################################################################

root.mainloop()