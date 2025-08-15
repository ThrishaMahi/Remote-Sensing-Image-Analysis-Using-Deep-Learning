from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import cv2

from keras.callbacks import ModelCheckpoint
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD

main = Tk()
main.title("Remote Sensing Image Analysis using Deep Learning")
main.geometry("1300x1200")

global filename, dataset
global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes, cnn_model
labels = ['Car', 'Vegetation', 'Airplane', 'Boat', 'Tree']
global X, Y, boundings

def convert_bb(img, width, height, xmin, ymin, xmax, ymax):
    bb = []
    conv_x = (200. / width)
    conv_y = (200. / height)
    height = ymax * conv_y
    width = xmax * conv_x
    x = max(xmin * conv_x, 0)
    y = max(ymin * conv_y, 0)     
    x = x / 200
    y = y / 200
    width = width/200
    height = height/200
    return x, y, width, height

def addBoxes(img, label_file, name):
    boxes = []
    yy = []
    height, width = img.shape[:2]
    if os.path.exists("Dataset/YOLO_labels/"+label_file+"/"+name):
        file = open("Dataset/YOLO_labels/"+label_file+"/"+name, 'r')
        lines = file.readlines()            
        file.close()
        if len(lines) > 0:
            for i in range(len(lines)):
                line = lines[i]
                line = line.split(" ")
                x1, y1, x2, y2 = getBox(img, line[1], line[2], line[3], line[4])
                x1, y1, x2, y2 = convert_bb(img, width, height, x1, y1, x2, y2)#normalized bounding boxes
                boxes.append([x1, y1, x2, y2])
                yy.append(int(line[0].strip()))
    return np.asarray(yy), np.asarray(boxes) 

#fucntion to upload dataset
def uploadDataset():
    global X, Y, boundings
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')#save all processed images
        Y = np.load('model/Y.txt.npy')                    
        boundings = np.load('model/bb.txt.npy')
    else:
        boundings = []
        X = []
        Y = []
        for root, dirs, directory in os.walk('Dataset/RSSOD_train_HR'):#if not processed images then loop all annotation files with bounidng boxes
            for j in range(len(directory)):
                name = directory[j]
                name = name.replace(".png", ".txt")
                img = cv2.imread("Dataset/RSSOD_train_HR/"+directory[j])
                #img = cv2.resize(img, (64, 64))
                yy1, box1 = addBoxes(img, "labels_1class", name)
                yy2, box2 = addBoxes(img, "labels_2classes", name)
                yy3, box3 = addBoxes(img, "labels_4classes", name)
                yy4, box4 = addBoxes(img, "labels_5classes", name)
                yy = []
                box = []
                for m in range(0, 20):
                    box.append(0)
                for m in range(0, 5):
                    yy.append(0)
                start = 0    
                for i in range(len(box1)):
                    if start < 20:
                        x1, y1, x2, y2 = box1[i]
                        box[start] = x1
                        start += 1
                        box[start] = y1
                        start += 1
                        box[start] = x2
                        start += 1
                        box[start] = y2
                        start += 1
                        yy[yy1[i]] = 1
                        label += 1
                for i in range(len(box2)):
                    if start < 20:
                        x1, y1, x2, y2 = box2[i]
                        box[start] = x1
                        start += 1
                        box[start] = y1
                        start += 1
                        box[start] = x2
                        start += 1
                        box[start] = y2
                        start += 1
                        yy[yy2[i]] = 1
                        label += 1
                for i in range(len(box3)):
                    if start < 20:
                        x1, y1, x2, y2 = box3[i]
                        box[start] = x1
                        start += 1
                        box[start] = y1
                        start += 1
                        box[start] = x2
                        start += 1
                        box[start] = y2
                        start += 1
                        yy[yy3[i]] = 1
                        label += 1
                for i in range(len(box4)):
                    if start < 20:
                        x1, y1, x2, y2 = box4[i]
                        box[start] = x1
                        start += 1
                        box[start] = y1
                        start += 1
                        box[start] = x2
                        start += 1
                        box[start] = y2
                        start += 1
                        yy[yy4[i]] = 1
                        label += 1
            if len(box) > 0:
                print(str(box)+" "+str(yy))
                img = cv2.resize(img, (200, 200))#Resize image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boundings.append(box)
                X.append(img)
                Y.append(yy)        
                
        X = np.asarray(X)#convert array to numpy format
        Y = np.asarray(Y)
        boundings = np.asarray(boundings)
        np.save('model/X.txt',X)#save all processed images
        np.save('model/Y.txt',Y)                    
        np.save('model/bb.txt',boundings)
    text.insert(END,"Remote Sensing Dataset Loaded\n\n")    
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Objects or Labels Found in Dataset : "+str(labels))
    unique, count = np.unique(np.argmax(Y, axis=1), return_counts=True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel('Objects Type')
    plt.ylabel('Count')
    plt.title("Dataset Class Labels Graph")
    plt.show()


def preprocess():
    text.delete('1.0', END)
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
    global X, Y, boundings
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    boundings = boundings[indices]
    split = train_test_split(X, Y, boundings, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    text.insert(END,"Dataset Image Processing & Normalization Completed\n\n")
    text.insert(END,"Total Images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each Image : "+str((X.shape[1] * X.shape[2] * X.shape[3]))+"\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(trainImages.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(testImages.shape[0])+"\n")
    text.update_idletasks()
    img = cv2.imread("Dataset/RSSOD_train_HR/airplane_085.png")
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)    

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runCNNAlgorithm():
    text.delete('1.0', END)
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes, cnn_model
    #define input shape
    input_img = Input(shape=(200, 200, 3))
    #create YoloV4 layers with 32, 64 and 512 neurons or data filteration size
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    #define output layer with 4 bounding box coordinate and 1 weapan class
    x = Dense(64, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    x_bb = Dense(20, name='bb',activation='softmax')(x)
    x_class = Dense(Y.shape[1], activation='sigmoid', name='class')(x)
    #create CNN Model with above input details
    cnn_model = Model([input_img], [x_bb, x_class])
    #compile the model
    cnn_model.compile(Adam(lr=0.0001), loss=['mse', 'binary_crossentropy'], metrics=['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:#if model not trained then train the model
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(trainImages, [trainBBoxes, trainLabels], batch_size=32, epochs=30, validation_data=(testImages, [testBBoxes, testLabels]), callbacks=[model_check_point])
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:#if model already trained then load it
        cnn_model = load_model("model/cnn_weights.hdf5")
    predict = cnn_model.predict(testImages)#perform prediction on test data
    predict = predict[1]
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(testLabels, axis=1)   
    calculateMetrics("Deep Learning CNN Algorithm", predict, y_test1)


def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    print(train_values)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value
    

def graph():
    acc, loss = values("model/cnn_history.pckl", "accuracy", "val_accuracy")    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.title('CNN Algorithm Training & Validation Accuracy Graph')
    plt.show()

def predict():
    global cnn_model, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (200, 200))
    img1 = img.reshape(1,200,200,3)
    predict_value = cnn_model.predict(img1)#perform prediction on test data using extension model
    predict = predict_value[0]#get bounding boxes
    predict = predict[0]
    predicted_label = predict_value[1][0]
    flag = True
    start = 0
    label = labels[np.argmax(predicted_label)]        
    while flag:#now loop and plot all detected objects
        if start < 20:
            x1 = predict[start] * 200
            start += 1
            y1 = predict[start] * 200
            start += 1
            x2 = predict[start] * 200
            start += 1
            y2 = predict[start] * 200
            start += 1
            print(str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+label)
            if x1 > 0 and y1 > 0 and x2 > 100 and y2 > 0:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1, 1)
                cv2.putText(img, str(label), (int(x1), int(y1+50)),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 1)            
        else:
            flag = False
    img = cv2.resize(img, (400, 400))        
    cv2.imshow("Predicted Output", img)
    cv2.waitKey(0)



def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Remote Sensing Image Analysis using Deep Learning')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Remote Sensing Dataset", command=uploadDataset)
uploadButton.place(x=20,y=550)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=300,y=550)
processButton.config(font=ff)

cnnButton = Button(main, text="Run CNN Object Detection Algorithm", command=runCNNAlgorithm)
cnnButton.place(x=510,y=550)
cnnButton.config(font=ff)

graphButton = Button(main, text="CNN Training Graph", command=graph)
graphButton.place(x=850,y=550)
graphButton.config(font=ff)

predictButton = Button(main, text="Object Detection from Test Image", command=predict)
predictButton.place(x=20,y=600)
predictButton.config(font=ff)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=300,y=600)
closeButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
