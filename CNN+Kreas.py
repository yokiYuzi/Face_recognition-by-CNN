"""
olivetti Faces是纽约大学组建的一个比较小的人脸数据库。有40个人，每人10张图片，组成一张有400张人脸的大图片。
像素灰度范围在[0,255]。整张图片大小是1190*942，20行320列，所以每张照片大小是(1190/20)*(942/20)= 57*47
程序需配置h5py：python -m pip install h5py
"""
import numpy as np
import keras
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD    # 梯度下降的优化器
from keras.optimizers import Adam #Adam的优化器
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
# 读取整张图片的数据，并设置对应标签
def get_load_data(dataset_path):
    img = Image.open(dataset_path)
    # 数据归一化。asarray是使用原内存将数据转化为np.ndarray
    img_ndarray = np.asarray(img, dtype = 'float64')/255
    # 400 pictures, size: 57*47 = 2679  
    faces_data = np.empty((400, 2679))
    for row in range(20):  
       for column in range(20):
           # flatten可将多维数组降成一维
           faces_data[row*20+column] = np.ndarray.flatten(img_ndarray[row*57:(row+1)*57, column*47:(column+1)*47])

    # 设置图片标签
    label = np.empty(400)
    for i in range(40):
        label[i*10:(i+1)*10] = i
    label = label.astype(int)

    # 分割数据集：每个人前8张图片做训练，第9张做验证，第10张做测试；所以train:320,valid:40,test:40
    train_data = np.empty((320, 2679))
    train_label = np.empty(320)
    valid_data = np.empty((40, 2679))
    valid_label = np.empty(40)
    test_data = np.empty((40, 2679))
    test_label = np.empty(40)
    for i in range(40):
        train_data[i*8:i*8+8] = faces_data[i*10:i*10+8] # 训练集对应的数据
        train_label[i*8:i*8+8] = label[i*10 : i*10+8]   # 训练集对应的标签
        valid_data[i] = faces_data[i*10+8]   # 验证集对应的数据
        valid_label[i] = label[i*10+8]       # 验证集对应的标签
        test_data[i] = faces_data[i*10+9]    # 测试集对应的数据
        test_label[i] = label[i*10+9]        # 测试集对应的标签
    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')
       
    result = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]
    return result

# CNN主体
def get_set_model(lr=0.005,decay=1e-6,momentum=0.9):
    model = Sequential()
    # 卷积1+池化1
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3), input_shape = (1, img_rows, img_cols)))
    else:
        model.add(Conv2D(nb_filters1, kernel_size=(2, 2), input_shape = (img_rows, img_cols, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 卷积2+池化2
    model.add(Conv2D(nb_filters2, kernel_size=(3, 3)))
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  

    # 全连接层1+分类器层
    model.add(Flatten())  
    model.add(Dense(1000))       #Full connection
    model.add(Activation('tanh'))  
    model.add(Dropout(0.5))  
    model.add(Dense(40))
    model.add(Activation('softmax'))  

    # 选择设置SGD优化器参数
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model  

# 训练过程，保存参数
def get_train_model(model,X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,  
          verbose=1, validation_data=(X_val, Y_val))
    # 保存参数
    model.save_weights('model_weights.h5', overwrite=True)  
    return model  

# 测试过程，调用参数
def get_test_model(model,X,Y):
    model.load_weights('model_weights.h5')  
    score = model.evaluate(X, Y, verbose=0)
    return score  

#历史记录函数
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#输出分类后图像
def save_classified_images(X_data, y_data, predictions, output_directory):
    # 确保基本输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 为每个类别创建一个文件夹
    for class_index in range(40):
        class_folder = os.path.join(output_directory, str(class_index))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # 保存每张图片到相应的类别文件夹
    for i, prediction in enumerate(predictions):
        class_folder = os.path.join(output_directory, str(prediction))
        image_array = X_data[i].reshape(img_rows, img_cols)  # 将数据重新格式化为图片格式
        image = Image.fromarray((image_array * 255).astype('uint8'))  # 转换回图像
        image.save(os.path.join(class_folder, f'image_{i}.png'))  # 保存图像


# [start]
epochs = 50          # 进行多少轮训练
batch_size = 20      # 每个批次迭代训练使用40个样本，一共可训练320/40=8个网络
img_rows, img_cols = 57, 47         # 每张人脸图片的大小
nb_filters1, nb_filters2 = 60, 80   # 两层卷积核的数目（即输出的维度）

if __name__ == '__main__':  
    # 将每个人10张图片，按8:1:1的比例拆分为训练集、验证集、测试集数据
    (X_train, y_train), (X_val, y_val),(X_test, y_test) = get_load_data('F:\\Face regonized\\Test01\\CNN_Kreas_olivettifaces\\001.gif')
    

    if K.image_data_format() == 'channels_first':    # 1为图像像素深度
        X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
        X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)  
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)  
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)  
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)  
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)  
        input_shape = (img_rows, img_cols, 1)
    
    print('X_train shape:', X_train.shape)
    # convert class vectors to binary class matrices  
    Y_train = np_utils.to_categorical(y_train, 40)
    Y_val = np_utils.to_categorical(y_val, 40)
    Y_test = np_utils.to_categorical(y_test, 40)

    model = get_set_model()
    get_train_model(model, X_train, Y_train, X_val, Y_val)
    # 训练过程，保存参数
    model = get_set_model()
    get_train_model(model, X_train, Y_train, X_val, Y_val)
    score = get_test_model(model, X_test, Y_test)

    # 测试过程，调用参数，得到准确率、预测输出
    model.load_weights('model_weights.h5')
    #classes = model.predict_classes(X_test, verbose=0)  
    #新版本已经禁用modle.predict
    predictions = model.predict(X_test, verbose=0)
    classes = np.argmax(predictions, axis=1)
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("last accuarcy:", test_accuracy)
    for i in range(0,40):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classes[i]);
    
    #绘图
    history = LossHistory()

    model.fit(X_train, Y_train, epochs=10, batch_size=32, callbacks=[history])

    # 绘制损失变化图
    plt.plot(history.losses)
    plt.title('Model Loss by Batch')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=32)

# 绘制准确度变化图
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


    # 预测测试数据
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # 保存分类后的图片
    save_classified_images(X_test, y_test, predicted_classes, 'F:\\Face regonized\\Test01\\CNN_Kreas_olivettifaces')