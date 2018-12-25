from rnnmodel import model

# #340classification
# train_path=r'F:\Date\draw\123\csv_340_shuffle_train.csv'
# test_path=r'F:\Date\draw\123\csv_340_shuffle_test.csv'
# modelpath=r'F:\Date\draw\result\re_model_340.h5'
# model(raw_train_path=train_path, raw_test_path=test_path, model_path=modelpath, classification_num=340, delfalse=1, continue_train=0)

#71classification
train_path=r'F:\Date\draw\123\csv_71_shuffle.csv'
test_path=r'F:\Date\draw\123\csv_71_shuffle.csv'
modelpath=r'F:\Date\draw\result\re_model_71.h5'
model(raw_train_path=train_path, raw_test_path=test_path, model_path=modelpath, classification_num=71, delfalse=1, continue_train=0,lines=10015855,n_epochs=1,maxlen = 150,lstm_units=200)