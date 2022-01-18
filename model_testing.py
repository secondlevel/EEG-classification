from ALL_model import *
from torch.autograd import Variable
from dataloader import read_bci_data
import pandas as pd
import torch
import numpy 
import os

def testing(x_test,y_test,device,model):

    model.eval()
    with torch.no_grad():
        model.to(device)
        n = x_test.shape[0]

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32").reshape(y_test.shape[0],)

        x_test, y_test = Variable(torch.from_numpy(x_test)),Variable(torch.from_numpy(y_test))
        x_test,y_test = x_test.to(device),y_test.to(device)
        y_pred_test = model(x_test)
        correct = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        # print("testing accuracy:",correct/n)
        return correct/n

if __name__ == "__main__":

    model_list=[EEGNet_ReLU, EEGNet_LeakyReLU, EEGNet_ELU, DeepConvNet_ReLU, DeepConvNet_LeakyReLU, DeepConvNet_ELU]
    model_file_path=["EEGNet_checkpoint_ReLU.rar","EEGNet_checkpoint_LeakyReLU.rar","EEGNet_checkpoint_ELU.rar","DeepConvNet_checkpoint_ReLU.rar","DeepConvNet_checkpoint_LeakyReLU.rar","DeepConvNet_checkpoint_ELU.rar"]

    ReLU_accuracy=[]
    LeakyReLU_accuracy=[]
    ELU_accuracy=[]

    for i in range(len(model_list)):


        filepath=os.path.abspath(os.path.dirname(__file__))+"\\checkpoint\\"+model_file_path[i]
        
        device = torch.device("cuda:0")
        model = model_list[i](2)
        model.load_state_dict(torch.load(filepath))

        train_data, train_label, test_data, test_label = read_bci_data()
        testing_accuracy = testing(test_data,test_label,device,model)

        if "LeakyReLU" in model_file_path[i]:
            LeakyReLU_accuracy.append(testing_accuracy)
        elif "ReLU" in model_file_path[i]:
            ReLU_accuracy.append(testing_accuracy)
        else:
            ELU_accuracy.append(testing_accuracy)

    df = pd.DataFrame({"ReLU":ReLU_accuracy,"LeakyReLU":LeakyReLU_accuracy,"ELU":ELU_accuracy},index=["EEGNet","DeepConvNet"])
    print(df)











