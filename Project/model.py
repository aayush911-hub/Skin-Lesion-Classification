import torch
import torch.nn as nn
import torchvision

sigmoid= nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        result= i*sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx,grad_output):
        i= ctx.saved_variables[0]
        sigmoid_i= sigmoid(i)
        return grad_output*(sigmoid_i*(1+i*(1-sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class SingleModality(nn.Module):
    def __init__(self, class_list):
        super().__init__()
        self.num_label= class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.dropout = nn.Dropout(0.6)

        self.single_modality= torchvision.models.resnet50(pretrained=True)

        self.conv1_single= self.single_modality.conv1
        self.bn1_single= self.single_modality.bn1
        self.relu_single= self.single_modality.relu
        self.maxpool_single= self.single_modality.maxpool

        self.layer1_single= self.single_modality.layer1
        self.layer2_single = self.single_modality.layer2
        self.layer3_single = self.single_modality.layer3
        self.layer4_single = self.single_modality.layer4
        self.avgpool_single= self.single_modality.avgpool

        self.single_mlp= nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),

            nn.Dropout(p=0.6),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module()
        )

        self.fc_single= nn.Linear(128, self.num_label)
        self.fc_pn_single = nn.Linear(128,self.num_pn)
        self.fc_str_single = nn.Linear(128,self.num_str)
        self.fc_pig_single = nn.Linear(128,self.num_pig)
        self.fc_rs_single = nn.Linear(128,self.num_rs)
        self.fc_dag_single = nn.Linear(128,self.num_dag)
        self.fc_bwv_single = nn.Linear(128,self.num_bwv)
        self.fc_vs_single = nn.Linear(128,self.num_vs)

    def forward(self, x):
        x_single= x
        x_single= self.conv1_single(x_single)
        x_single = self.bn1_single(x_single)
        x_single = self.relu_single(x_single)
        x_single = self.maxpool_single(x_single)

        x_single = self.layer1_single(x_single)
        x_single = self.layer2_single(x_single)
        x_single = self.layer3_single(x_single)
        x_single = self.layer4_single(x_single)
        x_single = self.avgpool_single(x_single)
        x_single= x_single.view(x_single.size(0), -1)

        x_single= self.single_mlp(x_single)
        x_single= self.dropout(x_single)

        logit_single= self.fc_single(x_single)
        logit_pn_single= self.fc_pn_single(x_single)
        logit_str_single= self.fc_str_single(x_single)
        logit_pig_single= self.fc_pig_single(x_single)
        logit_rs_single= self.fc_rs_single(x_single)
        logit_dag_single= self.fc_dag_single(x_single)
        logit_bwv_single= self.fc_bwv_single(x_single)
        logit_vs_single= self.fc_vs_single(x_single)

        return [logit_single, logit_pn_single, logit_str_single, logit_pig_single,
                logit_rs_single, logit_dag_single, logit_bwv_single, logit_vs_single]

    def criterion(self, logit, truth):
        loss= nn.CrossEntropyLoss()(logit, truth)
        return loss

    def criterion1(self, logit, truth):
        loss= nn.L1Loss()(logit, truth)
        return loss

    def metric(self, logit, truth):
        _, prediction= torch.max(logit.data, 1)
        acc= torch.max(logit, truth)
        return acc

    def set_mode(self, mode):
        self.mode= mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

##############################################################

class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 1x1 reduce
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 3x3 main conv (stride applied here)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1 expand
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # projection shortcut when shape/channels don't match
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Resnet50_uncerntainty(nn.Module):
    def __init__(self, class_list):
        super(Resnet50_uncerntainty, self).__init__()

        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.num_uncertainty= class_list[8]

        self.conv1 = nn.Conv1d(1,64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1= self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool= nn.AdaptiveAvgPool1d(1)

        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 64)

        self.fc = nn.Linear(64, self.num_label)
        self.fc_pn = nn.Linear(64, self.num_pn)
        self.fc_str = nn.Linear(64, self.num_str)
        self.fc_pig = nn.Linear(64, self.num_pig)
        self.fc_rs = nn.Linear(64, self.num_rs)
        self.fc_dag = nn.Linear(64, self.num_dag)
        self.fc_bwv = nn.Linear(64, self.num_bwv)
        self.fc_vs = nn.Linear(64, self.num_vs)
        self.fc_uncertainty = nn.Linear(64, self.num_uncertainty)


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels * 4, out_channels))
            return nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc_1(x)
        x=self.fc_2(x)
        x=self.fc_3(x)

        logit_x=self.fc(x)
        logit_pn_x=self.fc_pn(x)
        logit_str_single = self.fc_str(x)
        logit_pig_single = self.fc_pig(x)
        logit_rs_single = self.fc_rs(x)
        logit_dag_single = self.fc_dag(x)
        logit_bwv_single = self.fc_bwv(x)
        logit_vs_single = self.fc_vs(x)
        logit_uncertainty_x=self.fc_uncertainty(x)

        return [(logit_x, logit_pn_x, logit_str_single, logit_pig_single,
                 logit_rs_single, logit_vs_single, logit_dag_single, logit_bwv_single,
                 logit_uncertainty_x)]

    def criterion(self, logit,truth):
        loss=nn.CrossEntropyLoss()(logit, truth)

    def criterion1(self, logit, truth):
        loss= nn.L1Loss()(logit, truth)
        return loss

    def criterion_MSE(self, logit, truth):
        loss= nn.Sigmoid()(logit)
        loss= nn.MSELoss()(logit, truth)
        return loss

    def metric(self, logit, truth):
        _, prediction= torch.max(logit.data, 1)
        acc= torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode= mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


##########################################################
class MLP_with_uncertainty(nn.Module):
    def __init__(self,class_list):
        super().__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.num_uncertainty = class_list[8]
        self.dropout = nn.Dropout(0.6)


        #define MLP Net
        self.mlp = nn.Sequential(
            nn.Linear(42,1000),#(one_hot_vector,xx)
            nn.BatchNorm1d(1000),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(1000, 42),
            nn.BatchNorm1d(42),
            nn.ReLU(),
        )
        self.fc = nn.Linear(42, self.num_label)
        self.fc_pn = nn.Linear(42, self.num_pn)
        self.fc_str = nn.Linear(42, self.num_str)
        self.fc_pig = nn.Linear(42, self.num_pig)
        self.fc_rs = nn.Linear(42, self.num_rs)
        self.fc_dag = nn.Linear(42, self.num_dag)
        self.fc_bwv = nn.Linear(42, self.num_bwv)
        self.fc_vs = nn.Linear(42, self.num_vs)
        self.fc_uncertainty = nn.Linear(42, self.num_uncertainty)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        #print("x.shape:",x.shape)
        x = self.mlp(x)
        x = self.dropout(x)
        logit_x = self.fc(x)
        logit_pn_x = self.fc_pn(x)
        logit_str_x = self.fc_str(x)
        logit_pig_x = self.fc_pig(x)
        logit_rs_x = self.fc_rs(x)
        logit_dag_x = self.fc_dag(x)
        logit_bwv_x = self.fc_bwv(x)
        logit_vs_x = self.fc_vs(x)
        logit_uncertainty_x = self.fc_uncertainty(x)

        return [(logit_x, logit_pn_x, logit_str_x, logit_pig_x, logit_rs_x, logit_dag_x, logit_bwv_x, logit_vs_x,
                 logit_uncertainty_x)]

    def criterion(self,logit,truth):
        loss = nn.CrossEntropyLoss()(logit,truth)
        return loss

    def criterion1(self,logit,truth):
        loss = nn.L1Loss()(logit,truth)
        return loss

    def criterion_MSE(self,logit,truth):
        logit = nn.Sigmoid()(logit)
        loss = nn.MSELoss()(logit,truth)
        return loss

    def metric(self, logit, truth):
        _, prediction = torch.max(logit.data, 1)
        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ["eval", "valid", "test"]:
            self.eval()
        elif mode in ["train"]:
            self.train()
        else:
            raise NotImplementedError


#####################################################
#ResNet-with-Uncertainty: clinic
class Resnet_with_uncertainty_clinic(nn.Module):
    def __init__(self,class_list):
        super().__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.num_uncertainty = class_list[8]
        self.dropout = nn.Dropout(0.8)

        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm = torchvision.models.resnet50(pretrained=True)

        #define the clinic model
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3
        self.layer4_cli = self.model_clinic.layer4
        self.avgpool_cli = self.model_clinic.avgpool

        #define the derm model
        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.layer4_derm = self.model_derm.layer4
        self.avgpool_derm = self.model_derm.avgpool

        self.global_avg = nn.AdaptiveAvgPool2d((1,1))

        self.cli_mlp = nn.Sequential(
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.derm_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(2048*2,1024),
            nn.BatchNorm1d(1024),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(1024,256),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        self.fc_cli = nn.Linear(128, self.num_label)
        self.fc_pn_cli = nn.Linear(128, self.num_pn)
        self.fc_str_cli = nn.Linear(128, self.num_str)
        self.fc_pig_cli = nn.Linear(128, self.num_pig)
        self.fc_rs_cli = nn.Linear(128, self.num_rs)
        self.fc_dag_cli = nn.Linear(128, self.num_dag)
        self.fc_bwv_cli = nn.Linear(128, self.num_bwv)
        self.fc_vs_cli = nn.Linear(128, self.num_vs)
        self.fc_uncertainty_cli = nn.Linear(128,self.num_uncertainty)

        # self.fc_derm_ = nn.Linear(2048, 512)
        self.fc_derm = nn.Linear(128, self.num_label)
        self.fc_pn_derm = nn.Linear(128, self.num_pn)
        self.fc_str_derm = nn.Linear(128, self.num_str)
        self.fc_pig_derm = nn.Linear(128, self.num_pig)
        self.fc_rs_derm = nn.Linear(128, self.num_rs)
        self.fc_dag_derm = nn.Linear(128, self.num_dag)
        self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
        self.fc_vs_derm = nn.Linear(128, self.num_vs)
        self.fc_uncertainty_derm = nn.Linear(128, self.num_uncertainty)

        self.fc = nn.Linear(128, self.num_label)
        self.fc_pn = nn.Linear(128, self.num_pn)
        self.fc_str = nn.Linear(128, self.num_str)
        self.fc_pig = nn.Linear(128, self.num_pig)
        self.fc_rs = nn.Linear(128, self.num_rs)
        self.fc_dag = nn.Linear(128, self.num_dag)
        self.fc_bwv = nn.Linear(128, self.num_bwv)
        self.fc_vs = nn.Linear(128, self.num_vs)
        self.fc_uncertainty = nn.Linear(128,self.num_uncertainty)

    def forward(self,x):
        x_clinic = x
        x_clinic = self.conv1_cli(x_clinic)
        x_clinic = self.bn1_cli(x_clinic)
        x_clinic = self.relu_cli(x_clinic)
        x_clinic = self.maxpool_cli(x_clinic)
        x_clinic = self.layer1_cli(x_clinic)
        x_clinic = self.layer2_cli(x_clinic)
        x_clinic = self.layer3_cli(x_clinic)
        x_clinic = self.layer4_cli(x_clinic)
        x_clinic = self.avgpool_cli(x_clinic)

        x_clinic = self.global_avg(x_clinic)

        x_clinic = x_clinic.view(x_clinic.size(0),-1)
        x_clinic = self.cli_mlp(x_clinic)
        x_clinic = self.dropout(x_clinic)
        logit_clinic = self.fc_cli(x_clinic)
        logit_pn_clinic = self.fc_pn_cli(x_clinic)
        logit_str_clinic = self.fc_str_cli(x_clinic)
        logit_pig_clinic = self.fc_pig_cli(x_clinic)
        logit_rs_clinic = self.fc_rs_cli(x_clinic)
        logit_dag_clinic = self.fc_dag_cli(x_clinic)
        logit_bwv_clinic = self.fc_bwv_cli(x_clinic)
        logit_vs_clinic = self.fc_vs_cli(x_clinic)
        logit_uncertainty_clinic = self.fc_uncertainty_cli(x_clinic)

        return [(logit_clinic,logit_pn_clinic,logit_str_clinic,logit_pig_clinic,logit_rs_clinic,logit_dag_clinic,logit_bwv_clinic,logit_vs_clinic,logit_uncertainty_clinic)]

    def criterion(self,logit,truth):
        loss = nn.CrossEntropyLoss()(logit,truth)
        return loss

    def criterion1(self,logit,truth):
        loss = nn.L1Loss()(logit,truth)
        return loss

    def criterion_MSE(self,logit,truth):
        logit = nn.Sigmoid()(logit)
        loss = nn.MSELoss()(logit,truth)
        return loss

    def metric(self, logit, truth):
        _, prediction = torch.max(logit.data, 1)
        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ["eval", "valid", "test"]:
            self.eval()
        elif mode in ["train"]:
            self.train()
        else:
            raise NotImplementedError


#########################################################
class Resnet_with_uncertainty(nn.Module):
    def __init__(self,class_list):
        super().__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.num_uncertainty = class_list[8]
        self.dropout = nn.Dropout(0.8)

        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm = torchvision.models.resnet50(pretrained=True)

        #define the clinic model
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3
        self.layer4_cli = self.model_clinic.layer4
        self.avgpool_cli = self.model_clinic.avgpool

        #define the derm model
        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.layer4_derm = self.model_derm.layer4
        self.avgpool_derm = self.model_derm.avgpool

        self.global_avg = nn.AdaptiveAvgPool2d((1,1))

        self.cli_mlp = nn.Sequential(
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.derm_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(2048*2,1024),
            nn.BatchNorm1d(1024),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(1024,256),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        self.fc_cli = nn.Linear(128, self.num_label)
        self.fc_pn_cli = nn.Linear(128, self.num_pn)
        self.fc_str_cli = nn.Linear(128, self.num_str)
        self.fc_pig_cli = nn.Linear(128, self.num_pig)
        self.fc_rs_cli = nn.Linear(128, self.num_rs)
        self.fc_dag_cli = nn.Linear(128, self.num_dag)
        self.fc_bwv_cli = nn.Linear(128, self.num_bwv)
        self.fc_vs_cli = nn.Linear(128, self.num_vs)
        self.fc_uncertainty_cli = nn.Linear(128,self.num_uncertainty)

        # self.fc_derm_ = nn.Linear(2048, 512)
        self.fc_derm = nn.Linear(128, self.num_label)
        self.fc_pn_derm = nn.Linear(128, self.num_pn)
        self.fc_str_derm = nn.Linear(128, self.num_str)
        self.fc_pig_derm = nn.Linear(128, self.num_pig)
        self.fc_rs_derm = nn.Linear(128, self.num_rs)
        self.fc_dag_derm = nn.Linear(128, self.num_dag)
        self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
        self.fc_vs_derm = nn.Linear(128, self.num_vs)
        self.fc_uncertainty_derm = nn.Linear(128, self.num_uncertainty)

        self.fc = nn.Linear(128, self.num_label)
        self.fc_pn = nn.Linear(128, self.num_pn)
        self.fc_str = nn.Linear(128, self.num_str)
        self.fc_pig = nn.Linear(128, self.num_pig)
        self.fc_rs = nn.Linear(128, self.num_rs)
        self.fc_dag = nn.Linear(128, self.num_dag)
        self.fc_bwv = nn.Linear(128, self.num_bwv)
        self.fc_vs = nn.Linear(128, self.num_vs)
        self.fc_uncertainty = nn.Linear(128,self.num_uncertainty)

    def forward(self,x):
        (x_clic,x_derm) = x
        x_clic = self.conv1_cli(x_clic)
        x_clic = self.bn1_cli(x_clic)
        x_clic = self.relu_cli(x_clic)
        x_clic = self.maxpool_cli(x_clic)
        x_clic = self.layer1_cli(x_clic)
        x_clic = self.layer2_cli(x_clic)
        x_clic = self.layer3_cli(x_clic)
        x_clic = self.layer4_cli(x_clic)
        x_clic = self.avgpool_cli(x_clic)

        x_derm = self.conv1_derm(x_derm)
        x_derm = self.bn1_derm(x_derm)
        x_derm = self.relu_derm(x_derm)
        x_derm = self.maxpool_derm(x_derm)
        x_derm = self.layer1_derm(x_derm)
        x_derm = self.layer2_derm(x_derm)
        x_derm = self.layer3_derm(x_derm)
        x_derm = self.layer4_derm(x_derm)
        x_derm = self.avgpool_derm(x_derm)

        x_clic = self.global_avg(x_clic)
        x_derm = self.global_avg(x_derm)
        #print("x_clic.shape:",x_clic.size())
        #print("x_derm.shape:",x_derm.size())
        x = torch.cat((x_clic,x_derm),1)
        x = x.view(x.size(0),-1)
        #print("x.shape:",x.size())

        x_clic = x_clic.view(x_clic.size(0),-1)
        x_clic_coss = x_clic
        #print("x_clic_coss.size:",x_clic_coss.size())
        x_clic = self.cli_mlp(x_clic)
        x_clic = self.dropout(x_clic)
        logit_clic = self.fc_cli(x_clic)
        logit_pn_clic = self.fc_pn_cli(x_clic)
        logit_str_clic = self.fc_str_cli(x_clic)
        logit_pig_clic = self.fc_pig_cli(x_clic)
        logit_rs_clic = self.fc_rs_cli(x_clic)
        logit_dag_clic = self.fc_dag_cli(x_clic)
        logit_bwv_clic = self.fc_bwv_cli(x_clic)
        logit_vs_clic = self.fc_vs_cli(x_clic)
        logit_uncertainty_clic = self.fc_uncertainty_cli(x_clic)

        x_derm = x_derm.view(x.size(0),-1)
        x_derm_coss = x_derm
        x_derm = self.derm_mlp(x_derm)
        x_derm = self.dropout(x_derm)
        logit_derm = self.fc_derm(x_derm)
        logit_pn_derm = self.fc_pn_derm(x_derm)
        logit_str_derm = self.fc_str_derm(x_derm)
        logit_pig_derm = self.fc_pig_derm(x_derm)
        logit_rs_derm = self.fc_rs_derm(x_derm)
        logit_dag_derm = self.fc_dag_derm(x_derm)
        logit_bwv_derm = self.fc_bwv_derm(x_derm)
        logit_vs_derm = self.fc_vs_derm(x_derm)
        logit_uncertainty_derm = self.fc_uncertainty_derm(x_derm)


        x = self.mlp(x)
        x = self.dropout(x)
        logit_x = self.fc(x)
        logit_pn_x = self.fc_pn(x)
        logit_str_x = self.fc_str(x)
        logit_pig_x = self.fc_pig(x)
        logit_rs_x = self.fc_rs(x)
        logit_dag_x = self.fc_dag(x)
        logit_bwv_x = self.fc_bwv(x)
        logit_vs_x = self.fc_vs(x)
        logit_uncertainty_x = self.fc_uncertainty(x)

        return [(logit_clic,logit_pn_clic,logit_str_clic,logit_pig_clic,logit_rs_clic,logit_dag_clic,logit_bwv_clic,logit_vs_clic,logit_uncertainty_clic),
                (logit_derm,logit_pn_derm,logit_str_derm,logit_pig_derm,logit_rs_derm,logit_dag_derm,logit_bwv_derm,logit_vs_derm,logit_uncertainty_derm),
                (logit_x,logit_pn_x,logit_str_x,logit_pig_x,logit_rs_x,logit_dag_x,logit_bwv_x,logit_vs_x,logit_uncertainty_x),
                (x_clic_coss,x_derm_coss)]

    def criterion(self,logit,truth):
        loss = nn.CrossEntropyLoss()(logit,truth)
        return loss

    def criterion1(self,logit,truth):
        loss = nn.L1Loss()(logit,truth)
        return loss

    def criterion_MSE(self,logit,truth):
        logit = nn.Sigmoid()(logit)
        loss = nn.MSELoss()(logit,truth)
        return loss

    def metric(self, logit, truth):
        _, prediction = torch.max(logit.data, 1)
        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ["eval", "valid", "test"]:
            self.eval()
        elif mode in ["train"]:
            self.train()
        else:
            raise NotImplementedError

##############################################
#class ResNEt-with-Uncertainty: derm
class Resnet_with_uncertainty_derm(nn.Module):
    def __init__(self,class_list):
        super().__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.num_uncertainty = class_list[8]
        self.dropout = nn.Dropout(0.8)

        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm = torchvision.models.resnet50(pretrained=True)

        #define the clinic model
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3
        self.layer4_cli = self.model_clinic.layer4
        self.avgpool_cli = self.model_clinic.avgpool

        #define the derm model
        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.layer4_derm = self.model_derm.layer4
        self.avgpool_derm = self.model_derm.avgpool

        self.global_avg = nn.AdaptiveAvgPool2d((1,1))

        self.cli_mlp = nn.Sequential(
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.derm_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(2048*2,1024),
            nn.BatchNorm1d(1024),
            Swish_Module(),
            nn.Dropout(0.8),
            nn.Linear(1024,256),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        self.fc_cli = nn.Linear(128, self.num_label)
        self.fc_pn_cli = nn.Linear(128, self.num_pn)
        self.fc_str_cli = nn.Linear(128, self.num_str)
        self.fc_pig_cli = nn.Linear(128, self.num_pig)
        self.fc_rs_cli = nn.Linear(128, self.num_rs)
        self.fc_dag_cli = nn.Linear(128, self.num_dag)
        self.fc_bwv_cli = nn.Linear(128, self.num_bwv)
        self.fc_vs_cli = nn.Linear(128, self.num_vs)
        self.fc_uncertainty_cli = nn.Linear(128,self.num_uncertainty)

        # self.fc_derm_ = nn.Linear(2048, 512)
        self.fc_derm = nn.Linear(128, self.num_label)
        self.fc_pn_derm = nn.Linear(128, self.num_pn)
        self.fc_str_derm = nn.Linear(128, self.num_str)
        self.fc_pig_derm = nn.Linear(128, self.num_pig)
        self.fc_rs_derm = nn.Linear(128, self.num_rs)
        self.fc_dag_derm = nn.Linear(128, self.num_dag)
        self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
        self.fc_vs_derm = nn.Linear(128, self.num_vs)
        self.fc_uncertainty_derm = nn.Linear(128, self.num_uncertainty)

        self.fc = nn.Linear(128, self.num_label)
        self.fc_pn = nn.Linear(128, self.num_pn)
        self.fc_str = nn.Linear(128, self.num_str)
        self.fc_pig = nn.Linear(128, self.num_pig)
        self.fc_rs = nn.Linear(128, self.num_rs)
        self.fc_dag = nn.Linear(128, self.num_dag)
        self.fc_bwv = nn.Linear(128, self.num_bwv)
        self.fc_vs = nn.Linear(128, self.num_vs)
        self.fc_uncertainty = nn.Linear(128,self.num_uncertainty)

    def forward(self,x):
        x_derm = x

        x_derm = self.conv1_derm(x_derm)
        x_derm = self.bn1_derm(x_derm)
        x_derm = self.relu_derm(x_derm)
        x_derm = self.maxpool_derm(x_derm)
        x_derm = self.layer1_derm(x_derm)
        x_derm = self.layer2_derm(x_derm)
        x_derm = self.layer3_derm(x_derm)
        x_derm = self.layer4_derm(x_derm)
        x_derm = self.avgpool_derm(x_derm)

        x_derm = self.global_avg(x_derm)

        x_derm = x_derm.view(x.size(0),-1)
        x_derm = self.derm_mlp(x_derm)
        x_derm = self.dropout(x_derm)
        logit_derm = self.fc_derm(x_derm)
        logit_pn_derm = self.fc_pn_derm(x_derm)
        logit_str_derm = self.fc_str_derm(x_derm)
        logit_pig_derm = self.fc_pig_derm(x_derm)
        logit_rs_derm = self.fc_rs_derm(x_derm)
        logit_dag_derm = self.fc_dag_derm(x_derm)
        logit_bwv_derm = self.fc_bwv_derm(x_derm)
        logit_vs_derm = self.fc_vs_derm(x_derm)
        logit_uncertainty_derm = self.fc_uncertainty_derm(x_derm)

        return [(logit_derm,logit_pn_derm,logit_str_derm,logit_pig_derm,logit_rs_derm,logit_dag_derm,logit_bwv_derm,logit_vs_derm,logit_uncertainty_derm)]

    def criterion(self,logit,truth):
        loss = nn.CrossEntropyLoss()(logit,truth)
        return loss

    def criterion1(self,logit,truth):
        loss = nn.L1Loss()(logit,truth)
        return loss

    def criterion_MSE(self,logit,truth):
        logit = nn.Sigmoid()(logit)
        loss = nn.MSELoss()(logit,truth)
        return loss

    def metric(self, logit, truth):
        _, prediction = torch.max(logit.data, 1)
        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ["eval", "valid", "test"]:
            self.eval()
        elif mode in ["train"]:
            self.train()
        else:
            raise NotImplementedError


