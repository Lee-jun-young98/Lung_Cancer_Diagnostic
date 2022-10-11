import math

from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


# 분류 모델에는 테일, 백본(혹은 바디), 헤드로 구성된 구조가 흔함
# 테일은 입력을 신경망에 넣기 전 처리 과정을 담당하는 제일 첫 부분의 일부 계층
# 이러한 앞 단 계층은 백본이 원하는 형태로 입력을 만들어야 하므로 신경망의 나머지 부분과는 구조나 구성이 다른 경우가 많음
# 테일에 컨볼루션은 이미지 크기를 공격적으로 다운 샘플링하기 위한 용도가 대부부분임

# 신경망의 백본은 여러 계층을 가지는데 일반적으로는 연속된 블럭에 배치됨
# 각 블럭은 동일한 세트의 계층을 가지며 블럭과 블럭을 거칠 때마다 필요한 입력 크기나 필터 수가 달라짐
# 우리는 두 개의 3X3 컨볼루션과 하나의 활성화, 그리고 블록 끝에 맥스 풀링 연산이 이어진 블럭을 사용  

# 헤드는 백본의 출력을 받아 원하는 출력 형태로 바꿈
# 컨볼루션 신경망에서 이 작업은 중간 출력물을 평탄화(flattening)하고 완전 연결 계층(fully connected layer)에 전달하는 역할을 하기도 함
# 이 블럭에서는 3 X 3 X 3 컨볼루션을 사용하며, 하나의 3 X 3 X 3 컨볼루션은 거의 동일하게 3 X 3 X 3 크기의 수용 필드를 가짐
# 27개의 복셀이 들어오고, 1개를 출력함

class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1) # tail 부분 nn.BatchNorm3d를 사용해 입력을 정규화, 평균이 0이고 표준 정규분포 1을 따름

        # 백본 모델 
        # 4개의 층을 거치므로 이미지는 각 차원마다 16배 줄어들음
        # 데이터 크기가 32 X 48 X 48에서 백본을 거치면 2 X 3 X 3이 됨
        # 마지막 tail에는 완전 연결 계층 뒤로 nn.Softmax가 따라옴
        self.block1 = LunaBlock(in_channels, conv_channels) 
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    # 초기화
    # 모델이 좋은 성능을 낼 수 있도록 동작하려면, 모델의 가중치, 편향값, 여러 파라미터가 특정 속성을 드러내야 함
    # 모든 가중치가 1보다 커지는 좋지 않은 경우(잔차 연결이 없는 경우)
    # -> 가중치 값으로 반복되는 곱셈 연산은 신경망층으로 전달되는 과정을 거치면서 계층 출력 값이 매우 커짐
    # 모든 가중치가 1보다 작은 경우
    # -> 모든 층의 출력을 점점 작게 만들어 아예 없어지게 됨
    # 위의 문제를 해결하기 위해 여러 가지 정규화 기술로 계층의 출력을 잘 동작하게 만듦
    # -> 가장 단순한 방법은 중간 값이나 기울기가 이유 없이 작아지거나 커지지 않도록 신경망 가중치를 완전하게 초기화하는 방법
    # 파이토치에서는 직접 초기화를 해야함
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_( # 가중치 초기화
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound) # (tesnor, mean, std)의 정규분포로 초기화


    # self.block4의 출력에 완전 연결 계층을 넣을 수 없음
    # 출력은 샘플마다 2 X 3 X 3 이미지에 64개의 채널을 가지는데, 완전 연결 계층은 입력으로 1차원 벡터를 받음
    # 여기서 1차원 벡터는 1차원 벡터의 배치를 받으므로 (배치 수, 1차원 벡터)
    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0), # 배치 크기
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch) 
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
