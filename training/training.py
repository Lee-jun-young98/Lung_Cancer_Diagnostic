# 연습문제
# 1. DataLoader 인스턴스 내에 래핑한 LunaDataset 인스턴스를 순회하는 프로그램을 만들어라
# 순회할 때 걸리는 시간도 알 수 있게 만들기
#   a. num_worker = 0,1,2로 할 때 어떤 차이가 발생하는가
#       - 0일 때 CPU의 사용값이 높아지며 각 배치당 1분내지의 시간이 걸린다.
#   b. 메모리가 모자라지 않는 선에서 최대로 끌어올릴 수 있는 batch_size = ...와 num_workers=...는 얼마인가
# 2. noduleInfo_list의 정렬 순서를 반대로 해보자. 이렇게 바꾸면 훈련의 첫 에포크 후에 동작 방식에서 어떤 차이가 발생하는가?
# 3. logMetrics를 바꿔서 텐서보드가 사용하는 실행 아이템 이름과 키를 변경해보자
#   a. writer.add_scalar로 전달되는 키 값에 처음으로 나타나는 슬래시 문자 위치를 바꿔보자.
#   b. 동일한 쓰기 객체를 사용해서 훈련과 검증에 대해 돌려본 후 키 이름에 trn이나 val문자열을 덧붙여보자.
#   c. 로그 디렉토리와 키 값을 원하는 대로 바꿔보자.


import argparse
import datetime
import enum
from operator import neg
import os
import sys
from unittest.loader import VALID_MODULE_NAME

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import sys
sys.path.append("C:/Lung_Cancer_Diagnostic_py")
from util.util import enumerateWithEstimate
from LunaDataset.dsets import LunaDataset
from util.logconf import logging
from LunaDataset.model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
# ComputeBatchLoss와 logMetrics는 metrics_t/metrics_a로 인덱싱하는데 사용됨
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        # 16GB 램이 장착된 4코어와 8스레드 CPU와 8GB 램이 장착된 GPU를 사용하고 있다고 가정
        # GPU의 램 사이즈가 이보다 작다면 --batch-size를 줄이고 CPU 코어 수가 작거나 CPU램이 작다면 --num-worker를 줄이기
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help = 'Number of worker processes for background data loading',
            default=1, # 8
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16, # 32
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--balanced', # 데이터 균형 파라미터 추가
            help="Balance the training data to half positive, half negative.",
            action = 'store_true',
            default=False,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch11',
            help="data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs = '?',
            default='dwlpt',
        )
        self.   cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
    
    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1])
            model = model.to(self.device)
        return model
    
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
        )
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count() # gpu개수에 배치사이즈를 곱함

        train_dl = DataLoader(
            train_ds,
            batch_size = batch_size,
            num_workers = self.cli_args.num_workers,
            pin_memory = self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)
            
    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl() # train 데이터셋 초기화
        val_dl = self.initValDl() # validation 데이터셋 초기화

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1), 
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    # trnMetrics_g : 텐서가 훈련 중에 자세한 클래스 단위 메트릭을 수집함, 규모가 큰 프로젝트에서는 이 값으로 통찰을 얻는 경우가
    # 많기 때문에 필요
    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros( # 빈 메트릭 배열을 초기화 
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )
        # train_dl을 직접 순회하지 않으며, 완료 시간 예측을 제공하기 위해 enumerateWithEstimate를 사용 
        batch_iter = enumerateWithEstimate( # 시간을 예측하며 배치 루프를 설정함.
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad() # 남은 가중치 텐서를 해제함
            # 실제 손실 계산이 이루어지는 곳
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #       with torch.no_graad():
            #           model = LunaModel()
            #           self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #           self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval() # 훈련 때 사용했던 기능 끄기 -> 기울기 계산이 필요 없기 때문에 성능이 향상 된다.
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
        
        return valMetrics_g.to('cpu')
    

    
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        """
        훈련 루프와 검증 루프 양쪽에서 모두 호출됨
        샘플 배치에 대해 손실을 계산함
        각 클래스별로 계산이 얼마나 정확한지 백분율로 계산할 수 있음
        분류가 잘 되지 않는 클래스를 찾아 집중 개선할 수 있음
        """
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True) # 배치 튜플의 패킹을 풀고 텐서를 GPU로 옮김
        label_g = label_t.to(self.device, non_blocking=True) # 배치 튜플의 패킹을 풀고 텐서를 GPU로 옮김

        logits_g, probabillity_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none') # 샘플별 손실값을 얻음
        loss_g = loss_func(
            logits_g,
            label_g[:,1], # 원 핫 인코딩 클래스의 인덱스
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        # 기울기에 의존적인 메트릭이 없으므로 detach를 사용
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probabillity_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean() # 샘플별 손실값을 단일값으로 합침 -> 전체 배치에 대한 손실값

    def logMetrics(
            self,
            epoch_ndx, # epoch 표시
            mode_str, # train, val 확인
            metrics_t, # trnMetrics_t, valMetrics_t -> computeBatchLoss를 통해 만들어진 부동 소수점 텐서
            classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold # 결절 샘플에 대해서 메트릭을 제한함
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold # 예측 값에 대해서 메트릭을 제한함

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum()) # 음성 개수
        pos_count = int(posLabel_mask.sum()) # 양성 개수

        # 각 에포크마다 정밀도와 재현율을 출력하도록 추가

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum()) # trueNeg_count
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum()) # truePos_count

        falsePos_count = neg_count - neg_correct # 거짓 양성 비율 # 실제로 음성
        falseNeg_count = pos_count - pos_correct # 거짓 음성 비율 # 실제로 양성

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()
            
        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)
        

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                + "{correct/all:-5.1f}% correct, "
                + "{pr/precision:.4f} precision, "
                + "{pr/recall:.4f} recall, "
                + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        
        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise

if __name__ == '__main__':
    LunaTrainingApp().main()