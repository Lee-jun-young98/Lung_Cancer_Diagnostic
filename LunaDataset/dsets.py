import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk # 데이터 파일 포맷을 numpy로 불러들이기 위함
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

import sys
sys.path.append(r"C:/Lung_Cancer_Diagnostic_py")
from util.disk import getCache # diskcache disk에  cache를 저장해주는 라이브러리
from util.util   import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch11_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

"""
- 인메모리나 온디스크 캐싱을 적절하게 사용하여 데이터 파이프라인 속도를 올려 놓으면 훈련 속도가 상당히 개선됨
- 훈련 데이터셋 전체를 사용하게 되면 다운로드도 오래 걸리고 필요한 디스크 공간도 커지기 때문에 전체를 사용하는 대신 훈련 프로그램 실행에 집중
- requireOnDisk_bool 파라미터를 사용하여 디스크 상에서 시리즈 UID가 발견되는 LUNA 데이터만 사용하고 이에 해당하는 엔트리는 csv에서 거름
"""

@functools.lru_cache(1) # 표준 인메모리 캐싱 라이브러리
def getCandidateInfoList(requireOnDisk_bool=True): # disk에 없는 데이터를 거르기 위함
    # 모든 디스크에 있는 모든 세트를 series_uids로 구성함
    # 이렇게 하면 모든 하위 집합을 다운로드 하지 않더라도 사용할 수 있음
    mhd_list = glob.glob('C:/Luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('C:/Luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0] # user_id
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]]) # xyz 좌표
            annotationDiameter_mm = float(row[4]) # 차원 정보

            diameter_dict.setdefault(series_uid, []).append( # 첫번째 인자로 키값, 두번째 인자로 기본값을 넘김
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    with open('C:/Luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0] # user_id

            if series_uid not in presentOnDisk_set and requireOnDisk_bool: # series_uid가 없으면 서브셋에 있지만 디스크에는 없으므로 건너뜀
                continue

            isNodule_bool = bool(int(row[4])) # 클래스
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]]) # xyz 좌표

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4: # 반경을 얻기 위해 직경을 2로 나누고, 두 개의 결절 센터가 결절의 크기 기준으로
                        break                                # 너무 떨어져 있는 지를 반지름의 절반 길이를 기준으로 판정한다.(바운딩 박스 체크)
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True) # 내림차순 정렬
    return candidateInfo_list

# @functools.lru_cache(1, typed=True)
# def getCt(series_uid):
#     return Ct(series_uid)

# @raw_cache.memoize(typed=True)
# def getCtRawCandidate(series_uid, center_xyz, width_irc):
#     ct = getCt(series_uid)
#     ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
#     return ct_chunk, center_irc

"""
개별 CT 스캔 로딩
- 디스크에서 CT 데이터를 얻어와 파이썬 객체로 변환 후 3차원 결절 밀도 데이터로 사용할 수 있도록 만드는 작업
- CT 스캔 파일의 원래 포맷은 DICOM이라고 부름
- CT 스캔 복셀은 하운스필드 단위(https://en.wikipedia.org/wiki/Hounsfield_scale) 로 표시
"""
class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'C:/Luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        """
        공기는 -1000HU(0g/cc), 물은 0HU(1g/cc), 뼈는 +1000HU(2~3g/cc)다. 
        우리가 관심잇는 종양은 대체로 1g/cc(0HU) 근처이므로 -1000HU, 1000HU을 제거하고 1g/cc가 아닌 것도 제거함
        """ 
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        """
        환자 좌표계를 사용해 결절 위치 정하기
        util.py에 있음
        1. 좌표를 XYZ 체계로 만들기 위해 IRC에서 CRI로 뒤집음
        2. 인덱스를 복셀 크기로 확대 축소
        3. 파이썬의 @를 사용하여 방향을 나타내는 행렬과 행렬곱을 수행
        4. 기준으로부터 오프셋을 더함
        """

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3) # 방향 정보를 배열로 변환하고 3 x 3 행렬 모양의 9개 요소의 배열을 reshape함
    
    # 큰 CT 복셀 배열에서 후보의 중심 배열 좌표 정보(인덱스, 행, 열)을 사용하여 후보 샘플을 잘라내기
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            # assert 조건식이 False인 경우, AssertionError 예외가 발생한다.
            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

"""
데이터 파이프라인 구조를 잘 만들어 증강 전에 캐시 단계가 놓이게 만들어야 함
-> 안 그러면 증강된 데이터 상태로 고정되어 원래 목적을 잃어버림
"""
def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
        if use_cache: # cache를 사용
            ct_chunk, center_irc = \
                getCtRawCandidate(series_uid, center_xyz, width_irc)
        else: # cache 사용하지 않을 경우 직접 불러옴
            ct = getCt(series_uid)
            ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

        ct_t= torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

        transform_t = torch.eye(4) # 4 * 4의 단위 행렬을 만듦
        # ...<1> transform_tensor를 수정
        """
        random.random() -> 0에서 1사이의 실수를 리턴함
        grid_sample 함수는 [-1, 1] 범위를 이전 텐서와 새 텐서 모두에 매핑함
        flip : 범위 매핑으로 데이터를 미러링하기 위해 변환 행렬의 관련 요소에 -1을 곱함
        offset : 오프셋이 복셀 단위의 정수가 아닌 경우에 더 두드러진 차이를 만듦. 삼중 선형 보간법을 사용하여 약간의 흐림(blur) 처리가 들억는 형태로 데이터가 다시 샘플링됨
                샘플의 경계에 있는 복셀은 반복되므로 경계를 따라 얼룩진 줄무늬처럼 보임 [-1, 1] 범위와 같은 비율로 표시되는 최대 오프셋
        scale : 이미지의 크기를 확대 축소 하는 것, 경계 복셀이 반복되는 효과를 가짐 [-1, 1] 범위로 변환됨
        rotate : X축과 Y축만 회전함 
        noise : 샘플에 노이즈를 너무 많이 넣으면 실 데이터를 분류 못할 수 도 있음 
        """

        for i in range(3):
            if 'flip' in augmentation_dict:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if 'offset' in augmentation_dict:
                offset_float = augmentation_dict['offset']
                random_float = (random.random() * 2 - 1)
                transform_t[i, 3] = offset_float * random_float

            if 'scale' in augmentation_dict:
                scale_float = augmentation_dict['scale']
                random_float = (random.random() * 2 - 1)

        if 'rotate' in augmentation_dict:
            angle_red = random.random() * math.pi * 2
            s = math.sin(angle_red)
            c = math.cos(angle_red)

            rotation_t = torch.tensor([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

            transform_t @= rotation_t

        affine_t = F.affine_grid(
                transform_t[:3].unsqueeze(0).to(torch.float32),
                ct_t.size(),
                align_corners=False,
            ) # output Tensor of size(N X H X W X 2)

        
        augmented_chunk = F.grid_sample( 
                ct_t,
                affine_t,
                padding_mode = 'border',
                align_corners=False,
        ).to('cpu')

        if 'noise' in augmentation_dict:
            noise_t = torch.randn_like(augmented_chunk)
            noise_t *= augmentation_dict['noise']

            augmented_chunk += noise_t
        
        return augmented_chunk[0], center_irc



"""
간단한 데이터셋 구현
- Luna 데이터셋 구현
- 초기화 후에 하나의 상수값을 반환하는 __len__ 구현
- 인덱스를 인자로 받아 훈련(경우에 따라서는 검증)에서 사용할 샘플 데이터 튜플을 반환하는 __getitem__ 메소드
- __len__이 N값을 반환한다면 __getitem__은 0에서 N-1까지의 입력값에 대한 유효값을 넘겨줘야함
"""
class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str = 'random',
                 ratio_int=0 # ratio_int에 따라서 음성 샘플과 양성 샘플의 비율이 변함
            ):
        self.ratio_int = ratio_int
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool
        ]
        self.pos_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ]
        
        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self): # 매 에포크의 시작지점에서 이 메소드를 호출하여 샘플 순서를 랜덤하게 만듦
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)


    def __len__(self): # 데이터셋 길이 변경
        if self.ratio_int:
            return 200000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int: # ratio_int가 0이면 값이 고르게 분포된 것
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1): # 나머지가 0이 아니면 음성 샘플
                neg_ndx = ndx - 1 - pos_ndx # 데이터셋 인덱스에서 1을 뺀 후 
                neg_ndx %= len(self.negative_list) # 오버 플로되면 앞으로 돌아옴 실행
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx] # 오버 플로되면 앞으로 돌아옴 끝
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx] # 클래스 밸런싱이 아니면 N번째 샘플을 반환


        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )

        return (
            candidate_t, 
            pos_t, 
            candidateInfo_tup.series_uid, 
            torch.tensor(center_irc),
        )


# # 처음 tensor는 candidate_t
# # 두번째는 cls_t
# # 세번째는 candidate_tup.series_uid
# # 네번째는 center_irc
# print(LunaDataset()[0][0])
# print(LunaDataset()[1])