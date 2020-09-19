#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

    
class Simulator:
    def __init__(self):
        self.sample_submission = pd.read_csv("data/sample_submission.csv")
        self.sample_submission['time'] = pd.to_datetime(self.sample_submission['time'])

        self.max_count = pd.read_csv("data/max_count.csv")
        self.max_count['date'] = pd.to_datetime(self.max_count['date'])
        
        self.stock = pd.read_csv("data/stock.csv")
        
        self.cut_yield = pd.read_csv("data/cut_yield.csv")
        self.cut_yield['date'] = pd.to_datetime({'year':self.cut_yield['date'] // 100, 'month':self.cut_yield['date'] % 100, 'day':1})
        self.cut_yield[['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']] /= 100
        
        self.change_time = pd.read_csv("data/change_time.csv")
        
        self.order = pd.read_csv("data/order.csv")
        self.order['time'] = pd.to_datetime(self.order['time'])
        self.order_arr = self.order[['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']].values
        
        self.ini_stock_MOL = np.repeat(self.stock[['MOL_1', 'MOL_2', 'MOL_3', 'MOL_4']].values, self.sample_submission.shape[0], axis=0)
        self.ini_stock_BLK = np.repeat(self.stock[['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']].values, self.order.shape[0], axis=0)
        
    
    def BLK_to_MOL(self, BLK, item, date_idx):
        # 과부족한 BLK의 양을 input으로 받아 MOL_output을 return 하는 함수
        # date_idx : delivery가 발생한 시점
        pieces = {'1':506, '2':506, '3':400, '4':400}
        month_idx = self.order['time'].iloc[date_idx].month - 4
        month_cut_yield = self.cut_yield.iloc[month_idx, item]
        MOL_output = BLK / pieces[str(item)] / month_cut_yield
        return MOL_output


    def MOL_to_BLK(self, MOL_output, item, date_idx):
        # MOL_output을 받아 생산된 BLK를 return 하는 함수
        # date_idx : delivery가 발생한 시점 (order의 idx)
        pieces = {'1':506, '2':506, '3':400, '4':400}
        month_idx = self.order['time'].iloc[date_idx].month - 4  # 4월이 0, 5월이 1, 6월이 2
        month_cut_yield = self.cut_yield.iloc[month_idx, item]
        BLK = int(MOL_output * pieces[str(item)] * month_cut_yield)
        return BLK
    
    
    def F(self, x, a):
            if x < a:
                return (1 - x/a)
            else:
                return 0
    
    
    def cal_score(self, over_prod):
        # over_prod : order가 이루어지는 시점에서 "block 재고 - order"
        # over_prod가 양수이면 생산 초과, 음수면 생산 부족.
        Q = over_prod[over_prod > 0].sum()
        P = -over_prod[over_prod < 0].sum()
        N = self.order_arr.sum()
        fpn, fqn = self.F(P, 10*N), self.F(Q, 10*N)
        score = 50 * fpn + 20 * fqn
        return score
    
    
    def initialize_over_prod(self):
        # submission의 mol_input이 0인 상태에서 초기 재고와 order를 반영해 over_prod_ini를 생성
        stock_MOL = self.ini_stock_MOL.copy()
        stock_BLK = self.ini_stock_BLK.copy()
        stock_MOL_18h = stock_MOL[(np.arange(0, stock_MOL.shape[0], 24) + 18), :]
        
        over_prod_ini = np.zeros_like(self.order_arr, dtype=np.int32)        
        ordered_days = np.where((self.order_arr > 0).any(axis=1))[0]
        for date_idx in ordered_days:
            ordered_items = np.where(self.order_arr[date_idx] > 0)[0] + 1
            for item in ordered_items:
                order_unit = self.order_arr[date_idx, item-1]
                MOL_unit = stock_MOL_18h[date_idx, item-1]  # stock_MOL_18h 모두 자름
                stock_MOL_18h[date_idx:, item-1] -= MOL_unit
                stock_BLK[date_idx:, item-1] += self.MOL_to_BLK(MOL_unit, item, date_idx)
                BLK_unit = stock_BLK[date_idx, item-1]
                over_prod_ini[date_idx, item-1] = BLK_unit - order_unit
                stock_BLK[date_idx:, item-1] -= order_unit
                
        score = self.cal_score(over_prod_ini)
        return over_prod_ini, score
    
    
    def optimizer(self, submission):
        # submission의 MOL_A, MOL_B를 update (optimization)
        over_prod, score = self.initialize_over_prod()
        
        new_mol = np.zeros((submission.shape[0], 2), dtype=np.float32)
        for item in [1, 2, 3, 4]:
            ### make inout_mapping ###
            # inout_mapping : 각 item에 대한 input 시각 index와 output 시각 index를 매칭시킨 array
            inout_mapping = np.zeros((0, 2), dtype=np.uint8)
            inout_lines = np.zeros((0, 1), dtype=np.uint8)  # inout_mapping에 대응하는 line -> A:0 b:1
            for line_i, line in enumerate(['A', 'B']):
                state = submission['Event_'+line].str[-1].replace('S', np.nan)\
                        .replace('P', np.nan).fillna(method='ffill').values.astype(np.uint8)  # state : input 시점의 item을 저장
                item_idx = submission.loc[state==item].index
                process_idx = submission.loc[(submission['Event_'+line]=='PROCESS')].index
                
                in_time_idx = item_idx & process_idx[:-48]  # 특정 item의 input 시각의 index들
                if in_time_idx.shape[0] == 0:
                    continue
                temp_list = [np.where(process_idx==idx)[0][0] for idx in in_time_idx]
                out_time_idx = [process_idx[i + 47] + 1 for i in temp_list]  # output 시각의 index들
                temp_arr = np.append(np.array(in_time_idx)[:, np.newaxis], np.array(out_time_idx)[:, np.newaxis], axis=1)
                inout_mapping = np.append(inout_mapping, temp_arr, axis=0)
                
                line_idx = np.ones((in_time_idx.shape[0], 1)) * line_i
                inout_lines = np.append(inout_lines, line_idx, axis=0)
            if inout_mapping.shape[0] == 0:
                continue

            ### update MOL input ###
            # 각 item의 일별 order에 대해 inout_mapping에서 적절한 time을 찾아 부족분만큼 MOL input을 update
            valid_dates = np.where(self.order_arr[:, item-1] > 0)[0]  # delivery가 이루어질 idx (from order)
            first_out_idx = inout_mapping[:, 1].min() // 24  # 가장 빠른 output의 시간 
            for date_idx in valid_dates:
                # score가 감소하는 경우에는 update를 하지 않기 위한 copy
                temp_new_mol = new_mol.copy()
                temp_over_prod = over_prod.copy()
                for out_date_idx in range(date_idx, first_out_idx-1, -1):  # date_idx에서 하루씩 앞으로 이동하면서 mol_input을 증가시킴.
                    # 부족분 (필요 생산량, 단위 BLK)
                    shortage = -temp_over_prod[date_idx, item-1]
                    if shortage <= 0:  # 생산 초과(shortage < 0)는 고려하지 않음.
                        break
                    shortage += 5  # cut 과정에서 소수점이 잘려나가기 때문에 수요가 실제 부족분보다 5만큼 많다고 생각함.
                    
                    # mol input을 조정할 target index : output 시각이 time_from_와 time_to_ 사이인 input 시각의 index
                    time_from_ = (out_date_idx - 1) * 24 + 18
                    time_to_ = out_date_idx * 24 + 18
                    input_bool = (inout_mapping[:, 1] > time_from_) & (inout_mapping[:, 1] <= time_to_)
                    update_idx = inout_mapping[input_bool, 0]
                    update_lines = inout_lines[input_bool, 0]
                    if update_idx.shape[0] == 0:
                        continue
                    line_A_target_idx = update_idx[update_lines==0]
                    line_B_target_idx = update_idx[update_lines==1]
                    
                    # total_idle : 생산 여유 (부동 소수점 문제 때문에 기준을 6.666999로 함)
                    line_A_idle = 6.666999 - temp_new_mol[line_A_target_idx, 0]
                    line_B_idle = 6.666999 - temp_new_mol[line_B_target_idx, 1]
                    total_idle = line_A_idle.sum() + line_B_idle.sum()
                    # total_delta : 증가시켜야 할 MOL input의 총량
                    total_delta = self.BLK_to_MOL(shortage, item, out_date_idx)
                    total_delta /= 0.975  # MOL output -> MOL input
                    # total_delta가 total_idle보다 많은 경우엔 시간당 최대 생산량(6.666999)으로 맞춰줌
                    if total_delta >= total_idle:
                        temp_new_mol[line_A_target_idx, 0] = 6.666999
                        temp_new_mol[line_B_target_idx, 1] = 6.666999
                        total_delta = total_idle
                    # 그렇지 않은 경우엔 시간대별 각 line의 생산 여유의 비율대로 total_delta를 나눠가짐
                    else:
                        delta = total_delta / total_idle
                        delta_A = line_A_idle * delta
                        delta_B = line_B_idle * delta
                        temp_new_mol[line_A_target_idx, 0] += delta_A
                        temp_new_mol[line_B_target_idx, 1] += delta_B
                    # total_delta 만큼의 input 증가에 따른 temp_over_prod update (주의 : order > 0인 경우만 update)
                    total_delta_BLK = self.MOL_to_BLK(total_delta * 0.975, item, out_date_idx)
                    temp_over_prod[out_date_idx:, item-1][(self.order_arr[out_date_idx:, item-1] > 0)] += total_delta_BLK
                    
                    # max_count를 넘었을 경우 해당 기간에 전체적으로 감산
                    for line_i, indices in enumerate([line_A_target_idx, line_B_target_idx]):
                        indices_date = indices // 24
                        for date_idx2 in np.unique(indices_date):
                            max_input = self.max_count.iloc[date_idx2, 1] 
                            mol_input = temp_new_mol[date_idx2*24:(date_idx2+1)*24, line_i].sum()
                            input_cut = mol_input - max_input
                            if input_cut > 0:  # 일일 최대 투입량을 넘은 경우
                                temp_idx = indices[indices_date==date_idx2]
                                if max_input == 0:
                                    temp_new_mol[temp_idx, line_i] = 0
                                    input_cut = mol_input
                                else:
                                    input_cut += 0.001  # 부동소수점 때문에 0점이 나오는 경우를 방지
                                    cut_portion = input_cut / temp_idx.shape[0]
                                    temp_new_mol[temp_idx, line_i] -= cut_portion
                                # 감산에 대한 temp_over_prod update.
                                input_cut_BLK = self.MOL_to_BLK(input_cut * 0.975, item, out_date_idx)
                                temp_over_prod[out_date_idx:, item-1][(self.order_arr[out_date_idx:, item-1] > 0)] -= input_cut_BLK
                                
                    # score 계산 & update, score가 감소할 때에는 update 하지 않고 break.
                    temp_score = self.cal_score(temp_over_prod)
                    if temp_score >= score:
                        score = temp_score
                        new_mol = temp_new_mol.copy()
                        over_prod = temp_over_prod.copy()
                    else:
                        break
        
        new_mol[new_mol < 0] = 0  # 부동 소수점으로 인해 아주 작은 음수 값이 생긴 경우에 대한 처리
        submission['MOL_A'] += new_mol[:, 0]
        submission['MOL_B'] += new_mol[:, 1]
        return submission, over_prod, score
        

    def get_score(self, submission):
        submission, over_prod, score = self.optimizer(submission)
        return submission, over_prod, score