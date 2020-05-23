import mxnet as mx
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.bbox import bbox_iou

try:
    from mxnet.metric import EvalMetric
except:
    from mxnet.gluon.metric import EvalMetric


class COCOMApMetric(EvalMetric):
    def __init__(self, ovp_thresh=0.5, use_difficult=False, class_names=None, pred_idx=0, roc_output_path=None,
                 tensortboard_path=None, score_thresh=0.02, use_voc07=False):
        super(COCOMApMetric, self).__init__('coco_mAP')
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), 'must provide names as str'
            num = len(class_names)
            self.name = class_names + ['mAP']
            self.num = num + 1

        self.reset()
        if isinstance(ovp_thresh, float):
            ovp_thresh = [ovp_thresh]
        self.ovp_thresh = ovp_thresh
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)
        self.roc_output_path = roc_output_path
        self.tensorboard_path = tensortboard_path
        self.score_thresh = score_thresh
        self.use_voc07 = use_voc07
        print(self.ovp_thresh)

    def set_iou(self, new_iou=0.5):
        self.ovp_thresh = [new_iou]

    def save_roc_graph(self, recall=None, prec=None, classkey=1, iou_str='0.5', path=None, ap=None):
        if not os.path.exists(path):
            os.mkdir(path)
        plot_path = os.path.join(path, 'roc_'+ self.class_names[classkey] + iou_str + '.png')
        if os.path.exists(plot_path):
            os.remove(plot_path)
        fig = plt.figure()
        plt.title(self.class_names[classkey])
        plt.plot(recall, prec, 'b', label='AP=%0.2f' % ap)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(plot_path)
        plt.close(fig)

    def reset(self):
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [None] * self.num
        self._n_pos = defaultdict(int)  # map class to the number of that class objects
        self._score = defaultdict(list) # map class to the scores of that class bbox
        # self._match = defaultdict(defaultdict(list)) # map iou thresh to the precision & recall of each class
        self._match = defaultdict(lambda : defaultdict(list))

    def get(self):
        self._update()
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            # print(np.array(self.sum_metric[0]).shape)   # should be 10
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [np.array(x) / y if y != 0 else float('nan') for x, y in zip(self.sum_metric, self.num_inst)]
        return (names, values)

    def update(self, pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None):
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                try:
                    out = np.concatenate(out, axis=0)
                except ValueError:
                    out = np.array(out)
                return out
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        if gt_difficults is None:
            gt_difficults = [None for _ in as_numpy(gt_labels)]
        '''
        if isinstance(gt_labels, list):
            if len(gt_difficults) != len(gt_labels) * gt_labels[0].shape[0]:
                gt_difficults = [None] * len(gt_labels) * gt_labels[0].shape[0]
        '''

        # calculate the FD, suppose the predicted label is all correct
        # obj_nums = gt_bboxes[0].shape[1]
        # if obj_nums < 101:
        #     gt_labels = [X[:, 0:obj_nums, :] for X in pred_labels]
        # print('len of gt_labels: ', len(gt_labels))
        # print('len of pred_labels: ', len(pred_labels))
        # print('shape of gt_labels ele: ', gt_labels[0].shape)
        # print('shape of pred_labels ele: ', pred_labels[0].shape)

        for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores,
                                        gt_bboxes, gt_labels, gt_difficults]]):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]
            valid_gt = np.where(gt_label.flat >= 0)[0]
            gt_bbox = gt_bbox[valid_gt, :]
            gt_label = gt_label.flat[valid_gt].astype(int)
            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]
                # get the predicts, which socre > scores thresh
                pred_score_l_idx = np.where(pred_score_l >= self.score_thresh)[0]
                pred_bbox_l = pred_bbox_l[pred_score_l_idx]
                pred_score_l = pred_score_l[pred_score_l_idx]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                self._n_pos[l] += np.logical_not(gt_difficult_l).sum()
                self._score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1
                iou = bbox_iou(pred_bbox_l, gt_bbox_l)
                for iou_thresh in self.ovp_thresh:
                    iou_thresh_key = str(iou_thresh)
                    if len(gt_bbox_l) == 0:
                        self._match[iou_thresh_key][l].extend((0,) * pred_bbox_l.shape[0])
                        continue

                    # VOC evaluation follows integer typed bounding boxes.

                    gt_index = iou.argmax(axis=1)
                    # set -1 if there is no matching ground truth
                    gt_index[iou.max(axis=1) < iou_thresh] = -1

                    selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                    for gt_idx in gt_index:
                        if gt_idx >= 0:
                            if gt_difficult_l[gt_idx]:
                                self._match[iou_thresh_key][l].append(-1)
                            else:
                                if not selec[gt_idx]:
                                    self._match[iou_thresh_key][l].append(1)
                                else:
                                    self._match[iou_thresh_key][l].append(0)
                            selec[gt_idx] = True
                        else:
                            self._match[iou_thresh_key][l].append(0)

    def _update(self):
        aps = []
        recall, precs = self._recall_prec()
        for l, rec, prec in zip(range(len(precs)), recall, precs):
            ap = []
            for (kr, vr), (kp, vp) in zip(rec.items(), prec.items()):
                assert kr == kp, 'iou thresh did not match.'
                if self.use_voc07:
                    curr_ap = self._average_precision_07(vr, vp)
                else:
                    curr_ap = self._average_precision(vr, vp)
                if self.roc_output_path is not None:
                    self.save_roc_graph(vr, vp, l, kr, path=self.roc_output_path, ap=curr_ap)
                ap.append(curr_ap)
            aps.append(ap)
            if self.num is not None and l < (self.num - 1):
                self.sum_metric[l] = ap
                self.num_inst[l] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)

    def _recall_prec(self):
        n_fg_class = max(self._n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for l in self._n_pos.keys():
            prec_dict = dict()
            recall_dict = dict()
            for iou_thresh in self.ovp_thresh:
                iou_thresh = str(iou_thresh)

                score_l = np.array(self._score[l])
                match_l = np.array(self._match[iou_thresh][l], dtype=np.int32)

                order = score_l.argsort()[::-1]
                match_l = match_l[order]

                tp = np.cumsum(match_l == 1)
                fp = np.cumsum(match_l == 0)

                # If an element of fp + tp is 0,
                # the corresponding element of prec[l] is nan.
                with np.errstate(divide='ignore', invalid='ignore'):
                     prec_dict[iou_thresh] = tp / (fp + tp)
                    # prec_dict[iou_thresh] = tp / (fp + tp + np.spacing(1))
                # If n_pos[l] is 0, rec[l] is None.
                if self._n_pos[l] > 0:
                    recall_dict[iou_thresh] = tp / self._n_pos[l]
                else:
                    recall_dict = None
            prec[l] = prec_dict
            rec[l] = recall_dict

        return rec, prec

    def _average_precision(self, rec, prec):
        ap = 0
        for t in np.linspace(.0, 1.0, np.round((1.00 - 0.0) / .01) + 1, endpoint=True):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / (np.round((1.00 - 0.0) / .01) + 1)

        return ap

    def _average_precision_07(self, rec, prec):
        ap = 0.0
        for t in np.linspace(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.

        return ap
