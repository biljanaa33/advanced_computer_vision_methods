import argparse
import os
import cv2

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
from siamfc import TrackerSiamFC

import numpy as np 
def evaluate_tracker(dataset_path, network_path, results_dir, visualize, longterm = False):
    
    sequences = []
    redetect_averages_all = []

    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    

    tracker = TrackerSiamFC(net_path=network_path)

    if longterm: 
        #tracker.lost_threshold = 2.0
        # evaluate for 2.0, - 9.0
        tracker.redetect_threshold = 4.0 # needs tunnning
        tracker.is_lost = False

    saving_redetect_example = True
    lost_frame_saved = False
    redetected_frame_saved = False

    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)

        # print(f"Number of frames for redetection needed: {}")

        """
        for i in range(1, sequence.length()):

            img = cv2.imread(sequence.frame(i))

            # short term update
            prediction, score = tracker.update(img)
            sample_centers = None

            if longterm and score < tracker.lost_threshold and tracker.is_lost: 
                prediction, score, sample_centers = tracker.re_detect(img)
            
            results.append(prediction)
            scores.append([score])

            
            if visualize and sample_centers is not None:
                for (cx, cy) in sample_centers:
                    cv2.circle(img, (cx, cy), radius=2, color=(0,255,0), thickness=-1)

            if visualize:
                tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)

                cv2.imshow('win', img)
                key_ = cv2.waitKey(10)
                if key_ == 27:
                    exit(0)
        """

        for i in range(1, sequence.length()):

            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img)

            sample_centers = None
            if longterm and score < tracker.redetect_threshold:
                prediction, score, sample_centers = tracker.re_detect(img)

            results.append(prediction)
            scores.append([score])


            if visualize:

                tl = (int(round(prediction[0])), int(round(prediction[1])))
                br = (int(round(prediction[0] + prediction[2])),
                            int(round(prediction[1] + prediction[3])))
                cv2.rectangle(img, tl, br, (0,0,255), 1)

               

                if longterm and saving_redetect_example:
                    if tracker.is_lost and not lost_frame_saved:

                        lost_frame_saved = True

                        target_w = prediction[2]
                        target_h = prediction[3]

                        #if sample_centers is not None:
                        #    for (cx, cy) in sample_centers:
                        #        tl = (int(round(cx - target_w / 2)), int(round(cy - target_h / 2)))
                        #        br = (int(round(cx + target_w / 2)), int(round(cy + target_h / 2)))
                        #        cv2.rectangle(img, tl, br, (255, 0, 0), 1)

                        if sample_centers is not None:
                            for (cx, cy) in sample_centers:
                                cv2.circle(img, (int(cx), int(cy)), radius=2, color=(255, 0, 0), thickness=-1)


                        cv2.imwrite('lost_before_car9.jpg', img)



                    elif not tracker.is_lost and lost_frame_saved and not redetected_frame_saved:

                        tl = (int(round(prediction[0])), int(round(prediction[1])))
                        br = (int(round(prediction[0] + prediction[2])),
                            int(round(prediction[1] + prediction[3])))
                        cv2.rectangle(img, tl, br, (0,255,0), 1)
                        cv2.imwrite('recovered_after_car9.jpg', img)
                        saving_redetect_example = False
                        redetected_frame_saved = True

                cv2.imshow('win', img)
                if cv2.waitKey(10) == 27:
                    exit(0)
            
        if longterm and len(tracker.redetect_frame_counts) > 0: 
            avg_redetect_frames = np.mean(tracker.redetect_frame_counts)
            redetect_averages_all.append(avg_redetect_frames)
            print(f"Avg frames to redetect in sequence {avg_redetect_frames:.4f}")

        save_results(results, bboxes_path)
        save_results(scores, scores_path)
    
    if longterm and len(redetect_averages_all) > 0:
        overall_avg = np.mean(redetect_averages_all)
        print(f"Avg frames to redetect: {overall_avg}")
        


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')
parser.add_argument("--longterm", help = "Enable long-term re-detection", required = False, action='store_true')
args = parser.parse_args()

evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize,args.longterm)

