####################################################################################################
# RSTN: Recurrent Saliency Transformation Network for organ segmentation framework                 #
# This is PyTorch 0.4.0 Python 3.6 verison of OrganSegRSTN in CAFFE Python 2.7 .                   #
# Author: Tianwei Ni, Huangjie Zheng, Lingxi Xie.                                                  #
#                                                                                                  #
# If you use our codes, please cite our paper accordingly:                                         #
#     Qihang Yu, Lingxi Xie, Yan Wang, Yuyin Zhou, Elliot K. Fishman, Alan L. Yuille,              #
#         "Recurrent Saliency Transformation Network:                                              #
#             Incorporating Multi-Stage Visual Cues for Small Organ Segmentation",                 #
#         in IEEE Conference on Computer Vision and Pattern Recognition, 2018.                     #
#                                                                                                  #
# NOTE: this program can be used for multi-organ segmentation.                                     #
#     Please also refer to its previous version, OrganSegC2F.                                      #
####################################################################################################

####################################################################################################
# variables for convenience
CURRENT_ORGAN_ID=1
CURRENT_PLANE=A
CURRENT_FOLD=0
CURRENT_GPU=$CURRENT_FOLD

####################################################################################################
# turn on these switches to execute each module
ENABLE_INITIALIZATION=0
ENABLE_TRAINING=0
ENABLE_COARSE_TESTING=0
ENABLE_COARSE_FUSION=0
ENABLE_COARSE2FINE_TESTING=0
# training settings: X|Y|Z
TRAINING_ORGAN_ID=$CURRENT_ORGAN_ID
TRAINING_PLANE=$CURRENT_PLANE
TRAINING_GPU=$CURRENT_GPU
# coarse_testing settings: X|Y|Z, before this, coarse-scaled models shall be ready
COARSE_TESTING_ORGAN_ID=$CURRENT_ORGAN_ID
COARSE_TESTING_PLANE=$CURRENT_PLANE
COARSE_TESTING_GPU=$CURRENT_GPU
# coarse_fusion settings: before this, coarse-scaled results on 3 views shall be ready
COARSE_FUSION_ORGAN_ID=$CURRENT_ORGAN_ID
# fine_testing settings: before this, both coarse-scaled and fine-scaled models shall be ready
COARSE2FINE_TESTING_ORGAN_ID=$CURRENT_ORGAN_ID
COARSE2FINE_TESTING_GPU=$CURRENT_GPU

####################################################################################################
# defining the root path which stores image and label data
DATA_PATH='/media/5T2/Datasets/NIH/'

####################################################################################################
# data initialization: only needs to be run once
# variables
ORGAN_NUMBER=1
FOLDS=4
LOW_RANGE=-100
HIGH_RANGE=240
# init.py : data_path, organ_number, folds, low_range, high_range
if [ "$ENABLE_INITIALIZATION" = "1" ]
then
	python init.py \
		$DATA_PATH $ORGAN_NUMBER $FOLDS $LOW_RANGE $HIGH_RANGE
fi

####################################################################################################
# the individual and joint training processes
# variables
SLICE_THRESHOLD=0.98
SLICE_THICKNESS=3
LEARNING_RATE1=1e-5
LEARNING_RATE2=1e-5
LEARNING_RATE_M1=10
LEARNING_RATE_M2=10
TRAINING_MARGIN=20
TRAINING_PROB=0.5
TRAINING_SAMPLE_BATCH=1
TRAINING_EPOCH_S=2
TRAINING_EPOCH_I=6
TRAINING_EPOCH_J=8
LR_DECAY_EPOCH_J_STEP=2
if [ "$ENABLE_TRAINING" = "1" ]
then
	TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
else
	TRAINING_TIMESTAMP=_
fi
# training.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate2 (not used), margin, prob, sample_batch,
#     step, max_iterations1, max_iterations2 (not used), fraction, timestamp
if [ "$ENABLE_TRAINING" = "1" ]
then
	if [ "$TRAINING_PLANE" = "X" ] || [ "$TRAINING_PLANE" = "A" ]
	then
		TRAINING_MODELNAME=X${SLICE_THICKNESS}_${TRAINING_ORGAN_ID}
		TRAINING_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${TRAINING_MODELNAME}_${TRAINING_TIMESTAMP}.txt
		python training.py \
			$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
			$SLICE_THRESHOLD $SLICE_THICKNESS \
			$TRAINING_ORGAN_ID X $TRAINING_GPU \
			$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
			$TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
			$TRAINING_EPOCH_S $TRAINING_EPOCH_I $TRAINING_EPOCH_J \
			$LR_DECAY_EPOCH_J_STEP $TRAINING_TIMESTAMP 1 2>&1 | tee $TRAINING_LOG
	fi
	if [ "$TRAINING_PLANE" = "Y" ] || [ "$TRAINING_PLANE" = "A" ]
	then
		TRAINING_MODELNAME=Y${SLICE_THICKNESS}_${TRAINING_ORGAN_ID}
		TRAINING_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${TRAINING_MODELNAME}_${TRAINING_TIMESTAMP}.txt
		python training.py \
			$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
			$SLICE_THRESHOLD $SLICE_THICKNESS \
			$TRAINING_ORGAN_ID Y $TRAINING_GPU \
			$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
			$TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
			$TRAINING_EPOCH_S $TRAINING_EPOCH_I $TRAINING_EPOCH_J \
			$LR_DECAY_EPOCH_J_STEP $TRAINING_TIMESTAMP 1 2>&1 | tee $TRAINING_LOG
	fi
	if [ "$TRAINING_PLANE" = "Z" ] || [ "$TRAINING_PLANE" = "A" ]
	then
		TRAINING_MODELNAME=Z${SLICE_THICKNESS}_${TRAINING_ORGAN_ID}
		TRAINING_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${TRAINING_MODELNAME}_${TRAINING_TIMESTAMP}.txt
		python training.py \
			$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
			$SLICE_THRESHOLD $SLICE_THICKNESS \
			$TRAINING_ORGAN_ID Z $TRAINING_GPU \
			$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
			$TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
			$TRAINING_EPOCH_S $TRAINING_EPOCH_I $TRAINING_EPOCH_J \
			$LR_DECAY_EPOCH_J_STEP $TRAINING_TIMESTAMP 1 2>&1 | tee $TRAINING_LOG
	fi
fi

####################################################################################################
# the coarse-scaled testing processes
# variables
COARSE_TESTING_EPOCH_S=$TRAINING_EPOCH_S
COARSE_TESTING_EPOCH_I=$TRAINING_EPOCH_I
COARSE_TESTING_EPOCH_J=$TRAINING_EPOCH_J
COARSE_TESTING_EPOCH_STEP=$LR_DECAY_EPOCH_J_STEP
COARSE_TIMESTAMP1=$TRAINING_TIMESTAMP
COARSE_TIMESTAMP2=$TRAINING_TIMESTAMP
# coarse_testing.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate2, margin, prob, sample_batch,
#     step, max_iterations1, max_iterations2,
#     starting_iterations, step, max_iterations,
#     timestamp1, timestamp2 (optional)
if [ "$ENABLE_COARSE_TESTING" = "1" ]
then
	if [ "$COARSE_TESTING_PLANE" = "X" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
	then
		python coarse_testing.py \
			$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
			$SLICE_THRESHOLD $SLICE_THICKNESS \
			$COARSE_TESTING_ORGAN_ID X $COARSE_TESTING_GPU \
			$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
			$TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
			$COARSE_TESTING_EPOCH_S $COARSE_TESTING_EPOCH_I \
			$COARSE_TESTING_EPOCH_J $COARSE_TESTING_EPOCH_STEP \
			$COARSE_TIMESTAMP1 $COARSE_TIMESTAMP2
	fi
	if [ "$COARSE_TESTING_PLANE" = "Y" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
	then
		python coarse_testing.py \
			$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
			$SLICE_THRESHOLD $SLICE_THICKNESS \
			$COARSE_TESTING_ORGAN_ID Y $COARSE_TESTING_GPU \
			$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
			$TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
			$COARSE_TESTING_EPOCH_S $COARSE_TESTING_EPOCH_I \
			$COARSE_TESTING_EPOCH_J $COARSE_TESTING_EPOCH_STEP \
			$COARSE_TIMESTAMP1 $COARSE_TIMESTAMP2
	fi
	if [ "$COARSE_TESTING_PLANE" = "Z" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
	then
		python coarse_testing.py \
			$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
			$SLICE_THRESHOLD $SLICE_THICKNESS \
			$COARSE_TESTING_ORGAN_ID Z $COARSE_TESTING_GPU \
			$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
			$TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
			$COARSE_TESTING_EPOCH_S $COARSE_TESTING_EPOCH_I \
			$COARSE_TESTING_EPOCH_J $COARSE_TESTING_EPOCH_STEPvq
			$COARSE_TIMESTAMP1 $COARSE_TIMESTAMP2
	fi
fi

####################################################################################################
# the coarse-scaled fusion process
# variables
COARSE_FUSION_EPOCH_S=$TRAINING_EPOCH_S
COARSE_FUSION_EPOCH_I=$TRAINING_EPOCH_I
COARSE_FUSION_EPOCH_J=$TRAINING_EPOCH_J
COARSE_FUSION_EPOCH_STEP=$LR_DECAY_EPOCH_J_STEP
COARSE_FUSION_THRESHOLD=0.5
COARSE_TIMESTAMP1_X=$TRAINING_TIMESTAMP
COARSE_TIMESTAMP1_Y=$TRAINING_TIMESTAMP
COARSE_TIMESTAMP1_Z=$TRAINING_TIMESTAMP
COARSE_TIMESTAMP2_X=$TRAINING_TIMESTAMP
COARSE_TIMESTAMP2_Y=$TRAINING_TIMESTAMP
COARSE_TIMESTAMP2_Z=$TRAINING_TIMESTAMP
# coarse_fusion.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate_m1, learning_rate2, learning_rate_m2, margin,
#     starting_iterations, step, max_iterations, threshold,
#     timestamp1_X, timestamp1_Y, timestamp1_Z,
#     timestamp2_X (optional), timestamp2_Y (optional), timestamp2_Z (optional)
if [ "$ENABLE_COARSE_FUSION" = "1" ]
then
	python coarse_fusion.py \
		$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
		$SLICE_THRESHOLD $SLICE_THICKNESS $COARSE_TESTING_ORGAN_ID $COARSE_TESTING_GPU \
		$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 $TRAINING_MARGIN \
		$COARSE_FUSION_EPOCH_S $COARSE_FUSION_EPOCH_I $COARSE_FUSION_EPOCH_J \
		$COARSE_FUSION_EPOCH_STEP $COARSE_FUSION_THRESHOLD \
		$COARSE_TIMESTAMP1_X $COARSE_TIMESTAMP1_Y $COARSE_TIMESTAMP1_Z \
		$COARSE_TIMESTAMP2_X $COARSE_TIMESTAMP2_Y $COARSE_TIMESTAMP2_Z
fi

####################################################################################################
# the coarse-to-fine testing process
# variables
FINE_TESTING_EPOCH_S=$TRAINING_EPOCH_S
FINE_TESTING_EPOCH_I=$TRAINING_EPOCH_I
FINE_TESTING_EPOCH_J=$TRAINING_EPOCH_J
FINE_TESTING_EPOCH_STEP=$LR_DECAY_EPOCH_J_STEP
FINE_FUSION_THRESHOLD=0.5
COARSE2FINE_TIMESTAMP1_X=$TRAINING_TIMESTAMP
COARSE2FINE_TIMESTAMP1_Y=$TRAINING_TIMESTAMP
COARSE2FINE_TIMESTAMP1_Z=$TRAINING_TIMESTAMP
COARSE2FINE_TIMESTAMP2_X=$TRAINING_TIMESTAMP
COARSE2FINE_TIMESTAMP2_Y=$TRAINING_TIMESTAMP
COARSE2FINE_TIMESTAMP2_Z=$TRAINING_TIMESTAMP
MAX_ROUNDS=10
# coarse2fine_testing.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, GPU_ID,
#     learning_rate1, learning_rate_m1, learning_rate2, learning_rate_m2, margin,
#     coarse_fusion_starting_iterations, coarse_fusion_step, coarse_fusion_max_iterations,
#     coarse_fusion_threshold, coarse_fusion_code,
#     fine_starting_iterations, fine_step, fine_max_iterations,
#     fine_fusion_threshold, max_rounds,
#     timestamp1_X, timestamp1_Y, timestamp1_Z,
#     timestamp2_X (optional), timestamp2_Y (optional), timestamp2_Z (optional)
if [ "$ENABLE_COARSE2FINE_TESTING" = "1" ]
then
	python coarse2fine_testing.py \
		$DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
		$SLICE_THRESHOLD $SLICE_THICKNESS $COARSE2FINE_TESTING_ORGAN_ID $COARSE2FINE_TESTING_GPU \
		$LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 $TRAINING_MARGIN \
		$FINE_TESTING_EPOCH_S $FINE_TESTING_EPOCH_I $FINE_TESTING_EPOCH_J $FINE_TESTING_EPOCH_STEP \
		$COARSE_FUSION_THRESHOLD $FINE_FUSION_THRESHOLD $MAX_ROUNDS \
		$COARSE2FINE_TIMESTAMP1_X $COARSE2FINE_TIMESTAMP1_Y $COARSE2FINE_TIMESTAMP1_Z \
		$COARSE2FINE_TIMESTAMP2_X $COARSE2FINE_TIMESTAMP2_Y $COARSE2FINE_TIMESTAMP2_Z
fi

####################################################################################################
