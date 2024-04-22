CUDA_VISIBLE_DEVICES=0 python3 train.py \
	--algorithm drq \
	--seed $1 \
	--domain_name MetaWorld \
	--task_name soccer-v2 \
	--exp_name metaworld_soccer_drq_third_person \
	--num_train_steps 2000000 \
	--eval_freq 20000 \
	--save_freq 2000000 \
	--num_eval_episodes 20 \
	--action_repeat 2 \
	--cameras third_person \
	--frame_stack 3 \
	--feature_dim 100 \
	--use_proprioceptive_state True
