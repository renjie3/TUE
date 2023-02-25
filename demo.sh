TRAIN_STEP='2'

# MY_CMD="python3 -u ssl_perturbation.py --epochs 42 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --min_min_attack_fn eot_v1 --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --ssl_weight 0 --linear_noise_csd_weight 1e-25 --seed 1 --random_start --local 4 --no_save"

# MY_CMD="python3 -u ssl_perturbation.py --epochs 210 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 5 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --ssl_backbone moco --min_min_attack_fn eot_v1 --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --ssl_weight 1 --linear_noise_csd_weight 1 --moco_t 0.2 --asymmetric --SGD_optim --k_grad --seed 2 --local 4 --no_save"

MY_CMD="python3 -u ssl_perturbation.py --epochs 51 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --ssl_backbone simsiam --min_min_attack_fn eot_v1 --eot_size 1 --shuffle_train_perturb_data --ssl_weight 0 --pytorch_aug --linear_noise_csd_weight 0.1 --random_start --seed 1 --k_grad --local 4 --no_save"

echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
$MY_CMD