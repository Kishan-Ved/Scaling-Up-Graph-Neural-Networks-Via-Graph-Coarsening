#To overcome over-fitting, we fine-tuned the hyperparameter settings in the few-shot regime.
#To overcome over-smoothing, we fine-tuned the hyperparameter settings when the coarsening ratio is 0.1.
#example, the coarsening ratio is 0.5

# python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5
# python train.py --dataset cora --experiment few --epoch 100 --coarsening_ratio 0.5
# python train.py --dataset citeseer --experiment fixed --epoch 200 --coarsening_ratio 0.5
# python train.py --dataset pubmed --experiment fixed --epoch 200 --coarsening_ratio 0.5
# python train.py --dataset pubmed --experiment few --epoch 60 --coarsening_ratio 0.5
# python train.py --dataset dblp --experiment random --epoch 50 --coarsening_ratio 0.5
# python train.py --dataset Physics --experiment random --epoch 200 --lr 0.001 --weight_decay 0 --coarsening_ratio 0.5

# python train.py --dataset cora --epoch 60 --lr 0.01 --experiment fixed --coarsening_ratio 0.5 --num_layers 2

for i in $(seq 2 20);
do
    # python train.py --dataset cora --epoch 20 --coarsening_ratio 0.5 --lr 0.01 --num_layers $i
    python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5 --num_layers $i
done

# coarsening_ratio = 0.5 gives best accuracy for CORA according to the results in the paper