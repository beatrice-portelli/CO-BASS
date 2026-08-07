# effect of alpha - resnet
python 04_run_MTL.py --model resnet --alpha 0.1;
python 04_run_MTL.py --model resnet --alpha 0.2;
python 04_run_MTL.py --model resnet --alpha 0.3;
python 04_run_MTL.py --model resnet --alpha 0.4;
python 04_run_MTL.py --model resnet --alpha 0.5;
python 04_run_MTL.py --model resnet --alpha 0.6;
python 04_run_MTL.py --model resnet --alpha 0.7;
# python 04_run_MTL.py --model resnet --alpha 0.8; # already executed with experiment_best_models.sh
python 04_run_MTL.py --model resnet --alpha 0.9;

# effect of alpha - convnext
python 04_run_MTL.py --model convnext --alpha 0.1;
python 04_run_MTL.py --model convnext --alpha 0.2;
python 04_run_MTL.py --model convnext --alpha 0.3;
python 04_run_MTL.py --model convnext --alpha 0.4;
python 04_run_MTL.py --model convnext --alpha 0.5;
python 04_run_MTL.py --model convnext --alpha 0.6;
python 04_run_MTL.py --model convnext --alpha 0.7;
python 04_run_MTL.py --model convnext --alpha 0.8;
# python 04_run_MTL.py --model convnext --alpha 0.9; # already executed with experiment_best_models.sh