# ZS - symptom
python 02_run_ZS.py --target sym --model "vit.base.32";
python 02_run_ZS.py --target sym --model "vit.h.laion2b";
python 02_run_ZS.py --target sym --model "bioclip";
python 02_run_ZS.py --target sym --model "bioclip2";
# ZS - disease
python 02_run_ZS.py --target dis --model "vit.base.32";
python 02_run_ZS.py --target dis --model "vit.h.laion2b";
python 02_run_ZS.py --target dis --model "bioclip";
python 02_run_ZS.py --target dis --model "bioclip2";
# VISION - symptom
python 03_run_VISION.py --target sym --model resnet;
python 03_run_VISION.py --target sym --model mobilenet;
python 03_run_VISION.py --target sym --model efficientnet;
python 03_run_VISION.py --target sym --model convnext;
python 03_run_VISION.py --target sym --model swint;
# VISION - disease
python 03_run_VISION.py --target dis --model resnet;
python 03_run_VISION.py --target dis --model mobilenet;
python 03_run_VISION.py --target dis --model efficientnet;
python 03_run_VISION.py --target dis --model convnext;
python 03_run_VISION.py --target dis --model swint;
# MTL
python 04_run_MTL.py --model resnet       --alpha 0.8;
python 04_run_MTL.py --model mobilenet    --alpha 0.6;
python 04_run_MTL.py --model efficientnet --alpha 0.5;
python 04_run_MTL.py --model convnext     --alpha 0.9;
python 04_run_MTL.py --model swint        --alpha 0.6;
# S2D-MTL
python 05_run_S2D-MTL.py --model resnet       --alpha 0.7 --beta 0.5;
python 05_run_S2D-MTL.py --model mobilenet    --alpha 0.6 --beta 0.8;
python 05_run_S2D-MTL.py --model efficientnet --alpha 0.3 --beta 0.5;
python 05_run_S2D-MTL.py --model convnext     --alpha 0.6 --beta 0.4;
python 05_run_S2D-MTL.py --model swint        --alpha 0.4 --beta 0.7;
# S2D-MTL-Matrix
python 05_run_S2D-MTL.py --model resnet       --alpha 0.6 --beta 0.5 --gamma 0.5;
python 05_run_S2D-MTL.py --model mobilenet    --alpha 0.6 --beta 0.5 --gamma 0.5;
python 05_run_S2D-MTL.py --model efficientnet --alpha 0.4 --beta 0.7 --gamma 0.5;
python 05_run_S2D-MTL.py --model convnext     --alpha 0.4 --beta 0.5 --gamma 0.5;
python 05_run_S2D-MTL.py --model swint        --alpha 0.4 --beta 0.5 --gamma 0.5;