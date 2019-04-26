python do_quant.py --batch_size 1 --num_samples 1
python do_predict.py
python do_eval.py > test_result/res_0432_ckpt
python do_freeze.py
python do_convert.py
python do_predict_lite.py
python do_eval.py > test_result/res_0432_lite
exit

python do_quant.py --batch_size 8 --num_samples 32
python do_predict.py
python do_eval.py > test_result/res_0832_ckpt
python do_freeze.py
python do_convert.py
python do_predict_lite.py
python do_eval.py > test_result/res_0832_lite

python do_quant.py --batch_size 8 --num_samples 64
python do_predict.py
python do_eval.py > test_result/res_0864_ckpt
python do_freeze.py
python do_convert.py
python do_predict_lite.py
python do_eval.py > test_result/res_0864_lite

python do_quant.py --batch_size 16 --num_samples 64
python do_predict.py
python do_eval.py > test_result/res_1664_ckpt
python do_freeze.py
python do_convert.py
python do_predict_lite.py
python do_eval.py > test_result/res_1664_lite

python do_quant.py --batch_size 16 --num_samples 128
python do_predict.py
python do_eval.py > test_result/res_16128_ckpt
python do_freeze.py
python do_convert.py
python do_predict_lite.py
python do_eval.py > test_result/res_16128_lite

python do_quant.py --batch_size 16 --num_samples 256
python do_predict.py
python do_eval.py > test_result/res_16256_ckpt
python do_freeze.py
python do_convert.py
python do_predict_lite.py
python do_eval.py > test_result/res_16256_lite
