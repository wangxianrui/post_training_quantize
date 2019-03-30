python do_quant.py --batch_size 4 --num_samples 32
python do_freeze.py
python do_convert.py
python do_final_test.py
python do_eval.py > res_0432

python do_quant.py --batch_size 8 --num_samples 32
python do_freeze.py
python do_convert.py
python do_final_test.py
python do_eval.py > res_0832

python do_quant.py --batch_size 8 --num_samples 64
python do_freeze.py
python do_convert.py
python do_final_test.py
python do_eval.py > res_0864

python do_quant.py --batch_size 16 --num_samples 64
python do_freeze.py
python do_convert.py
python do_final_test.py
python do_eval.py > res_1664

python do_quant.py --batch_size 16 --num_samples 128
python do_freeze.py
python do_convert.py
python do_final_test.py
python do_eval.py > res_16128

python do_quant.py --batch_size 16 --num_samples 256
python do_freeze.py
python do_convert.py
python do_final_test.py
python do_eval.py > res_16256