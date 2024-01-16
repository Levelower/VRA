# 主实验
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --vision_constraint > log/new_wmt19.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --device 0 --vision_constraint > log/new_wmt18.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --device 1 --vision_constraint > log/new_ted.log 2>&1 &
nohup python get_adversarial_examples.py --src ja --tgt en --dataset aspec --device 7 --sc glyph --vision_constraint > log/new_aspec.log 2>&1 &

# 消融实验（超参消融）
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --percent 1 --vision_constraint > log/percent_1.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --percent 2 --vision_constraint > log/percent_2.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --percent 0.1 --vision_constraint > log/percent_0.1.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --percent 0.25 --vision_constraint > log/percent_0.25.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --percent 0.3 --vision_constraint > log/percent_0.3.log 2>&1 &

nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --thresh 0.90 --vision_constraint > log/thresh_0.90.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --thresh 0.91 --vision_constraint > log/thresh_0.91.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --thresh 0.92 --vision_constraint > log/thresh_0.92.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --thresh 0.93 --vision_constraint > log/thresh_0.93.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --thresh 0.94 --vision_constraint > log/thresh_0.94.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --thresh 0.96 --vision_constraint > log/thresh_0.96.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --thresh 0.97 --vision_constraint > log/thresh_0.97.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --thresh 0.98 --vision_constraint > log/thresh_0.98.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --thresh 0.99 --vision_constraint > log/thresh_0.99.log 2>&1 &

# 消融实验（方法消融）
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --sc glyph --vision_constraint > log/sc_glyph.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --sc radicals --vision_constraint > log/sc_radicals.log 2>&1 &

# 消融实验（组件消融）
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --search_method semantics > log/semantics_F.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 1 --search_method semantics --vision_constraint > log/semantics_T.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --device 0 --search_method vision > log/vision_F.log 2>&1 &


# merge方法测试
nohup python get_adversarial_examples.py --dataset wmt19 --merge 1 --device 0 --vision_constraint > log/wmt19_merge_1.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --merge 2 --device 1 --vision_constraint > log/wmt19_merge_2.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --merge 3 --device 0 --vision_constraint > log/wmt19_merge_3.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --merge 4 --device 1 --vision_constraint > log/wmt19_merge_4.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --merge 5 --device 0 --vision_constraint > log/wmt19_merge_5.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --merge 6 --device 1 --vision_constraint > log/wmt19_merge_6.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt19 --merge 7 --device 2 --vision_constraint > log/wmt19_merge_7.log 2>&1 &

nohup python get_adversarial_examples.py --dataset wmt18 --merge 1 --device 0 --vision_constraint > log/wmt18_merge_1.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --merge 2 --device 1 --vision_constraint > log/wmt18_merge_2.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --merge 3 --device 0 --vision_constraint > log/wmt18_merge_3.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --merge 4 --device 1 --vision_constraint > log/wmt18_merge_4.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --merge 5 --device 0 --vision_constraint > log/wmt18_merge_5.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --merge 6 --device 1 --vision_constraint > log/wmt18_merge_6.log 2>&1 &
nohup python get_adversarial_examples.py --dataset wmt18 --merge 7 --device 2 --vision_constraint > log/wmt18_merge_7.log 2>&1 &

nohup python get_adversarial_examples.py --dataset ted --merge 1 --device 0 --vision_constraint > log/ted_merge_1.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --merge 2 --device 1 --vision_constraint > log/ted_merge_2.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --merge 3 --device 0 --vision_constraint > log/ted_merge_3.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --merge 4 --device 1 --vision_constraint > log/ted_merge_4.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --merge 5 --device 0 --vision_constraint > log/ted_merge_5.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --merge 6 --device 1 --vision_constraint > log/ted_merge_6.log 2>&1 &
nohup python get_adversarial_examples.py --dataset ted --merge 7 --device 2 --vision_constraint > log/ted_merge_7.log 2>&1 &
