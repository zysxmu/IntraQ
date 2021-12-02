for g in 1 2 3 4
do
python generate_data.py 		\
		--model=resnet18 			\
		--batch_size=256 		\
		--test_batch_size=512 \
		--group=$g \
		--targetPro=1 \
		--cosineMargin=0.0 \
		--cosineMargin_upper=2 \
		--augMargin=0.5 \
		--save_path_head=hardsample
done
