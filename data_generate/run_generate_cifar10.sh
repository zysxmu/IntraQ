for g in 1 2 3 4
do
python generate_data.py 		\
		--model=resnet20_cifar10 			\
		--batch_size=256 		\
		--test_batch_size=512 \
		--group=$g \
		--targetPro=0.95 \
		--cosineMargin=0.05 \
		--cosineMargin_upper=0.8 \
		--augMargin=0.5 \
		--save_path_head=hardsample
done
